"""FairChem multi-tile XPU support for Aurora FLAT hierarchy.

Upstream ParallelMLIPPredictUnit assumes CUDA + Ray GPUs + NCCL.
This module patches device assignment and worker launch so each Ray worker
binds one PVC tile via ZE_AFFINITY_MASK and uses XCCL (else gloo).
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEVICE_PATCHED = False
_WORKER_PATCHED = False
_PARALLEL_PATCHED = False
_HenXPUMLIPWorker = None

# Phase-0 collective / predict stage timers (process-local; reset per SPE)
_stats_lock = threading.Lock()
_collective_stats: dict[str, float | int] = {
    "all_reduce_calls": 0,
    "all_reduce_s": 0.0,
    "all_gather_calls": 0,
    "all_gather_s": 0.0,
    "all_reduce_bytes": 0,
    "all_gather_bytes": 0,
    "predict_calls": 0,
    "predict_s": 0.0,
}


def reset_stage_timers() -> None:
    with _stats_lock:
        for k in _collective_stats:
            _collective_stats[k] = 0 if k.endswith("_calls") or k.endswith("_bytes") else 0.0


def get_stage_timers() -> dict[str, float | int]:
    with _stats_lock:
        return dict(_collective_stats)


def _acc(name: str, dt: float, nbytes: int = 0) -> None:
    with _stats_lock:
        _collective_stats[f"{name}_calls"] = int(_collective_stats[f"{name}_calls"]) + 1
        _collective_stats[f"{name}_s"] = float(_collective_stats[f"{name}_s"]) + dt
        if nbytes:
            key = f"{name}_bytes"
            if key in _collective_stats:
                _collective_stats[key] = int(_collective_stats[key]) + int(nbytes)


def _patch_edgewise_gather_inside_checkpoint() -> None:
    """Move GP Gather into the activation-checkpointed Edgewise region.

    Upstream ``Edgewise.forward`` runs
    ``gather_from_model_parallel_region_sum_grad`` *once outside* the
    ``torch.utils.checkpoint`` region (see FairChem ``escn_md_block.py``),
    then checkpoints ``forward_chunk`` with the already-gathered ``x_full``.

    That is fine for energy-only, but under XPU+gloo force autograd the
    Gather backward must run while all ranks recompute together. Leaving
    Gather outside the checkpoint yields vanishing / W-dependent forces
    (energies still match). Disabling AC fixes forces on small cells but
    OOMs 20³ (~17 GiB extra on the edge Wigner path).

    Fix: keep AC on; gather *inside* each checkpointed chunk so forward
    and backward-recompute both execute the collective on every rank.
    Checkpoint inputs then save the local shard (smaller than ``x_full``).
    """
    import torch
    from fairchem.core.common import gp_utils
    from fairchem.core.models.uma import escn_md_block as block_mod

    if getattr(block_mod, "_hen_edgewise_gather_in_ckpt_patched", False):
        return

    Edgewise = block_mod.Edgewise

    def forward(  # type: ignore[no-untyped-def]
        self,
        x,
        x_edge,
        edge_index,
        wigner,
        wigner_inv_envelope,
        total_atoms_across_gp_ranks,
        node_offset: int = 0,
    ):
        def _chunk_with_gather(  # type: ignore[no-untyped-def]
            x_local,
            total_atoms,
            x_original_shape,
            x_edge_chunk,
            edge_index_chunk,
            wigner_chunk,
            wigner_inv_chunk,
            node_off,
            ac_mole_start_idx,
        ):
            if gp_utils.initialized():
                x_full = gp_utils.gather_from_model_parallel_region_sum_grad(
                    x_local, total_atoms
                )
            else:
                x_full = x_local
            return self.forward_chunk(
                x_full,
                x_original_shape,
                x_edge_chunk,
                edge_index_chunk,
                wigner_chunk,
                wigner_inv_chunk,
                node_off,
                ac_mole_start_idx,
            )

        if self.activation_checkpoint_chunk_size is None:
            return _chunk_with_gather(
                x,
                total_atoms_across_gp_ranks,
                x.shape[0],
                x_edge,
                edge_index,
                wigner,
                wigner_inv_envelope,
                node_offset,
                0,
            )

        edge_index_partitions = edge_index.split(
            self.activation_checkpoint_chunk_size, dim=1
        )
        wigner_partitions = wigner.split(
            self.activation_checkpoint_chunk_size, dim=0
        )
        wigner_inv_partitions = wigner_inv_envelope.split(
            self.activation_checkpoint_chunk_size, dim=0
        )
        x_edge_partitions = x_edge.split(
            self.activation_checkpoint_chunk_size, dim=0
        )
        new_embeddings = []
        ac_mole_start_idx = 0
        x_n = x.shape[0]

        for idx in range(len(edge_index_partitions)):
            new_embeddings.append(
                torch.utils.checkpoint.checkpoint(
                    _chunk_with_gather,
                    x,
                    total_atoms_across_gp_ranks,
                    x_n,
                    x_edge_partitions[idx],
                    edge_index_partitions[idx],
                    wigner_partitions[idx],
                    wigner_inv_partitions[idx],
                    node_offset,
                    ac_mole_start_idx,
                    use_reentrant=False,
                )
            )
            ac_mole_start_idx += edge_index_partitions[idx].shape[1]
            if len(new_embeddings) > 8:
                new_embeddings = [torch.stack(new_embeddings).sum(axis=0)]
        return torch.stack(new_embeddings).sum(axis=0)

    Edgewise.forward = forward  # type: ignore[method-assign]
    block_mod._hen_edgewise_gather_in_ckpt_patched = True
    log.info(
        "HEN Edgewise patch: GP gather inside activation checkpoint "
        "(marker=hen_gather_in_ckpt_v1)"
    )


def _patch_force_autograd_for_gp() -> None:
    """Fix multi-tile force autograd (retain_graph for multi-head EFS).

    UMA checkpoints expose multiple ``MLP_EFS_Head`` modules. Each calls
    ``compute_forces`` with ``create_graph=False``, which frees the shared
    backbone graph after the first head. Under graph-parallel this interacts
    badly with Gather/checkpoint and yields vanishing or W-dependent garbage
    forces while energies (forward-only) stay correct.

    Always retain the graph through force autograd; still free on the last use
    via normal forward teardown.
    """
    import torch
    from fairchem.core.models.uma import outputs as uma_outputs

    _patch_edgewise_gather_inside_checkpoint()

    if getattr(uma_outputs, "_hen_force_autograd_patched", False):
        return

    def compute_forces(energy_part, pos, training: bool = True):  # type: ignore[no-untyped-def]
        (grad,) = torch.autograd.grad(
            energy_part.sum(),
            pos,
            create_graph=training,
            retain_graph=True,
        )
        forces = torch.neg(grad)
        if uma_outputs.gp_utils.initialized():
            forces = uma_outputs.gp_utils.reduce_from_model_parallel_region(forces)
        return forces

    def compute_forces_and_stress(  # type: ignore[no-untyped-def]
        energy_part, pos, cell, batch, training: bool = True
    ):
        grads = torch.autograd.grad(
            [energy_part.sum()],
            [pos, cell],
            create_graph=training,
            retain_graph=True,
        )
        if uma_outputs.gp_utils.initialized():
            grads = (
                uma_outputs.gp_utils.reduce_from_model_parallel_region(grads[0]),
                uma_outputs.gp_utils.reduce_from_model_parallel_region(grads[1]),
            )
        num_systems = cell.shape[0]
        forces = torch.neg(grads[0])
        pos_virial_per_atom = grads[0].unsqueeze(2) * pos.unsqueeze(1)
        pos_virial, _ = uma_outputs.reduce_node_to_system(
            pos_virial_per_atom, batch, num_systems
        )
        cell_virial = cell.mT @ grads[1]
        virial = (pos_virial + pos_virial.mT + cell_virial + cell_virial.mT) / 2
        volume = torch.det(cell).abs().unsqueeze(-1)
        stress = virial / volume.view(-1, 1, 1)
        stress = stress.view(-1, 9)
        return forces, stress

    uma_outputs.compute_forces = compute_forces  # type: ignore[assignment]
    uma_outputs.compute_forces_and_stress = compute_forces_and_stress  # type: ignore[assignment]
    # escn_md imported these by name at module load — rebind there too
    from fairchem.core.models.uma import escn_md as escn_md_mod

    escn_md_mod.compute_forces = compute_forces  # type: ignore[assignment]
    escn_md_mod.compute_forces_and_stress = compute_forces_and_stress  # type: ignore[assignment]

    uma_outputs._hen_force_autograd_patched = True
    log.info(
        "HEN force patch: compute_forces retain_graph=True "
        "(multi-head EFS + GP); marker=hen_force_retain_v2"
    )


def _patch_gp_utils_gloo_xpu() -> None:
    """gloo cannot collective XPU tensors; stage via CPU with autograd-safe ops.

    CPU staging lives *inside* FairChem Gather/Reduce Autograd Functions so the
    custom backward still owns the graph (detach only for the host copy).
    Also patches force autograd (retain_graph) — the previous Gather-only change
    was a no-op vs Phase-0 (bit-identical wrong forces).
    """
    import torch
    import torch.distributed as dist
    import torch.distributed.nn.functional as dist_nn_fn
    from fairchem.core.common import gp_utils

    if getattr(gp_utils, "_hen_gloo_xpu_patched", False):
        _patch_force_autograd_for_gp()
        return

    _cpu_ok = frozenset({"cpu", "cuda"})
    _orig_all_reduce = dist.all_reduce
    _orig_all_gather = dist.all_gather
    _orig_fn_all_reduce = dist_nn_fn.all_reduce

    def _nbytes(t: torch.Tensor) -> int:
        return int(t.numel() * t.element_size())

    def _cpu_all_reduce_numerical(tensor, op, group):  # type: ignore[no-untyped-def]
        """Out-of-place CPU-staged all_reduce (no Autograd Function). Detach only here."""
        if tensor.device.type in _cpu_ok:
            out = tensor.clone(memory_format=torch.contiguous_format)
            t0 = time.perf_counter()
            _orig_all_reduce(out, op=op, group=group, async_op=False)
            _acc("all_reduce", time.perf_counter() - t0, _nbytes(out))
            return out
        t0 = time.perf_counter()
        cpu = tensor.detach().to(device="cpu").contiguous()
        _orig_all_reduce(cpu, op=op, group=group, async_op=False)
        out = cpu.to(device=tensor.device, dtype=tensor.dtype)
        _acc("all_reduce", time.perf_counter() - t0, _nbytes(cpu))
        return out

    def _cpu_all_gather_numerical(input_tensor, group):  # type: ignore[no-untyped-def]
        """CPU-staged all_gather → tuple of device tensors (no Autograd Function)."""
        world = dist.get_world_size(group=group) if group is not None else dist.get_world_size()
        if input_tensor.device.type in _cpu_ok:
            tensor_list = [torch.empty_like(input_tensor) for _ in range(world)]
            t0 = time.perf_counter()
            _orig_all_gather(tensor_list, input_tensor, group=group, async_op=False)
            _acc("all_gather", time.perf_counter() - t0, _nbytes(input_tensor) * world)
            return tuple(tensor_list)
        device = input_tensor.device
        dtype = input_tensor.dtype
        t0 = time.perf_counter()
        cpu_in = input_tensor.detach().to(device="cpu").contiguous()
        cpu_out = [torch.empty_like(cpu_in) for _ in range(world)]
        _orig_all_gather(cpu_out, cpu_in, group=group, async_op=False)
        tensor_list = tuple(t.to(device=device, dtype=dtype) for t in cpu_out)
        _acc("all_gather", time.perf_counter() - t0, _nbytes(cpu_in) * world)
        return tensor_list

    class _GroupRef:
        __slots__ = ("group",)

        def __init__(self, group):  # type: ignore[no-untyped-def]
            self.group = group

    class _CPUStagedAllReduceFn(torch.autograd.Function):
        """Out-of-place all_reduce via CPU gloo; backward = same all_reduce (like PyTorch _AllReduce)."""

        @staticmethod
        @torch.compiler.disable
        def forward(ctx, tensor, op, group_ref):  # type: ignore[no-untyped-def]
            ctx.op = op
            ctx.group = group_ref.group
            return _cpu_all_reduce_numerical(tensor, op, ctx.group)

        @staticmethod
        @torch.compiler.disable
        def backward(ctx, grad_output):  # type: ignore[no-untyped-def]
            # Numerical only — already inside Autograd backward; do not nest .apply
            return _cpu_all_reduce_numerical(grad_output, ctx.op, ctx.group), None, None

    def _fn_all_reduce_xpu_safe(tensor, op=None, group=None):  # type: ignore[no-untyped-def]
        if op is None:
            op = dist.ReduceOp.SUM
        if tensor.device.type in _cpu_ok:
            # During another Function's backward, avoid nesting Autograd Function
            if not torch.is_grad_enabled():
                return _cpu_all_reduce_numerical(tensor, op, group)
            return _orig_fn_all_reduce(tensor, op=op, group=group)
        if not torch.is_grad_enabled():
            return _cpu_all_reduce_numerical(tensor, op, group)
        return _CPUStagedAllReduceFn.apply(tensor, op, _GroupRef(group))

    class ReduceFromModelParallelRegion(torch.autograd.Function):
        """Match FairChem: all_reduce in forward, identity backward (SUM all-reduce)."""

        @staticmethod
        @torch.compiler.disable
        def forward(ctx, input: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return _cpu_all_reduce_numerical(input, dist.ReduceOp.SUM, gp_utils.get_gp_group())

        @staticmethod
        @torch.compiler.disable
        def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override, no-untyped-def]
            return grad_output

    class GatherFromModelParallelRegionGradPadded(torch.autograd.Function):
        """all_gather in forward; backward returns local rank's grad chunk."""

        @staticmethod
        @torch.compiler.disable
        def forward(ctx, input: torch.Tensor):  # type: ignore[override, no-untyped-def]
            ctx.rank = gp_utils.get_gp_rank()
            ctx.group = gp_utils.get_gp_group()
            ctx.device = input.device
            ctx.dtype = input.dtype
            ctx.shape = input.shape
            return _cpu_all_gather_numerical(input, ctx.group)

        @staticmethod
        @torch.compiler.disable
        def backward(ctx, *grad_outputs):  # type: ignore[no-untyped-def]
            g = grad_outputs[ctx.rank]
            if g is None:
                return torch.zeros(ctx.shape, device=ctx.device, dtype=ctx.dtype)
            return g.contiguous()

    class GatherFromModelParallelRegionSumGradPadded(torch.autograd.Function):
        """FairChem gloo SumGrad: gather forward; cat grads → all_reduce → slice local.

        Marker class id logged so we can confirm Ray workers actually use this
        implementation (hen_gather_sumgrad_v2).
        """

        _hen_marker = "hen_gather_sumgrad_v2"

        @staticmethod
        @torch.compiler.disable
        def forward(ctx, input: torch.Tensor):  # type: ignore[override, no-untyped-def]
            ctx.rank = gp_utils.get_gp_rank()
            ctx.group = gp_utils.get_gp_group()
            ctx.shape = input.shape
            ctx.device = input.device
            ctx.dtype = input.dtype
            return _cpu_all_gather_numerical(input, ctx.group)

        @staticmethod
        @torch.compiler.disable
        def backward(ctx, *grad_outputs):  # type: ignore[no-untyped-def]
            # Materialize None grads (unused gathered chunks under checkpointing)
            padded = ctx.shape[0]
            for h in grad_outputs:
                if h is not None:
                    padded = h.shape[0]
                    break
            pieces = []
            for g in grad_outputs:
                if g is None:
                    pieces.append(
                        torch.zeros(
                            (padded, *ctx.shape[1:]),
                            device=ctx.device,
                            dtype=ctx.dtype,
                        )
                    )
                else:
                    pieces.append(g.contiguous())
            cat = torch.cat(pieces, dim=0)
            reduced = _cpu_all_reduce_numerical(cat, dist.ReduceOp.SUM, ctx.group)
            start = int(padded) * ctx.rank
            return reduced[start : start + ctx.shape[0]].contiguous()

    def _safe_all_reduce(tensor, op=None, group=None, async_op=False):  # type: ignore[no-untyped-def]
        """In-place hook for remaining dist.all_reduce call sites (not Gather/Reduce)."""
        if op is None:
            op = dist.ReduceOp.SUM
        if tensor.device.type in _cpu_ok:
            t0 = time.perf_counter()
            work = _orig_all_reduce(tensor, op=op, group=group, async_op=async_op)
            _acc("all_reduce", time.perf_counter() - t0, _nbytes(tensor))
            return work
        if async_op:
            raise RuntimeError("async all_reduce not supported for XPU+gloo CPU staging")
        # Never nest Autograd Function inside in-place dist.all_reduce: copy_ would
        # discard the Function output's grad_fn. Always stage numerically here;
        # GP Reduce/Gather Autograd Functions call _cpu_*_numerical directly.
        t0 = time.perf_counter()
        cpu = tensor.detach().to(device="cpu").contiguous()
        _orig_all_reduce(cpu, op=op, group=group, async_op=False)
        tensor.copy_(cpu.to(device=tensor.device, dtype=tensor.dtype))
        _acc("all_reduce", time.perf_counter() - t0, _nbytes(cpu))
        return None

    def _safe_all_gather(tensor_list, tensor, group=None, async_op=False):  # type: ignore[no-untyped-def]
        """Fallback for non-GP all_gather; GP Gather* classes stage internally."""
        if tensor.device.type in _cpu_ok:
            t0 = time.perf_counter()
            work = _orig_all_gather(tensor_list, tensor, group=group, async_op=async_op)
            _acc(
                "all_gather",
                time.perf_counter() - t0,
                _nbytes(tensor) * max(len(tensor_list), 1),
            )
            return work
        if async_op:
            raise RuntimeError("async all_gather not supported for XPU+gloo CPU staging")
        gathered = _cpu_all_gather_numerical(tensor, group)
        for dst, src in zip(tensor_list, gathered):
            dst.copy_(src)
        return None

    # Monkeypatch FairChem GP Autograd Functions (primary force-correctness path)
    gp_utils.ReduceFromModelParallelRegion = ReduceFromModelParallelRegion  # type: ignore[misc, assignment]
    gp_utils.GatherFromModelParallelRegionGradPadded = (  # type: ignore[misc, assignment]
        GatherFromModelParallelRegionGradPadded
    )
    gp_utils.GatherFromModelParallelRegionSumGradPadded = (  # type: ignore[misc, assignment]
        GatherFromModelParallelRegionSumGradPadded
    )

    dist.all_reduce = _safe_all_reduce  # type: ignore[assignment]
    dist.all_gather = _safe_all_gather  # type: ignore[assignment]
    dist_nn_fn.all_reduce = _fn_all_reduce_xpu_safe  # type: ignore[assignment]
    from fairchem.core.models.uma import escn_md as escn_md_mod

    escn_md_mod.all_reduce_with_grad = _fn_all_reduce_xpu_safe  # type: ignore[assignment]

    # gp_utils imported all_reduce by name from functional
    if hasattr(gp_utils, "all_reduce"):
        gp_utils.all_reduce = _fn_all_reduce_xpu_safe  # type: ignore[assignment]

    gp_utils._hen_gloo_xpu_patched = True
    _patch_force_autograd_for_gp()
    log.info(
        "Patched FairChem GP Gather/Reduce + collectives for XPU+gloo "
        "(marker=hen_gather_sumgrad_v2 id=%s)",
        id(GatherFromModelParallelRegionSumGradPadded),
    )


def patch_fairchem_xpu_device() -> None:
    """Minimal XPU acceptance for vanilla single-tile FairChem (no Ray / GP / collectives).

    Upstream ``MLIPPredictUnit._setup_device`` asserts ``device in ["cpu", "cuda"]``
    and ``set_seed`` calls ``torch.cuda.manual_seed_all``. This is the only patch
    needed for ``FAIRChemCalculator`` + ``MLIPPredictUnit(device="xpu")``.
    """
    global _DEVICE_PATCHED
    if _DEVICE_PATCHED:
        return

    import random

    import numpy as np
    import torch

    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit

    _orig_setup = MLIPPredictUnit._setup_device

    def _setup_device(self, device: str) -> None:  # type: ignore[no-untyped-def]
        key = str(device).strip().lower()
        if key == "xpu" or key.startswith("xpu:"):
            if not hasattr(torch, "xpu") or not torch.xpu.is_available():
                raise RuntimeError(f"requested {device!r} but torch.xpu is unavailable")
            self.device = "xpu"
            return
        return _orig_setup(self, device)

    def _set_seed(self, seed: int) -> None:  # type: ignore[no-untyped-def]
        logging.debug("Setting random seed to %s", seed)
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            torch.xpu.manual_seed_all(seed)

    MLIPPredictUnit._setup_device = _setup_device  # type: ignore[method-assign]
    MLIPPredictUnit.set_seed = _set_seed  # type: ignore[method-assign]
    _DEVICE_PATCHED = True
    log.info("HEN vanilla XPU device patch applied (no multi-GPU)")


def ensure_worker_patches() -> None:
    """Patch MLIPPredictUnit + distutils + worker _distributed_setup (safe in Ray workers)."""
    global _WORKER_PATCHED
    if _WORKER_PATCHED:
        return

    import torch
    import torch.distributed as dist

    from fairchem.core.common import distutils as distutils_mod
    from fairchem.core.units.mlip_unit import predict as predict_mod
    from fairchem.core.units.mlip_unit.predict import MLIPWorkerLocal

    patch_fairchem_xpu_device()
    _patch_gp_utils_gloo_xpu()

    def assign_device_for_local_rank(cpu: bool, local_rank: int) -> None:
        if cpu:
            os.environ[distutils_mod.CURRENT_DEVICE_TYPE_STR] = "cpu"
            return
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            os.environ[distutils_mod.CURRENT_DEVICE_TYPE_STR] = "xpu"
            torch.xpu.set_device(local_rank)
            return
        assert torch.cuda.is_available(), "no cuda/xpu available"
        os.environ[distutils_mod.CURRENT_DEVICE_TYPE_STR] = "cuda"
        torch.cuda.set_device(local_rank)

    def get_device_for_local_rank() -> str:
        cur = os.environ.get(distutils_mod.CURRENT_DEVICE_TYPE_STR)
        if cur is None:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                os.environ[distutils_mod.CURRENT_DEVICE_TYPE_STR] = (
                    f"xpu:{torch.xpu.current_device()}"
                )
            elif torch.cuda.is_available():
                os.environ[distutils_mod.CURRENT_DEVICE_TYPE_STR] = (
                    f"cuda:{torch.cuda.current_device()}"
                )
            else:
                os.environ[distutils_mod.CURRENT_DEVICE_TYPE_STR] = "cpu"
            logging.warning(
                "assign_device_for_local_rank was never called, defaulting to %s",
                os.environ[distutils_mod.CURRENT_DEVICE_TYPE_STR],
            )
            return os.environ[distutils_mod.CURRENT_DEVICE_TYPE_STR]

        if cur == "xpu" or str(cur).startswith("xpu"):
            assert hasattr(torch, "xpu") and torch.xpu.is_available()
            return f"xpu:{torch.xpu.current_device()}"
        if "cuda" in str(cur):
            assert torch.cuda.is_available()
            return f"cuda:{torch.cuda.current_device()}"
        if cur == "cpu":
            return "cpu"
        raise ValueError(f"unsupported device type: {cur}")

    distutils_mod.assign_device_for_local_rank = assign_device_for_local_rank
    distutils_mod.get_device_for_local_rank = get_device_for_local_rank
    predict_mod.assign_device_for_local_rank = assign_device_for_local_rank
    predict_mod.get_device_for_local_rank = get_device_for_local_rank

    def _distributed_setup(self) -> None:  # type: ignore[no-untyped-def]
        import hydra
        from fairchem.core.common import gp_utils

        logging.info("Initializing worker %s...", self.worker_id)
        distutils_mod.setup_env_local_multi_gpu(
            self.worker_id, self.master_port, self.master_address
        )

        device = str(self.predictor_config.get("device", "cpu")).strip().lower()
        os.environ.setdefault("ZE_FLAT_DEVICE_HIERARCHY", "FLAT")
        os.environ["ZE_AFFINITY_MASK"] = str(int(self.worker_id))

        if device == "xpu" or device.startswith("xpu:"):
            os.environ[distutils_mod.CURRENT_DEVICE_TYPE_STR] = "xpu"
            if not torch.xpu.is_available():
                raise RuntimeError(
                    f"worker {self.worker_id}: xpu unavailable under "
                    f"ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK')}"
                )
            torch.xpu.set_device(0)
            # Same-node Ray workers: gloo is reliable. XCCL/oneCCL needs ATL/OFI
            # providers that often fail under Ray without a PMI launcher.
            backend = os.environ.get("HEN_XPU_DIST_BACKEND", "gloo")
        elif device == "cpu":
            assign_device_for_local_rank(True, 0)
            backend = "gloo"
        else:
            assign_device_for_local_rank(False, 0)
            backend = "nccl"

        dist.init_process_group(
            backend=backend,
            rank=self.worker_id,
            world_size=self.world_size,
        )
        gp_utils.setup_graph_parallel_groups(self.world_size, backend)
        self.predict_unit = hydra.utils.instantiate(self.predictor_config)
        self.device = get_device_for_local_rank()
        from fairchem.core.common import gp_utils as _gpu
        from fairchem.core.models.uma import outputs as _uma_out

        gather_cls = getattr(_gpu, "GatherFromModelParallelRegionSumGradPadded", None)
        from fairchem.core.models.uma import escn_md_block as _escn_block

        logging.info(
            "Worker %s loaded predict unit on %s (backend=%s, mask=%s) "
            "hen_gather=%s id=%s force_patch=%s gather_in_ckpt=%s",
            self.worker_id,
            self.device,
            backend,
            os.environ.get("ZE_AFFINITY_MASK"),
            getattr(gather_cls, "_hen_marker", "MISSING"),
            id(gather_cls),
            getattr(_uma_out, "_hen_force_autograd_patched", False),
            getattr(_escn_block, "_hen_edgewise_gather_in_ckpt_patched", False),
        )
        self.is_setup = True

    MLIPWorkerLocal._distributed_setup = _distributed_setup  # type: ignore[method-assign]
    _WORKER_PATCHED = True


def _get_hen_worker_actor():
    """Lazily define Ray remote worker class (driver only)."""
    global _HenXPUMLIPWorker
    if _HenXPUMLIPWorker is not None:
        return _HenXPUMLIPWorker

    import ray
    from fairchem.core.units.mlip_unit.predict import MLIPWorkerLocal

    @ray.remote
    class HenXPUMLIPWorker(MLIPWorkerLocal):
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            ensure_worker_patches()
            super().__init__(*args, **kwargs)

    _HenXPUMLIPWorker = HenXPUMLIPWorker
    return _HenXPUMLIPWorker


def patch_fairchem_xpu_parallel() -> None:
    """Idempotent patches for multi-tile FairChem XPU inference (Ray + GP).

    Single-tile callers should use :func:`patch_fairchem_xpu_device` only.
    """
    global _PARALLEL_PATCHED
    ensure_worker_patches()
    if _PARALLEL_PATCHED:
        return

    import copy

    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    from fairchem.core.units.mlip_unit.predict import (
        MLIPPredictUnit,
        MLIPWorkerLocal,
        ParallelMLIPPredictUnit,
    )

    _orig_parallel_init = ParallelMLIPPredictUnit.__init__

    def _parallel_init(  # type: ignore[no-untyped-def]
        self,
        inference_model_path: str,
        device: str = "cpu",
        overrides: dict | None = None,
        inference_settings: Any = None,
        seed: int = 41,
        atom_refs: dict | None = None,
        form_elem_refs: dict | None = None,
        assert_on_nans: bool = False,
        num_workers: int = 1,
        num_workers_per_node: int = 8,
        log_level: int = logging.INFO,
    ) -> None:
        device_key = str(device).strip().lower()
        if not (device_key == "xpu" or device_key.startswith("xpu:")):
            return _orig_parallel_init(
                self,
                inference_model_path=inference_model_path,
                device=device,
                overrides=overrides,
                inference_settings=inference_settings,
                seed=seed,
                atom_refs=atom_refs,
                form_elem_refs=form_elem_refs,
                assert_on_nans=assert_on_nans,
                num_workers=num_workers,
                num_workers_per_node=num_workers_per_node,
                log_level=log_level,
            )

        if num_workers_per_node < num_workers:
            num_workers_per_node = num_workers

        os.environ.setdefault("ZE_FLAT_DEVICE_HIERARCHY", "FLAT")
        os.environ["ZE_AFFINITY_MASK"] = "0"

        hen_root = Path(__file__).resolve().parents[1]
        sqs_path = str(hen_root / "sqs_sampling")
        py_path = os.environ.get("PYTHONPATH", "")
        if sqs_path not in py_path.split(":"):
            os.environ["PYTHONPATH"] = sqs_path + (":" + py_path if py_path else "")

        _mlip_pred_unit = MLIPPredictUnit(
            inference_model_path=inference_model_path,
            device="cpu",
            overrides=overrides,
            inference_settings=inference_settings,
            seed=seed,
            atom_refs=atom_refs,
            form_elem_refs=form_elem_refs,
        )
        if inference_settings is None:
            inference_settings = InferenceSettings()
        self.inference_settings = inference_settings
        self._dataset_to_tasks = copy.deepcopy(_mlip_pred_unit.dataset_to_tasks)
        self._validate_atoms_data_fn = _mlip_pred_unit.model.module.validate_atoms_data

        predict_unit_config = {
            "_target_": "fairchem.core.units.mlip_unit.predict.MLIPPredictUnit",
            "inference_model_path": inference_model_path,
            "device": "xpu",
            "overrides": overrides,
            "inference_settings": inference_settings.to_omegaconf(),
            "seed": seed,
            "atom_refs": atom_refs,
            "form_elem_refs": form_elem_refs,
            "assert_on_nans": assert_on_nans,
        }

        logging.basicConfig(
            level=log_level,
            force=True,
            stream=sys.stdout,
            format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
        )
        logging.getLogger("ray").setLevel(log_level)

        # PBS TMPDIR paths are too long for AF_UNIX (107 byte limit).
        ray_tmp = Path("/tmp") / f"ray-{os.environ.get('USER', 'hen')}-{os.getpid()}"
        ray_tmp.mkdir(parents=True, exist_ok=True)
        os.environ["RAY_TMPDIR"] = str(ray_tmp)
        os.environ["TMPDIR"] = str(ray_tmp)

        if not ray.is_initialized():
            # Dashboard pulls broken opencensus in this env; not needed for inference.
            os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
            os.environ.setdefault("RAY_raylet_start_wait_time_s", "120")
            ray.init(
                logging_level=log_level,
                num_cpus=max(num_workers_per_node, num_workers),
                resources={"xpu_tile": float(num_workers)},
                _temp_dir=str(ray_tmp),
                include_dashboard=False,
            )

        self.atomic_data_on_device = None
        num_nodes = math.ceil(num_workers / num_workers_per_node)
        num_workers_on_node_array = [num_workers_per_node] * num_nodes
        if num_workers % num_workers_per_node > 0:
            num_workers_on_node_array[-1] = num_workers % num_workers_per_node

        placement_groups = []
        for workers in num_workers_on_node_array:
            bundle = {"CPU": workers, "xpu_tile": float(workers)}
            pg = placement_group([bundle], strategy="STRICT_PACK")
            placement_groups.append(pg)
        ray.get(placement_groups[0].ready())

        HenWorker = _get_hen_worker_actor()

        def _actor_opts(worker_id: int, pg):  # type: ignore[no-untyped-def]
            return HenWorker.options(
                num_cpus=1,
                num_gpus=0,
                resources={"xpu_tile": 1.0},
                runtime_env={
                    "env_vars": {
                        "ZE_FLAT_DEVICE_HIERARCHY": "FLAT",
                        "ZE_AFFINITY_MASK": str(worker_id),
                        "PYTHONPATH": os.environ.get("PYTHONPATH", sqs_path),
                    }
                },
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=0,
                    placement_group_capture_child_tasks=True,
                ),
            )

        _ = _actor_opts(0, placement_groups[0]).remote(
            0, num_workers, predict_unit_config
        )

        self.workers = []
        self.local_rank0 = MLIPWorkerLocal(
            worker_id=0,
            world_size=num_workers,
            predictor_config=predict_unit_config,
        )
        master_addr, master_port = self.local_rank0.get_master_address_and_port()
        logging.info("Started XPU rank0 on %s:%s", master_addr, master_port)

        worker_id = 0
        for pg_idx, pg in enumerate(placement_groups):
            workers = num_workers_on_node_array[pg_idx]
            for i in range(workers):
                if pg_idx == 0 and i == 0:
                    worker_id += 1
                    continue
                actor = _actor_opts(worker_id, pg).remote(
                    worker_id,
                    num_workers,
                    predict_unit_config,
                    master_port,
                    master_addr,
                )
                self.workers.append(actor)
                worker_id += 1

        log.info(
            "Parallel XPU predict unit: workers=%s tiles=0..%s",
            num_workers,
            num_workers - 1,
        )

    ParallelMLIPPredictUnit.__init__ = _parallel_init  # type: ignore[method-assign]
    _PARALLEL_PATCHED = True
    log.info("FairChem XPU parallel patches applied")
