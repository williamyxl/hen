"""PYTHONPATH shim for ``fairchem_xpu_parallel`` (do not edit sqs_sampling).

Put ``hen/shim`` *first* on PYTHONPATH. This file ``exec``s the real
``sqs_sampling/fairchem_xpu_parallel.py`` into *this* module's globals so Ray
actors pickle/import a single ``fairchem_xpu_parallel`` (no alias module),
then:

1. Replaces ``_patch_edgewise_gather_inside_checkpoint`` with Phase 1
   patches: ``fxpu_ckpt_seq_v6`` (Edgewise) **and** ``edeg_v1``
   (EdgeDegreeEmbedding).
2. Phase 2: when ``FXPU_DIST_BACKEND=xccl``, skip CPU staging and use
   native XPU collectives (+ ATL/OFI env for Ray).
3. Phase 3: launch / affinity — no CPU double-load, no discarded rank0
   actor, eager distributed setup, ZE_AFFINITY before torch.xpu.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_REAL = _PROJECT_ROOT / "sqs_sampling" / "fairchem_xpu_parallel.py"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Execute the real implementation into this module namespace.
_code = _REAL.read_text(encoding="utf-8")
exec(compile(_code, str(_REAL), "exec"), globals())  # noqa: S102


def _patch_edgewise_gather_inside_checkpoint() -> None:
    """Override Edgewise + EdgeDegreeEmbedding AC with Phase-1 fixes."""
    from patches.phase1_force_correctness import apply_all_phase1_patches

    apply_all_phase1_patches()


# In-module callers look this name up in globals() at call time.
globals()["_patch_edgewise_gather_inside_checkpoint"] = (
    _patch_edgewise_gather_inside_checkpoint
)

_orig_patch_fairchem_xpu_device = globals()["patch_fairchem_xpu_device"]


def patch_fairchem_xpu_device() -> None:
    """XPU device allowlist + faithful prepare_wigner edge-chunk workaround.

    Large-edge ``torch.einsum("mk,nkj->nmj", to_m, wigner)`` on Intel XPU FP64
    yields wrong reverse-mode forces (NaCl N≥10 AG≠FD). Chunking that
    contraction along edges restores AG≡FD without changing UMA architecture.
    Disable with ``FXPU_SKIP_WIGNER_PREP_CHUNK=1``.
    """
    _orig_patch_fairchem_xpu_device()
    skip = os.environ.get("FXPU_SKIP_WIGNER_PREP_CHUNK", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if skip:
        return
    from patches.xpu_prepare_wigner import apply_xpu_prepare_wigner_chunking

    apply_xpu_prepare_wigner_chunking()


globals()["patch_fairchem_xpu_device"] = patch_fairchem_xpu_device

_orig_patch_force_autograd_for_gp = globals()["_patch_force_autograd_for_gp"]

def _patch_force_autograd_for_gp() -> None:
    """retain_graph force patch + optional pre-reduce force dump (in one install).

    Dump must live *inside* this function: a later re-apply of the stock
    retain_graph patch otherwise overwrites a post-hoc dump wrapper, and
    ``escn_md``'s ``from outputs import compute_forces`` binding only updates
    when we assign ``escn_md.compute_forces`` here.
    """
    import logging
    import numpy as np
    import torch
    from fairchem.core.models.uma import outputs as uma_outputs
    from fairchem.core.models.uma import escn_md as escn_md_mod

    # Always refresh Edgewise/edeg Phase1 first (idempotent).
    globals()["_patch_edgewise_gather_inside_checkpoint"]()

    dump = os.environ.get("FXPU_DUMP_PREREDUCE_FORCES", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    dump_dir = os.environ.get("FXPU_DUMP_PREREDUCE_DIR", "").strip()
    if dump:
        if not dump_dir:
            dump_dir = str(
                _PROJECT_ROOT
                / "pbs/out/parity_nacl_18/fix_probes/AI_prereduce/prereduce"
            )
        dump_dir = str(Path(dump_dir).resolve())
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        Path(dump_dir, f"dump_patch_installed_pid{os.getpid()}.txt").write_text(
            f"pid={os.getpid()} dump_dir={dump_dir}\n", encoding="utf-8"
        )
    else:
        dump_dir = ""

    def _dump_local(forces_local, tag: str) -> None:
        if not dump:
            return
        try:
            from fairchem.core.common import gp_utils

            rank = gp_utils.get_gp_rank() if gp_utils.initialized() else 0
        except Exception:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        Path(dump_dir, f"called_{tag}_rank{rank:02d}_pid{os.getpid()}.txt").write_text(
            f"tag={tag} rank={rank} shape={tuple(forces_local.shape)}\n",
            encoding="utf-8",
        )
        path = str(Path(dump_dir) / f"prereduce_rank{rank:02d}.npy")
        np.save(path, forces_local.detach().float().cpu().numpy())

    def compute_forces(energy_part, pos, training: bool = True):  # type: ignore[no-untyped-def]
        (grad,) = torch.autograd.grad(
            energy_part.sum(),
            pos,
            create_graph=training,
            retain_graph=True,
        )
        forces_local = torch.neg(grad)
        _dump_local(forces_local, "forces")
        if uma_outputs.gp_utils.initialized():
            forces = uma_outputs.gp_utils.reduce_from_model_parallel_region(
                forces_local
            )
        else:
            forces = forces_local
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
        forces_local = torch.neg(grads[0])
        _dump_local(forces_local, "forces_stress")
        if uma_outputs.gp_utils.initialized():
            grads = (
                uma_outputs.gp_utils.reduce_from_model_parallel_region(grads[0]),
                uma_outputs.gp_utils.reduce_from_model_parallel_region(grads[1]),
            )
            forces = torch.neg(grads[0])
        else:
            forces = forces_local
        num_systems = cell.shape[0]
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

    compute_forces._fxpu_is_dump_prereduce = bool(dump)  # type: ignore[attr-defined]
    uma_outputs.compute_forces = compute_forces  # type: ignore[assignment]
    uma_outputs.compute_forces_and_stress = compute_forces_and_stress  # type: ignore[assignment]
    escn_md_mod.compute_forces = compute_forces  # type: ignore[assignment]
    escn_md_mod.compute_forces_and_stress = compute_forces_and_stress  # type: ignore[assignment]
    # Rebind any already-imported fairchem module globals that hold compute_forces.
    for _mod in list(sys.modules.values()):
        if _mod is None:
            continue
        try:
            if getattr(_mod, "compute_forces", None) is not None and "fairchem" in getattr(
                _mod, "__name__", ""
            ):
                _mod.compute_forces = compute_forces  # type: ignore[attr-defined]
            if getattr(_mod, "compute_forces_and_stress", None) is not None and "fairchem" in getattr(
                _mod, "__name__", ""
            ):
                _mod.compute_forces_and_stress = compute_forces_and_stress  # type: ignore[attr-defined]
        except Exception:
            pass
    uma_outputs._fxpu_force_autograd_patched = True
    logging.getLogger(__name__).info(
        "FXPU force patch: retain_graph=True dump=%s dir=%s", dump, dump_dir or "-"
    )
    # Pos-only stress path must win over retain's joint compute_forces_and_stress.
    from patches.phase1_force_correctness import apply_force_no_stress

    apply_force_no_stress()


globals()["_patch_force_autograd_for_gp"] = _patch_force_autograd_for_gp

# --- Phase 2: XCCL path (override gloo CPU staging when backend=xccl) ---
_orig_patch_gp_utils_gloo_xpu = globals()["_patch_gp_utils_gloo_xpu"]
_orig_ensure_worker_patches = globals()["ensure_worker_patches"]
_orig_patch_fairchem_xpu_parallel = globals()["patch_fairchem_xpu_parallel"]
_PHASE2_WORKER_WRAPPED = False


def _patch_gp_utils_gloo_xpu() -> None:
    """gloo → CPU staging; xccl → native XPU collectives (no host staging)."""
    from patches import phase2_xccl as p2

    if p2.use_xccl():
        p2.install_init_process_group_hook()
        p2.apply_native_xccl_collectives(
            acc_fn=globals()["_acc"],
            force_patch_fn=globals()["_patch_force_autograd_for_gp"],
        )
        return
    _orig_patch_gp_utils_gloo_xpu()


def ensure_worker_patches() -> None:
    global _PHASE2_WORKER_WRAPPED
    from patches import phase2_xccl as p2

    if p2.use_xccl():
        p2.configure_xccl_env_for_ray()
        p2.install_init_process_group_hook()

    _orig_ensure_worker_patches()
    # Stock gloo path early-returns on second call without reinstalling force
    # when already patched; force a refresh so dump+retain stay in sync.
    globals()["_patch_force_autograd_for_gp"]()

    if _PHASE2_WORKER_WRAPPED:
        return

    # Stock _distributed_setup sets ZE_AFFINITY_MASK=str(worker_id) (global
    # rank) and setup_env_local_multi_gpu sets LOCAL_RANK=global. On multi-node
    # that becomes 12..23 on node2 and poisons device/CCL init.
    # CRITICAL: os.environ[k]=v uses Environ.__setitem__ (class), NOT the
    # instance method — an instance monkeypatch is a no-op (verified py3.13).
    from fairchem.core.units.mlip_unit.predict import MLIPWorkerLocal

    _inner = MLIPWorkerLocal._distributed_setup

    def _distributed_setup(self) -> None:  # type: ignore[no-untyped-def]
        import logging

        wid = int(self.worker_id)
        world = int(self.world_size)
        try:
            from patches.phase6_multinode import tiles_per_node as _tpn

            per = int(_tpn())
        except Exception:  # noqa: BLE001
            per = int(os.environ.get("FXPU_TILES_PER_NODE", "12") or 12)
        multinode = world > per or os.environ.get(
            "FXPU_PHASE6_MULTINODE", ""
        ).strip().lower() in ("1", "true", "on", "yes")
        if multinode:
            local_rank = wid % per
            local_size = per
        else:
            local_rank = wid
            local_size = world
        if p2.use_xccl():
            p2.scrub_mpi_launcher_env()
            p2.configure_xccl_env_for_ray(
                local_rank=local_rank,
                local_size=local_size,
                world_size=world,
            )
            os.environ["CCL_PROCESS_LAUNCHER"] = "none"

        Environ = type(os.environ)
        orig_cls_setitem = Environ.__setitem__

        def _guarded_cls_setitem(self_env, key, value):  # type: ignore[no-untyped-def]
            k = str(key)
            if k == "ZE_AFFINITY_MASK":
                value = str(local_rank)
            elif multinode and k in ("LOCAL_RANK", "CCL_LOCAL_RANK"):
                value = str(local_rank)
            elif multinode and k in ("LOCAL_WORLD_SIZE", "CCL_LOCAL_SIZE"):
                value = str(local_size)
            return orig_cls_setitem(self_env, k, value)

        try:
            Environ.__setitem__ = _guarded_cls_setitem  # type: ignore[method-assign]
            os.environ["ZE_AFFINITY_MASK"] = str(local_rank)
            os.environ["LOCAL_RANK"] = str(local_rank)
            os.environ["CCL_LOCAL_RANK"] = str(local_rank)
            os.environ["LOCAL_WORLD_SIZE"] = str(local_size)
            os.environ["CCL_LOCAL_SIZE"] = str(local_size)
            os.environ["WORLD_SIZE"] = str(world)
            os.environ["RANK"] = str(wid)
            out = _inner(self)
        finally:
            Environ.__setitem__ = orig_cls_setitem  # type: ignore[method-assign]
            os.environ["ZE_AFFINITY_MASK"] = str(local_rank)
            os.environ["LOCAL_RANK"] = str(local_rank)
            os.environ["CCL_LOCAL_RANK"] = str(local_rank)
            os.environ["LOCAL_WORLD_SIZE"] = str(local_size)
            os.environ["CCL_LOCAL_SIZE"] = str(local_size)
        logging.getLogger(__name__).info(
            "FXPU Phase2/6 affinity guard: worker=%s tile=%s/%s mask=%s "
            "(multinode=%s backend=%s FI=%s)",
            wid,
            local_rank,
            local_size,
            os.environ.get("ZE_AFFINITY_MASK"),
            multinode,
            os.environ.get("FXPU_DIST_BACKEND", "gloo"),
            os.environ.get("FI_PROVIDER"),
        )
        if multinode:
            try:
                mask_i = int(os.environ.get("ZE_AFFINITY_MASK", "-1"))
            except ValueError:
                mask_i = -1
            if not (0 <= mask_i < local_size):
                raise RuntimeError(
                    f"FXPU affinity guard: worker={wid} ZE_AFFINITY_MASK="
                    f"{os.environ.get('ZE_AFFINITY_MASK')!r} not in "
                    f"[0,{local_size}) after _distributed_setup"
                )
        return out

    MLIPWorkerLocal._distributed_setup = _distributed_setup  # type: ignore[method-assign]
    _PHASE2_WORKER_WRAPPED = True


def patch_fairchem_xpu_parallel() -> None:
    """Multi-tile patches; inject XCCL env; Phase 3 launch overrides."""
    from patches import phase2_xccl as p2
    from patches import phase3_launch as p3

    # Multi-node Ray+XCCL: apply launcher=none / PMI scrub; optional demote via
    # FXPU_DEMOTE_MULTINODE_XCCL=1 (default keeps xccl — proven PASS 8729300).
    p2.maybe_demote_multinode_xccl_to_gloo()

    if p2.use_xccl():
        p2.configure_xccl_env_for_ray()
        # Merge XCCL env into process env before Ray forks workers.
        for k, v in p2.ray_worker_xccl_env_vars().items():
            os.environ.setdefault(k, v)

    _orig_patch_fairchem_xpu_parallel()

    import logging

    _log = logging.getLogger(__name__)
    if p2.use_xccl():
        _log.info(
            "FXPU Phase2: patch_fairchem_xpu_parallel with XCCL "
            "(FI_PROVIDER=%s CCL_ATL_TRANSPORT=%s)",
            os.environ.get("FI_PROVIDER"),
            os.environ.get("CCL_ATL_TRANSPORT"),
        )

    # Phase 3 replaces ParallelMLIPPredictUnit.__init__ (keeps Phase1/2 patches).
    marker = p3.apply_launch_patches()
    _log.info("FXPU Phase3: launch patches marker=%s", marker)

    # Phase 6: multi-node wrapper (delegates to Phase3 when W ≤ tiles/node).
    from patches import phase6_multinode as p6

    _p6flag = os.environ.get("FXPU_PHASE6_MULTINODE", "auto").strip().lower()
    if _p6flag not in ("0", "false", "off", "no"):
        m6 = p6.apply_phase6_multinode_launch()
        _log.info("FXPU Phase6: multi-node launch marker=%s", m6)


globals()["_patch_gp_utils_gloo_xpu"] = _patch_gp_utils_gloo_xpu
globals()["ensure_worker_patches"] = ensure_worker_patches
globals()["patch_fairchem_xpu_parallel"] = patch_fairchem_xpu_parallel
