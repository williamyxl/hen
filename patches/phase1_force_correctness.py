"""Phase 1 force-correctness patch (external to sqs_sampling).

Root cause
----------
Upstream ``Edgewise.forward`` gathers once, then passes the **same** ``x_full``
into many ``torch.utils.checkpoint(..., use_reentrant=False)`` chunks.
Non-reentrant checkpoint corrupts grads for shared inputs → energies OK,
forces W-dependent garbage (cos≈0). Zero-copy ``view`` forks (v4) still share
storage → 20³ residual ΔF (cos~0.95–0.98).

``EdgeDegreeEmbedding.forward`` has the same bug class on a **chained** ``x``:
each ``forward_chunk`` does ``x.index_add(...)`` and the next checkpoint takes
that updated ``x``. Edgewise-only fixes leave a bit-identical residual
(Phase 1g ≈ 1i).

Failed attempts
---------------
- ``fxpu_ckpt_clone_v1``: clone per fine chunk — 6³ PASS, **OOM 20³**
- ``fxpu_ckpt_single_v2`` / ``reentrant_v3``: OOM or incompatible with
  ``torch.autograd.grad`` forces
- ``fxpu_ckpt_fork_v4``: 6³ PASS; 20³ residual shared-storage error
- ``fxpu_ckpt_seq_v5``: custom AC saved **only** ``x_full`` (closured
  ``x_edge``/wigner) → **wrong forces** (6³ Fmax~0.067 vs ~0.528); also
  OOM W=1/2 on 20³
- ``fxpu_ckpt_bisect_v6`` N=2 one-ckpt-per-half: 6³ PASS (Fmax 0.528) but
  20³ **OOM** recomputing ~half the edges under grad
- ``fxpu_ckpt_seq_v6`` (+ bisect2): Edgewise fixed; 6³ PASS; 20³ residual
  identical to fork_v4 → EdgeDegreeEmbedding still broken

Fix (marker=fxpu_ckpt_seq_v6[+bisectN]+edeg_v1)
-----------------------------------------------
Edgewise: custom ``autograd.Function`` — forward under ``no_grad`` sum of fine
chunks; ``save_for_backward`` one ``x_full`` + all partition tensors;
backward recomputes one fine chunk at a time.

EdgeDegreeEmbedding (``edeg_v1``): sequential **chain** AC — forward chains
``forward_chunk`` under ``no_grad``; saves initial ``x`` + all partitions;
backward walks chunks reverse, recomputing prefixes under ``no_grad`` and one
chunk with grad (compatible with ``torch.autograd.grad``; no
``use_reentrant=True``).

Optional ``FXPU_EDGEWISE_AC_BISECT=2..4``: N coarse groups, each with
``x_full.clone()`` **and** cloned floating partitions (views of ``x_edge``/
wigner must not be shared across separate AC Functions — that reproduced the
fork_v4 residual). Each group still uses sequential fine recompute.

Diagnostic / fix knobs (18³ multi-tile Fz residual):
- ``FXPU_EDGEWISE_GATHER_IN_AC=1``: GP gather inside SeqChunkAC forward/recompute
  (collective-aligned; evolves frozen gather-in-ckpt idea).
- ``FXPU_EDGEWISE_ONE_CHUNK=1``: disable edge splits (single AC partition).

Delivery: ``hen/shim`` first on PYTHONPATH.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

MARKER = "fxpu_ckpt_seq_v6"
EDEG_MARKER = "edeg_v1"
_APPLIED_EDGEWISE = False
_APPLIED_EDEG = False


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_project_pythonpath() -> None:
    root = _project_root()
    shim = root / "shim"
    sqs = root / "sqs_sampling"
    parts = [str(shim), str(root), str(sqs)]
    cur = os.environ.get("PYTHONPATH", "")
    existing = [p for p in cur.split(":") if p]
    merged: list[str] = []
    for p in parts + existing:
        if p and p not in merged:
            merged.append(p)
    os.environ["PYTHONPATH"] = ":".join(merged)
    for p in reversed(parts):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


def _bisect_n() -> int:
    """Optional coarse groups (each with real x clone + seq fine AC).

    Default 0 = single sequential region over all fine chunks (best memory).
    Set ``FXPU_EDGEWISE_AC_BISECT=2`` (or 3–4) for N unique clones.
    """
    raw = os.environ.get("FXPU_EDGEWISE_AC_BISECT", "").strip()
    if not raw or raw.lower() in ("0", "off", "false", "seq"):
        return 0
    try:
        n = int(raw)
    except ValueError:
        return 0
    return n if n >= 2 else 0


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _gather_in_ac() -> bool:
    """If set, GP gather runs inside each SeqChunkAC forward/recompute.

    Historical vanishing-force fix (gather-in-ckpt) but with seq AC instead of
    ``torch.utils.checkpoint(use_reentrant=False)``. Needed when gather-outside
    + multi-chunk AC corrupts Cartesian force components under GP.
    """
    return _env_flag("FXPU_EDGEWISE_GATHER_IN_AC")


def _one_chunk() -> bool:
    """Force a single AC partition (no edge splits) — diagnostic / possible fix."""
    return _env_flag("FXPU_EDGEWISE_ONE_CHUNK")


def _gamma_zero() -> bool:
    """Force Wigner SO(2) gamma=0 (disable torch.rand roll)."""
    return _env_flag("FXPU_WIGNER_GAMMA_ZERO")


def _clone_parts() -> bool:
    """Clone floating edge/wigner partitions even for single-seq AC (GP stress test)."""
    return _env_flag("FXPU_EDGEWISE_CLONE_PARTS")


def _hold_mole_idx() -> bool:
    """Keep MoLE ac_start_idx set through autograd.grad recompute."""
    return _env_flag("FXPU_EDGEWISE_HOLD_MOLE_IDX")


def _ac_contig() -> bool:
    """Save full edge/wigner tensors and slice inside SeqChunkAC (no multi-view inputs).

    Default ON (hygiene). Probe K (18³ W=2) was **bit-identical** to seq A —
    multi-view ``wigner.split()`` inputs are **not** the Fz root cause.
    Set ``FXPU_EDGEWISE_AC_CONTIG=0`` to restore legacy multi-view inputs.
    """
    raw = os.environ.get("FXPU_EDGEWISE_AC_CONTIG", "1").strip().lower()
    return raw not in ("0", "off", "false", "no")


def _detach_wigner() -> bool:
    """Detach Wigner matrices so force grads skip the edge→Y Jacobian (diag)."""
    return _env_flag("FXPU_DETACH_WIGNER")


def _detach_envelope() -> bool:
    """Detach polynomial envelope so force grads skip envelope→pos (diag)."""
    return _env_flag("FXPU_DETACH_ENVELOPE")


def _detach_edge_dist() -> bool:
    """Detach Gaussian distance embedding (cut scalar edge_distance→radial grads)."""
    return _env_flag("FXPU_DETACH_EDGE_DIST")


def _so2_radial_outside_ac() -> bool:
    """Precompute SO2 rad_func outside Edgewise multi-chunk AC (diag/fix)."""
    return _env_flag("FXPU_SO2_RADIAL_OUTSIDE_AC")


def _n2e_clone() -> bool:
    """Clone after node_to_edge_wigner_permute (diag)."""
    return _env_flag("FXPU_N2E_CLONE")


def _detach_after_mp() -> bool:
    """Banned: zeros Edgewise (architecture change). Refuses if env set."""
    if _env_flag("FXPU_DETACH_AFTER_MP"):
        raise RuntimeError(
            "FXPU_DETACH_AFTER_MP is forbidden: it replaces Edgewise with zeros "
            "(UMA architecture change). Hardware scaling requires the full stock model."
        )
    return False


def _edgevec_safe_backward() -> bool:
    return _env_flag("FXPU_EDGEVEC_SAFE_BACKWARD")


def _pbc_out_of_place() -> bool:
    return _env_flag("FXPU_PBC_OUT_OF_PLACE")


def _edgevec_index_select() -> bool:
    return _env_flag("FXPU_EDGEVEC_INDEX_SELECT")


def _scatter_safe_backward() -> bool:
    """Replace ExecutionBackend index_add scatter with explicit Autograd Function."""
    return _env_flag("FXPU_SCATTER_SAFE_BACKWARD")


def _edgevec_chunk_size() -> int | None:
    """If set, build edge_distance_vec in chunks of this many edges."""
    raw = os.environ.get("FXPU_EDGEVEC_CHUNK_SIZE", "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def _edgewise_max_layers() -> int | None:
    """Banned: truncates Edgewise depth (architecture change). Refuses if env set."""
    raw = os.environ.get("FXPU_EDGEWISE_MAX_LAYERS", "").strip()
    if raw:
        raise RuntimeError(
            "FXPU_EDGEWISE_MAX_LAYERS is forbidden: truncating Edgewise layers changes "
            "UMA architecture. Full-depth only; if OOM, document failure — do not shrink the model."
        )
    return None


def _no_xedge_grad() -> bool:
    """Omit x_edge from SeqChunkAC grad_targets (cut radial path in Edgewise AC)."""
    return _env_flag("FXPU_SEQAC_NO_XEDGE_GRAD")


def _no_winv_grad() -> bool:
    """Omit wigner_inv from SeqChunkAC grad_targets (cut winv/envelope path)."""
    return _env_flag("FXPU_SEQAC_NO_WINV_GRAD")


def _no_wig_grad() -> bool:
    """Omit wigner from SeqChunkAC grad_targets."""
    return _env_flag("FXPU_SEQAC_NO_WIG_GRAD")


def _no_x_grad() -> bool:
    """Omit node x from SeqChunkAC grad_targets (cut grad_x / scatter path)."""
    return _env_flag("FXPU_SEQAC_NO_X_GRAD")


def _skip_edeg() -> bool:
    """Skip EdgeDegreeEmbedding Phase1 patch (use upstream edeg)."""
    return _env_flag("FXPU_SKIP_EDEG_PATCH")


def _skip_edgewise() -> bool:
    """Skip Edgewise Phase1 SeqChunkAC (use upstream torch.utils.checkpoint)."""
    return _env_flag("FXPU_SKIP_EDGEWISE_PATCH")


def _scatter_clone() -> bool:
    """Clone tensors before index_add scatter (diag hygiene under GP)."""
    return _env_flag("FXPU_SCATTER_CLONE")


def _dump_prereduce_forces() -> bool:
    """Dump per-rank ∂E_part/∂pos before GP force all_reduce (diag)."""
    return _env_flag("FXPU_DUMP_PREREDUCE_FORCES")


def _force_no_stress() -> bool:
    """Force E+F via compute_forces only (skip compute_forces_and_stress)."""
    return _env_flag("FXPU_FORCE_NO_STRESS")


def _edeg_one_chunk() -> bool:
    """Force EdgeDegreeEmbedding to a single partition (Edgewise unchanged)."""
    return _env_flag("FXPU_EDEG_ONE_CHUNK") or _one_chunk()


def _chunk_size_override() -> int | None:
    """Optional ``FXPU_EDGEWISE_CHUNK_SIZE`` override for AC edge splits."""
    raw = os.environ.get("FXPU_EDGEWISE_CHUNK_SIZE", "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


def _effective_chunk_size(module_chunk: int | None) -> int | None:
    if _one_chunk():
        return None
    ov = _chunk_size_override()
    if ov is not None:
        return ov
    return module_chunk


def _edgewise_marker() -> str:
    n = _bisect_n()
    base = f"fxpu_ckpt_seq_v6_bisect{n}" if n >= 2 else "fxpu_ckpt_seq_v6"
    if _gather_in_ac():
        base = f"{base}+gatherAC"
    if _one_chunk():
        base = f"{base}+1chunk"
    elif _chunk_size_override() is not None:
        base = f"{base}+chunk{_chunk_size_override()}"
    if _gamma_zero():
        base = f"{base}+g0"
    if _clone_parts():
        base = f"{base}+cloneParts"
    if _hold_mole_idx():
        base = f"{base}+holdMole"
    if _ac_contig():
        base = f"{base}+contig"
    if _detach_wigner():
        base = f"{base}+detachW"
    if _detach_envelope():
        base = f"{base}+detachEnv"
    if _detach_edge_dist():
        base = f"{base}+detachEdist"
    if _so2_radial_outside_ac():
        base = f"{base}+so2RadOut"
    if _n2e_clone():
        base = f"{base}+n2eClone"
    if _detach_after_mp():
        base = f"{base}+detachMP"
    if _edgewise_max_layers() is not None:
        base = f"{base}+maxL{_edgewise_max_layers()}"
    if _no_xedge_grad():
        base = f"{base}+noXedge"
    if _no_winv_grad():
        base = f"{base}+noWinv"
    if _no_wig_grad():
        base = f"{base}+noWig"
    if _no_x_grad():
        base = f"{base}+noX"
    if _skip_edgewise():
        base = f"{base}+upstreamEW"
    if _scatter_clone():
        base = f"{base}+scatterClone"
    if _dump_prereduce_forces():
        base = f"{base}+dumpPreF"
    if _force_no_stress():
        base = f"{base}+noStress"
    if _edgevec_safe_backward():
        base = f"{base}+edgeSafe"
    if _pbc_out_of_place():
        base = f"{base}+pbcOop"
    if _edgevec_index_select():
        base = f"{base}+idxSel"
    return base


def effective_marker() -> str:
    """Combined Edgewise + EdgeDegreeEmbedding marker."""
    return f"{_edgewise_marker()}+{EDEG_MARKER}"

def _run_parts_no_grad(self, x_in, x_n, node_offset, parts):
    import torch

    new_embeddings = []
    for x_edge_p, ei_p, wig_p, winv_p, mole_idx in parts:
        new_embeddings.append(
            self.forward_chunk(
                x_in,
                x_n,
                x_edge_p,
                ei_p,
                wig_p,
                winv_p,
                node_offset,
                mole_idx,
            )
        )
        if len(new_embeddings) > 8:
            new_embeddings = [torch.stack(new_embeddings).sum(axis=0)]
    return torch.stack(new_embeddings).sum(axis=0)


def _make_seq_ac(
    self,
    x_n: int,
    node_offset: int,
    parts: list,
    *,
    total_atoms: int | None = None,
    gather: bool = False,
):
    """Build a sequential AC Function for one list of fine parts.

    If ``gather`` is True, ``x_in`` is the *local* GP shard and each
    forward/recompute calls ``gather_from_model_parallel_region_sum_grad``
    before ``forward_chunk`` (all ranks stay collective-aligned).
    """
    import torch
    from fairchem.core.common import gp_utils

    mole_indices = [int(m) for *_, m in parts]
    n_parts = len(parts)
    gather_natoms = int(total_atoms) if gather else 0

    class _SeqChunkAC(torch.autograd.Function):
        """Save x + all partition tensors; recompute one fine chunk at a time."""

        @staticmethod
        @torch.compiler.disable
        def forward(ctx, x_in, *flat):  # type: ignore[no-untyped-def]
            # flat: n_parts × (x_edge, ei, wig, winv)
            ctx.x_n = x_n
            ctx.node_offset = node_offset
            ctx.n_parts = n_parts
            ctx.mole = mole_indices
            ctx.gather = gather
            ctx.gather_natoms = gather_natoms
            ctx.save_for_backward(x_in, *flat)
            acc = None
            with torch.no_grad():
                for i in range(n_parts):
                    base = 4 * i
                    x_use = x_in
                    if gather:
                        x_use = gp_utils.gather_from_model_parallel_region_sum_grad(
                            x_in, gather_natoms
                        )
                    out = self.forward_chunk(
                        x_use,
                        x_n,
                        flat[base],
                        flat[base + 1],
                        flat[base + 2],
                        flat[base + 3],
                        node_offset,
                        mole_indices[i],
                    )
                    acc = out if acc is None else acc + out
            return acc

        @staticmethod
        @torch.compiler.disable
        def backward(ctx, grad_out):  # type: ignore[no-untyped-def]
            saved = ctx.saved_tensors
            x_saved = saved[0]
            flat = saved[1:]
            n = ctx.n_parts
            grad_x = None
            # Per flat tensor: accumulate if floating; None for integer indices
            grad_flat: list = [None] * len(flat)

            for i in range(n):
                base = 4 * i
                x_edge_s = flat[base]
                ei_s = flat[base + 1]
                wig_s = flat[base + 2]
                winv_s = flat[base + 3]

                x_leaf = x_saved.detach().requires_grad_(True)
                # Integer edge index: no grad. Float partitions: differentiate.
                x_edge_leaf = (
                    x_edge_s.detach().requires_grad_(True)
                    if x_edge_s.is_floating_point() and not _no_xedge_grad()
                    else x_edge_s.detach() if x_edge_s.is_floating_point() else x_edge_s
                )
                ei_leaf = ei_s
                wig_leaf = (
                    wig_s.detach().requires_grad_(True)
                    if wig_s.is_floating_point() and not _no_wig_grad()
                    else wig_s.detach() if wig_s.is_floating_point() else wig_s
                )
                winv_leaf = (
                    winv_s.detach().requires_grad_(True)
                    if winv_s.is_floating_point() and not _no_winv_grad()
                    else winv_s.detach() if winv_s.is_floating_point() else winv_s
                )

                grad_targets: list = []
                target_slots: list[tuple[str, int | None]] = []
                if not _no_x_grad():
                    grad_targets.append(x_leaf)
                    target_slots.append(("x", None))
                else:
                    x_leaf = x_saved.detach()
                if x_edge_leaf.requires_grad:
                    grad_targets.append(x_edge_leaf)
                    target_slots.append(("flat", base))
                if wig_leaf.requires_grad:
                    grad_targets.append(wig_leaf)
                    target_slots.append(("flat", base + 2))
                if winv_leaf.requires_grad:
                    grad_targets.append(winv_leaf)
                    target_slots.append(("flat", base + 3))
                if not grad_targets:
                    # No floating targets — skip autograd for this partition.
                    continue

                with torch.enable_grad():
                    x_use = x_leaf
                    if ctx.gather:
                        x_use = gp_utils.gather_from_model_parallel_region_sum_grad(
                            x_leaf, ctx.gather_natoms
                        )
                    # Optionally keep MoLE ac_start_idx alive through autograd.grad
                    # (upstream forward_chunk resets it to 0 before return).
                    if _hold_mole_idx():
                        from fairchem.core.models.uma.escn_md_block import (
                            set_mole_ac_start_index,
                        )

                        set_mole_ac_start_index(self, ctx.mole[i])
                    out = self.forward_chunk(
                        x_use,
                        ctx.x_n,
                        x_edge_leaf,
                        ei_leaf,
                        wig_leaf,
                        winv_leaf,
                        ctx.node_offset,
                        ctx.mole[i],
                    )
                    if _hold_mole_idx():
                        # Re-assert after forward_chunk's trailing reset.
                        set_mole_ac_start_index(self, ctx.mole[i])
                    grads = torch.autograd.grad(
                        out,
                        grad_targets,
                        grad_outputs=grad_out,
                        retain_graph=False,
                        allow_unused=True,
                    )
                    if _hold_mole_idx():
                        set_mole_ac_start_index(self, 0)

                for g, slot in zip(grads, target_slots):
                    if g is None:
                        continue
                    kind, idx = slot
                    if kind == "x":
                        grad_x = g if grad_x is None else grad_x + g
                    else:
                        assert idx is not None
                        if grad_flat[idx] is None:
                            grad_flat[idx] = g
                        else:
                            grad_flat[idx] = grad_flat[idx] + g

            if grad_x is None:
                grad_x = torch.zeros_like(x_saved)
            # Fill missing flat grads with zeros for floating tensors that
            # require grad (autograd.Function requires a grad or None per input).
            out_flat = []
            for t, g in zip(flat, grad_flat):
                if g is not None:
                    out_flat.append(g)
                elif t.requires_grad and t.is_floating_point():
                    out_flat.append(torch.zeros_like(t))
                else:
                    out_flat.append(None)
            return (grad_x, *out_flat)

    def _apply(x_in):  # type: ignore[no-untyped-def]
        flat: list = []
        for x_edge_p, ei_p, wig_p, winv_p, _mole in parts:
            flat.extend([x_edge_p, ei_p, wig_p, winv_p])
        return _SeqChunkAC.apply(x_in, *flat)

    return _apply


def _make_seq_ac_contig(
    self,
    x_n: int,
    node_offset: int,
    chunk_size: int,
    *,
    total_atoms: int | None = None,
    gather: bool = False,
):
    """SeqChunkAC that saves *full* edge/wigner tensors and slices inside.

    Avoids passing many ``tensor.split()`` views as distinct Function inputs,
    which can corrupt autograd into a shared base (Fz bias under GP).
    """
    import torch
    from fairchem.core.common import gp_utils

    gather_natoms = int(total_atoms) if gather else 0

    class _SeqChunkACContig(torch.autograd.Function):
        @staticmethod
        @torch.compiler.disable
        def forward(ctx, x_in, x_edge, edge_index, wigner, wigner_inv):  # type: ignore[no-untyped-def]
            ctx.x_n = x_n
            ctx.node_offset = node_offset
            ctx.chunk_size = int(chunk_size)
            ctx.gather = gather
            ctx.gather_natoms = gather_natoms
            ctx.save_for_backward(x_in, x_edge, edge_index, wigner, wigner_inv)
            n_edges = edge_index.shape[1]
            acc = None
            with torch.no_grad():
                start = 0
                mole = 0
                while start < n_edges:
                    end = min(start + ctx.chunk_size, n_edges)
                    x_use = x_in
                    if gather:
                        x_use = gp_utils.gather_from_model_parallel_region_sum_grad(
                            x_in, gather_natoms
                        )
                    out = self.forward_chunk(
                        x_use,
                        x_n,
                        x_edge[start:end],
                        edge_index[:, start:end],
                        wigner[start:end],
                        wigner_inv[start:end],
                        node_offset,
                        mole,
                    )
                    acc = out if acc is None else acc + out
                    mole += end - start
                    start = end
            return acc

        @staticmethod
        @torch.compiler.disable
        def backward(ctx, grad_out):  # type: ignore[no-untyped-def]
            x_saved, x_edge_s, ei_s, wig_s, winv_s = ctx.saved_tensors
            n_edges = ei_s.shape[1]
            grad_x = None
            grad_x_edge = (
                torch.zeros_like(x_edge_s) if x_edge_s.is_floating_point() else None
            )
            grad_wig = torch.zeros_like(wig_s) if wig_s.is_floating_point() else None
            grad_winv = (
                torch.zeros_like(winv_s) if winv_s.is_floating_point() else None
            )

            start = 0
            mole = 0
            while start < n_edges:
                end = min(start + ctx.chunk_size, n_edges)
                x_leaf = (
                    x_saved.detach().requires_grad_(True)
                    if not _no_x_grad()
                    else x_saved.detach()
                )
                x_edge_leaf = (
                    x_edge_s[start:end].detach().requires_grad_(True)
                    if x_edge_s.is_floating_point() and not _no_xedge_grad()
                    else (
                        x_edge_s[start:end].detach()
                        if x_edge_s.is_floating_point()
                        else x_edge_s[start:end]
                    )
                )
                ei_leaf = ei_s[:, start:end]
                wig_leaf = (
                    wig_s[start:end].detach().requires_grad_(True)
                    if wig_s.is_floating_point() and not _no_wig_grad()
                    else (
                        wig_s[start:end].detach()
                        if wig_s.is_floating_point()
                        else wig_s[start:end]
                    )
                )
                winv_leaf = (
                    winv_s[start:end].detach().requires_grad_(True)
                    if winv_s.is_floating_point() and not _no_winv_grad()
                    else (
                        winv_s[start:end].detach()
                        if winv_s.is_floating_point()
                        else winv_s[start:end]
                    )
                )

                grad_targets: list = []
                slots: list[str] = []
                if x_leaf.requires_grad:
                    grad_targets.append(x_leaf)
                    slots.append("x")
                if x_edge_leaf.requires_grad:
                    grad_targets.append(x_edge_leaf)
                    slots.append("x_edge")
                if wig_leaf.requires_grad:
                    grad_targets.append(wig_leaf)
                    slots.append("wig")
                if winv_leaf.requires_grad:
                    grad_targets.append(winv_leaf)
                    slots.append("winv")
                if not grad_targets:
                    mole += end - start
                    start = end
                    continue

                with torch.enable_grad():
                    x_use = x_leaf
                    if ctx.gather:
                        x_use = gp_utils.gather_from_model_parallel_region_sum_grad(
                            x_leaf, ctx.gather_natoms
                        )
                    out = self.forward_chunk(
                        x_use,
                        ctx.x_n,
                        x_edge_leaf,
                        ei_leaf,
                        wig_leaf,
                        winv_leaf,
                        ctx.node_offset,
                        mole,
                    )
                    grads = torch.autograd.grad(
                        out,
                        grad_targets,
                        grad_outputs=grad_out,
                        retain_graph=False,
                        allow_unused=True,
                    )

                for g, slot in zip(grads, slots):
                    if g is None:
                        continue
                    if slot == "x":
                        grad_x = g if grad_x is None else grad_x + g
                    elif slot == "x_edge":
                        assert grad_x_edge is not None
                        grad_x_edge[start:end] += g
                    elif slot == "wig":
                        assert grad_wig is not None
                        grad_wig[start:end] += g
                    elif slot == "winv":
                        assert grad_winv is not None
                        grad_winv[start:end] += g

                mole += end - start
                start = end

            if grad_x is None:
                grad_x = torch.zeros_like(x_saved)
            return (grad_x, grad_x_edge, None, grad_wig, grad_winv)

    def _apply(x_in, x_edge, edge_index, wigner, wigner_inv):  # type: ignore[no-untyped-def]
        return _SeqChunkACContig.apply(
            x_in, x_edge, edge_index, wigner, wigner_inv
        )

    return _apply


def _make_edeg_seq_ac(self, node_offset: int, parts: list):
    """Sequential chain AC for EdgeDegreeEmbedding fine chunks.

    Each chunk updates ``x`` via ``index_add`` (chained). Forward runs the
    chain under ``no_grad``. Backward walks reverse: recompute prefix to
    obtain ``x_i``, then one chunk with grad.
    """
    import torch

    n_parts = len(parts)

    class _EdegSeqChainAC(torch.autograd.Function):
        @staticmethod
        @torch.compiler.disable
        def forward(ctx, x_in, *flat):  # type: ignore[no-untyped-def]
            # flat: n_parts × (x_edge, ei, winv)
            ctx.node_offset = node_offset
            ctx.n_parts = n_parts
            ctx.save_for_backward(x_in, *flat)
            x = x_in
            with torch.no_grad():
                for i in range(n_parts):
                    base = 3 * i
                    x = self.forward_chunk(
                        x,
                        flat[base],
                        flat[base + 1],
                        flat[base + 2],
                        node_offset,
                    )
            return x

        @staticmethod
        @torch.compiler.disable
        def backward(ctx, grad_out):  # type: ignore[no-untyped-def]
            saved = ctx.saved_tensors
            x0 = saved[0]
            flat = saved[1:]
            n = ctx.n_parts
            grad = grad_out
            grad_flat: list = [None] * len(flat)

            for i in reversed(range(n)):
                # Recompute x_i = f_{i-1}(...f_0(x0)...) under no_grad.
                with torch.no_grad():
                    x_i = x0
                    for j in range(i):
                        b = 3 * j
                        x_i = self.forward_chunk(
                            x_i,
                            flat[b],
                            flat[b + 1],
                            flat[b + 2],
                            ctx.node_offset,
                        )

                base = 3 * i
                x_edge_s = flat[base]
                ei_s = flat[base + 1]
                winv_s = flat[base + 2]

                x_leaf = x_i.detach().requires_grad_(True)
                x_edge_leaf = (
                    x_edge_s.detach().requires_grad_(True)
                    if x_edge_s.is_floating_point()
                    else x_edge_s
                )
                ei_leaf = ei_s
                winv_leaf = (
                    winv_s.detach().requires_grad_(True)
                    if winv_s.is_floating_point()
                    else winv_s
                )

                grad_targets = [x_leaf]
                target_slots: list[tuple[str, int | None]] = [("x", None)]
                if x_edge_leaf.requires_grad:
                    grad_targets.append(x_edge_leaf)
                    target_slots.append(("flat", base))
                if winv_leaf.requires_grad:
                    grad_targets.append(winv_leaf)
                    target_slots.append(("flat", base + 2))

                with torch.enable_grad():
                    out = self.forward_chunk(
                        x_leaf,
                        x_edge_leaf,
                        ei_leaf,
                        winv_leaf,
                        ctx.node_offset,
                    )
                    grads = torch.autograd.grad(
                        out,
                        grad_targets,
                        grad_outputs=grad,
                        retain_graph=False,
                        allow_unused=True,
                    )

                grad = None
                for g, slot in zip(grads, target_slots):
                    if g is None:
                        continue
                    kind, idx = slot
                    if kind == "x":
                        grad = g
                    else:
                        assert idx is not None
                        if grad_flat[idx] is None:
                            grad_flat[idx] = g
                        else:
                            grad_flat[idx] = grad_flat[idx] + g

                if grad is None:
                    grad = torch.zeros_like(x0)

            out_flat = []
            for t, g in zip(flat, grad_flat):
                if g is not None:
                    out_flat.append(g)
                elif t.requires_grad and t.is_floating_point():
                    out_flat.append(torch.zeros_like(t))
                else:
                    out_flat.append(None)
            return (grad, *out_flat)

    def _apply(x_in):  # type: ignore[no-untyped-def]
        flat: list = []
        for x_edge_p, ei_p, winv_p in parts:
            flat.extend([x_edge_p, ei_p, winv_p])
        return _EdegSeqChainAC.apply(x_in, *flat)

    return _apply


def apply_edgewise_ckpt_clone() -> str:
    """Replace Edgewise.forward with sequential unique-x AC (v6).

    Returns the active combined marker string (Edgewise + edeg if applied).
    """
    global _APPLIED_EDGEWISE, MARKER
    import torch
    from fairchem.core.common import gp_utils
    from fairchem.core.models.uma import escn_md_block as block_mod

    marker = effective_marker()
    MARKER = marker

    if getattr(block_mod, "_fxpu_edgewise_ckpt_seq_v6_patched", False):
        _APPLIED_EDGEWISE = True
        return marker

    Edgewise = block_mod.Edgewise
    n_bisect = _bisect_n()

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
        do_gather_in = _gather_in_ac() and gp_utils.initialized()
        if gp_utils.initialized() and not do_gather_in:
            x_full = gp_utils.gather_from_model_parallel_region_sum_grad(
                x, total_atoms_across_gp_ranks
            )
        else:
            x_full = x

        chunk_sz = _effective_chunk_size(self.activation_checkpoint_chunk_size)
        if chunk_sz is None:
            if do_gather_in:
                # Single gather+chunk via SeqChunkAC for collective-aligned backward.
                part = [(x_edge, edge_index, wigner, wigner_inv_envelope, 0)]
                apply_seq = _make_seq_ac(
                    self,
                    x.shape[0],
                    node_offset,
                    part,
                    total_atoms=int(total_atoms_across_gp_ranks),
                    gather=True,
                )
                return apply_seq(x)
            return self.forward_chunk(
                x_full,
                x.shape[0],
                x_edge,
                edge_index,
                wigner,
                wigner_inv_envelope,
                node_offset,
            )

        edge_index_partitions = edge_index.split(chunk_sz, dim=1)
        wigner_partitions = wigner.split(chunk_sz, dim=0)
        wigner_inv_partitions = wigner_inv_envelope.split(chunk_sz, dim=0)
        x_edge_partitions = x_edge.split(chunk_sz, dim=0)
        x_n = x.shape[0]
        n_parts = len(edge_index_partitions)
        need_grad = bool(torch.is_grad_enabled() and x.requires_grad)

        parts: list[tuple] = []
        ac_mole_start_idx = 0
        for idx in range(n_parts):
            parts.append(
                (
                    x_edge_partitions[idx],
                    edge_index_partitions[idx],
                    wigner_partitions[idx],
                    wigner_inv_partitions[idx],
                    ac_mole_start_idx,
                )
            )
            ac_mole_start_idx += edge_index_partitions[idx].shape[1]

        if not need_grad:
            if do_gather_in:
                # Match forward path: gather per chunk under no_grad.
                acc = None
                for x_edge_p, ei_p, wig_p, winv_p, mole_idx in parts:
                    x_use = gp_utils.gather_from_model_parallel_region_sum_grad(
                        x, total_atoms_across_gp_ranks
                    )
                    out = self.forward_chunk(
                        x_use, x_n, x_edge_p, ei_p, wig_p, winv_p, node_offset, mole_idx
                    )
                    acc = out if acc is None else acc + out
                return acc
            return _run_parts_no_grad(self, x_full, x_n, node_offset, parts)

        # Contig AC: one Function, full tensors, slice inside (fixes multi-view grads).
        # Supersedes bisect/cloneParts when enabled (default ON).
        if _ac_contig():
            apply_c = _make_seq_ac_contig(
                self,
                x_n,
                node_offset,
                chunk_sz,
                total_atoms=int(total_atoms_across_gp_ranks)
                if do_gather_in
                else None,
                gather=do_gather_in,
            )
            x_in = x if do_gather_in else x_full
            return apply_c(x_in, x_edge, edge_index, wigner, wigner_inv_envelope)

        def _unique_parts(group: list[tuple]) -> list[tuple]:
            """Clone floating partitions so AC regions do not share storage."""
            out: list[tuple] = []
            for x_edge_p, ei_p, wig_p, winv_p, mole_idx in group:
                out.append(
                    (
                        x_edge_p.clone()
                        if x_edge_p.is_floating_point()
                        else x_edge_p,
                        ei_p,
                        wig_p.clone() if wig_p.is_floating_point() else wig_p,
                        winv_p.clone()
                        if winv_p.is_floating_point()
                        else winv_p,
                        mole_idx,
                    )
                )
            return out

        def _apply_group(group: list[tuple], x_in):
            apply_seq = _make_seq_ac(
                self,
                x_n,
                node_offset,
                group,
                total_atoms=int(total_atoms_across_gp_ranks)
                if do_gather_in
                else None,
                gather=do_gather_in,
            )
            return apply_seq(x_in)

        # Optional N-way bisect: unique x + unique float partitions per group
        if n_bisect >= 2:
            groups: list[list[tuple]] = [[] for _ in range(n_bisect)]
            for i, p in enumerate(parts):
                groups[(i * n_bisect) // n_parts].append(p)
            new_embeddings = []
            for group in groups:
                if not group:
                    continue
                x_in = x if do_gather_in else x_full.clone()
                if do_gather_in:
                    # Local shard: clone so multiple AC Functions do not share.
                    x_in = x.clone()
                new_embeddings.append(_apply_group(_unique_parts(group), x_in))
        return torch.stack(new_embeddings).sum(axis=0)

        # Default: one sequential AC over all fine parts.
        x_in = x if do_gather_in else x_full
        parts_use = _unique_parts(parts) if _clone_parts() else parts
        return _apply_group(parts_use, x_in)

    Edgewise.forward = forward  # type: ignore[method-assign]
    block_mod._fxpu_edgewise_ckpt_seq_v6_patched = True
    block_mod._fxpu_edgewise_ckpt_bisect_v6_patched = False
    block_mod._fxpu_edgewise_ckpt_seq_v5_patched = False
    block_mod._fxpu_edgewise_ckpt_fork_v4_patched = False
    block_mod._fxpu_edgewise_ckpt_reentrant_v3_patched = False
    block_mod._fxpu_edgewise_ckpt_single_v2_patched = False
    block_mod._fxpu_edgewise_ckpt_clone_patched = False
    block_mod._fxpu_edgewise_gather_in_ckpt_patched = False
    _APPLIED_EDGEWISE = True
    log.info(
        "FXPU Edgewise patch: sequential unique-x AC v6 "
        "(marker=%s, bisect=%s, gather_in_ac=%s, one_chunk=%s)",
        _edgewise_marker(),
        n_bisect or "off",
        _gather_in_ac(),
        _one_chunk(),
    )
    return marker
def apply_edge_degree_embedding_ckpt_seq() -> str:
    """Replace EdgeDegreeEmbedding.forward with sequential chain AC (edeg_v1).

    Returns the combined marker string.
    """
    global _APPLIED_EDEG, MARKER
    import torch
    from fairchem.core.models.uma.nn import embedding as emb_mod

    marker = effective_marker()
    MARKER = marker

    if getattr(emb_mod, "_fxpu_edeg_seq_v1_patched", False):
        _APPLIED_EDEG = True
        return marker

    EdgeDegreeEmbedding = emb_mod.EdgeDegreeEmbedding

    def forward(  # type: ignore[no-untyped-def]
        self,
        x,
        x_edge,
        edge_index,
        wigner_inv_envelope,
        node_offset=0,
    ):
        chunk_sz = _effective_chunk_size(self.activation_checkpoint_chunk_size)
        if _edeg_one_chunk():
            chunk_sz = None
        if chunk_sz is None:
            return self.forward_chunk(
                x,
                x_edge,
                edge_index,
                wigner_inv_envelope,
                node_offset,
            )

        edge_index_partitions = edge_index.split(chunk_sz, dim=1)
        wigner_inv_partitions = wigner_inv_envelope.split(chunk_sz, dim=0)
        x_edge_partitions = x_edge.split(chunk_sz, dim=0)
        n_parts = len(edge_index_partitions)
        need_grad = bool(torch.is_grad_enabled() and x.requires_grad)

        parts: list[tuple] = []
        for idx in range(n_parts):
            parts.append(
                (
                    x_edge_partitions[idx],
                    edge_index_partitions[idx],
                    wigner_inv_partitions[idx],
                )
            )

        if not need_grad:
            x_out = x
            for x_edge_p, ei_p, winv_p in parts:
                x_out = self.forward_chunk(
                    x_out, x_edge_p, ei_p, winv_p, node_offset
                )
            return x_out

        apply_seq = _make_edeg_seq_ac(self, node_offset, parts)
        return apply_seq(x)

    EdgeDegreeEmbedding.forward = forward  # type: ignore[method-assign]
    emb_mod._fxpu_edeg_seq_v1_patched = True
    _APPLIED_EDEG = True
    log.info(
        "FXPU EdgeDegreeEmbedding patch: sequential chain AC "
        "(marker=%s, one_chunk=%s)",
        EDEG_MARKER,
        _one_chunk(),
    )
    return marker


def apply_detach_wigner() -> str | None:
    """Detach Wigner / Wigner_inv after construction (``FXPU_DETACH_WIGNER=1``).

    Keeps radial/envelope differentiable via ``edge_distance``. If Fz bias
    disappears (scales→~1, cos jumps), the bug is in the Wigner→pos Jacobian
    under GP×multi-chunk AC. If Fz stays ~0.61, look elsewhere.
    """
    if not _detach_wigner():
        return None
    from fairchem.core.models.uma import escn_md as escn_md_mod

    cls = escn_md_mod.eSCNMDBackbone
    if getattr(cls, "_fxpu_detach_wigner_patched", False):
        return "detachW"

    _orig = cls._get_rotmat_and_wigner

    def _get_rotmat_and_wigner(self, edge_distance_vec):  # type: ignore[no-untyped-def]
        wigner, wigner_inv = _orig(self, edge_distance_vec)
        return wigner.detach(), wigner_inv.detach()

    cls._get_rotmat_and_wigner = _get_rotmat_and_wigner  # type: ignore[method-assign]
    cls._fxpu_detach_wigner_patched = True
    log.info("FXPU Wigner patch: detach wigner/wigner_inv (FXPU_DETACH_WIGNER=1)")
    return "detachW"


def apply_detach_envelope() -> str | None:
    """Detach polynomial envelope output (``FXPU_DETACH_ENVELOPE=1``).

    Complements detach-Wigner: envelope is fused into ``wigner_inv_envelope``
    after Wigner construction. If Fz→1, envelope→pos under GP×AC is the bug.
    """
    if not _detach_envelope():
        return None
    from fairchem.core.models.uma.nn.radial import PolynomialEnvelope

    if getattr(PolynomialEnvelope, "_fxpu_detach_envelope_patched", False):
        return "detachEnv"

    _orig = PolynomialEnvelope.forward

    def forward(self, x):  # type: ignore[no-untyped-def]
        return _orig(self, x).detach()

    PolynomialEnvelope.forward = forward  # type: ignore[method-assign]
    PolynomialEnvelope._fxpu_detach_envelope_patched = True
    log.info("FXPU envelope patch: detach envelope (FXPU_DETACH_ENVELOPE=1)")
    return "detachEnv"




def apply_detach_edge_dist() -> str | None:
    """Detach GaussianSmearing output (``FXPU_DETACH_EDGE_DIST=1``).

    Cuts force grads through scalar ``edge_distance`` → radial embedding while
    leaving ``edge_distance_vec`` → Wigner attached. Complements N/P/Q.
    """
    if not _detach_edge_dist():
        return None
    from fairchem.core.models.uma.nn.radial import GaussianSmearing

    if getattr(GaussianSmearing, "_fxpu_detach_edist_patched", False):
        return "detachEdist"

    _orig = GaussianSmearing.forward

    def forward(self, dist):  # type: ignore[no-untyped-def]
        return _orig(self, dist).detach()

    GaussianSmearing.forward = forward  # type: ignore[method-assign]
    GaussianSmearing._fxpu_detach_edist_patched = True
    log.info("FXPU radial patch: detach GaussianSmearing (FXPU_DETACH_EDGE_DIST=1)")
    return "detachEdist"



def apply_so2_radial_outside_ac() -> str | None:
    """Precompute ``SO2_Convolution.rad_func`` outside Edgewise AC.

    ``FXPU_SO2_RADIAL_OUTSIDE_AC=1``: ``get_layer_radial_emb`` runs each layer's
    ``rad_func(x_edge)`` once (outside chunked AC); SO2 forward skips a second
    ``rad_func`` call. Isolates whether rad_func-inside-chunked-AC causes Fz bias.
    """
    if not _so2_radial_outside_ac():
        return None
    from fairchem.core.models.uma.nn import execution_backends as eb
    from fairchem.core.models.uma.nn import so2_layers as so2_mod

    if getattr(eb.ExecutionBackend, "_fxpu_so2_rad_out_patched", False):
        return "so2RadOut"

    def get_layer_radial_emb(x_edge, model):  # type: ignore[no-untyped-def]
        import torch
        out = []
        for block in model.blocks:
            ew = block.edge_wise
            rf = getattr(ew.so2_conv_1, "rad_func", None)
            if rf is None:
                out.append(x_edge)
            else:
                # Checkpoint radial MLP so we do not keep all-layer activations
                out.append(
                    torch.utils.checkpoint.checkpoint(
                        rf, x_edge, use_reentrant=False
                    )
                )
        return out

    eb.ExecutionBackend.get_layer_radial_emb = staticmethod(get_layer_radial_emb)  # type: ignore[method-assign]
    eb.ExecutionBackend._fxpu_so2_rad_out_patched = True

    _orig_so2 = so2_mod.SO2_Convolution.forward

    def so2_forward(self, x, x_edge=None):  # type: ignore[no-untyped-def]
        # x_edge is already radial when FXPU_SO2_RADIAL_OUTSIDE_AC is on
        saved = self.rad_func
        try:
            self.rad_func = None
            return _orig_so2(self, x, x_edge)
        finally:
            self.rad_func = saved

    so2_mod.SO2_Convolution.forward = so2_forward  # type: ignore[method-assign]
    # SO2_Convolution_Module / other variants if present
    if hasattr(so2_mod, "SO2_Convolution_Module"):
        _orig2 = so2_mod.SO2_Convolution_Module.forward
        def so2_forward2(self, x, x_edge=None):  # type: ignore[no-untyped-def]
            saved = self.rad_func
            try:
                self.rad_func = None
                return _orig2(self, x, x_edge)
            finally:
                self.rad_func = saved
        so2_mod.SO2_Convolution_Module.forward = so2_forward2  # type: ignore[method-assign]

    log.info(
        "FXPU SO2 patch: rad_func outside Edgewise AC (FXPU_SO2_RADIAL_OUTSIDE_AC=1)"
    )
    return "so2RadOut"





def apply_edgewise_max_layers() -> str | None:
    """Limit how many blocks run Edgewise (``FXPU_EDGEWISE_MAX_LAYERS=N``).

    Later blocks keep atomwise but Edgewise returns zeros. Used with
    ``FXPU_EDGEWISE_ONE_CHUNK=1`` to try a memory-feasible one-chunk test on 18³.
    """
    n = _edgewise_max_layers()
    if n is None:
        return None
    import torch
    from fairchem.core.models.uma import escn_md_block as block_mod

    if getattr(block_mod, "_fxpu_max_layers_patched", False):
        return f"maxL{n}"

    Edgewise = block_mod.Edgewise
    _orig = Edgewise.forward
    # Stable per-module layer index. A call-counter resets on backbone forward but
    # NOT on activation-checkpoint recompute of Edgewise.forward — so recompute
    # would see idx>=n and return zeros (wrong AG). Module identity is stable.
    layer_index: dict[int, int] = {}

    def forward(self, x, x_edge, edge_index, wigner, wigner_inv_envelope,
                total_atoms_across_gp_ranks, node_offset: int = 0):  # type: ignore[no-untyped-def]
        key = id(self)
        if key not in layer_index:
            layer_index[key] = len(layer_index)
        idx = layer_index[key]
        if idx >= n:
            return torch.zeros_like(x)
        return _orig(
            self, x, x_edge, edge_index, wigner, wigner_inv_envelope,
            total_atoms_across_gp_ranks, node_offset,
        )

    Edgewise.forward = forward  # type: ignore[method-assign]
    block_mod._fxpu_max_layers_patched = True
    log.info(
        "FXPU Edgewise patch: max layers=%s (FXPU_EDGEWISE_MAX_LAYERS, per-module index)",
        n,
    )
    return f"maxL{n}"


def apply_detach_after_mp() -> str | None:
    """Skip Edgewise message path (``FXPU_DETACH_AFTER_MP=1``).

    Replaces ``Edgewise.forward`` with zeros so each block keeps only the
    residual (+ atomwise). Isolates whether Fz bias is carried by Edgewise
    under GP×multi-chunk AC vs edeg/atomwise/embed.
    """
    if not _detach_after_mp():
        return None
    import torch
    from fairchem.core.models.uma import escn_md_block as block_mod

    if getattr(block_mod, "_fxpu_skip_edgewise_msg_patched", False):
        return "detachMP"

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
        return torch.zeros_like(x)

    Edgewise.forward = forward  # type: ignore[method-assign]
    block_mod._fxpu_skip_edgewise_msg_patched = True
    log.info(
        "FXPU Edgewise patch: skip message path / zeros (FXPU_DETACH_AFTER_MP=1)"
    )
    return "detachMP"


def apply_n2e_clone() -> str | None:
    """Clone edge messages after node→edge Wigner permute (``FXPU_N2E_CLONE=1``)."""
    if not _n2e_clone():
        return None
    from fairchem.core.models.uma.nn import execution_backends as eb

    if getattr(eb.ExecutionBackend, "_fxpu_n2e_clone_patched", False):
        return "n2eClone"

    _orig = eb.ExecutionBackend.node_to_edge_wigner_permute

    @staticmethod
    def node_to_edge_wigner_permute(x_full, edge_index, wigner):  # type: ignore[no-untyped-def]
        out = _orig(x_full, edge_index, wigner)
        return out.clone() if hasattr(out, "clone") else out

    # bind correctly whether _orig is staticmethod or function
    def _wrapped(x_full, edge_index, wigner):  # type: ignore[no-untyped-def]
        if isinstance(_orig, staticmethod):
            out = _orig.__func__(x_full, edge_index, wigner)
        else:
            out = _orig(x_full, edge_index, wigner)
        return out.clone()

    eb.ExecutionBackend.node_to_edge_wigner_permute = staticmethod(_wrapped)  # type: ignore[method-assign]
    if hasattr(eb, "UMASFastPytorchBackend"):
        eb.UMASFastPytorchBackend.node_to_edge_wigner_permute = staticmethod(_wrapped)  # type: ignore[method-assign]
    eb.ExecutionBackend._fxpu_n2e_clone_patched = True
    log.info("FXPU n2e patch: clone after node_to_edge_wigner_permute (FXPU_N2E_CLONE=1)")
    return "n2eClone"


def apply_scatter_clone() -> str | None:
    """Clone before index_add in ExecutionBackend scatter paths (``FXPU_SCATTER_CLONE=1``)."""
    if not _scatter_clone():
        return None
    from fairchem.core.models.uma.nn import execution_backends as eb

    if getattr(eb.ExecutionBackend, "_fxpu_scatter_clone_patched", False):
        return "scatterClone"

    _perm = eb.ExecutionBackend.permute_wigner_inv_edge_to_node
    _edeg = eb.ExecutionBackend.edge_degree_scatter

    @staticmethod
    def permute_wigner_inv_edge_to_node(  # type: ignore[no-untyped-def]
        x_message, wigner_inv, edge_index, num_nodes, node_offset: int = 0
    ):
        import torch
        x_rotated = torch.bmm(wigner_inv, x_message).clone()
        new_embedding = torch.zeros(
            (num_nodes,) + x_rotated.shape[1:],
            dtype=x_rotated.dtype,
            device=x_rotated.device,
        )
        new_embedding = new_embedding.index_add(0, edge_index[1] - node_offset, x_rotated)
        return new_embedding

    @staticmethod
    def edge_degree_scatter(  # type: ignore[no-untyped-def]
        x,
        radial_output,
        wigner_inv,
        edge_index,
        m_0_num_coefficients,
        sphere_channels,
        rescale_factor,
        node_offset: int = 0,
    ):
        import torch
        radial = radial_output.reshape(-1, m_0_num_coefficients, sphere_channels)
        wigner_inv_m0 = wigner_inv[:, :, :m_0_num_coefficients]
        x_edge_embedding = torch.bmm(wigner_inv_m0, radial).to(x.dtype).clone()
        return x.index_add(
            0,
            edge_index[1] - node_offset,
            x_edge_embedding / rescale_factor,
        )

    eb.ExecutionBackend.permute_wigner_inv_edge_to_node = permute_wigner_inv_edge_to_node  # type: ignore[method-assign]
    eb.ExecutionBackend.edge_degree_scatter = edge_degree_scatter  # type: ignore[method-assign]
    eb.ExecutionBackend._fxpu_scatter_clone_patched = True
    # Also patch fast backend if it overrides
    if hasattr(eb, "UMASFastPytorchBackend"):
        eb.UMASFastPytorchBackend.permute_wigner_inv_edge_to_node = permute_wigner_inv_edge_to_node  # type: ignore[method-assign]
        eb.UMASFastPytorchBackend.edge_degree_scatter = edge_degree_scatter  # type: ignore[method-assign]
    log.info("FXPU scatter patch: clone before index_add (FXPU_SCATTER_CLONE=1)")
    return "scatterClone"


def apply_scatter_safe_backward() -> str | None:
    """``FXPU_SCATTER_SAFE_BACKWARD=1``: custom index_add with explicit gather backward.

    Bypasses XPU IndexAddBackward; forward still uses device index_add (chunked).
    Covers message + edeg scatter — the other collision-heavy reverse path besides edgevec.
    """
    if not _scatter_safe_backward():
        return None
    import torch
    from fairchem.core.models.uma.nn import execution_backends as eb

    if getattr(eb.ExecutionBackend, "_fxpu_scatter_safe_patched", False):
        return "scatterSafe"

    class _SafeIndexAdd(torch.autograd.Function):
        @staticmethod
        def forward(ctx, dest, index, source):  # type: ignore[no-untyped-def]
            ctx.save_for_backward(index)
            ctx.src_shape = tuple(source.shape)
            # Chunk large scatters to keep launch sizes in the W=12-safe regime.
            chunk = int(os.environ.get("FXPU_SCATTER_CHUNK_SIZE", "131072") or "131072")
            out = dest
            n = int(index.shape[0])
            if chunk <= 0 or n <= chunk:
                return out.index_add(0, index, source)
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                out = out.index_add(0, index[s:e], source[s:e])
            return out

        @staticmethod
        def backward(ctx, grad_out):  # type: ignore[no-untyped-def]
            (index,) = ctx.saved_tensors
            # dL/dsource[i] = dL/dout[index[i]]  (gather — avoid IndexAddBackward)
            grad_source = grad_out.index_select(0, index)
            return grad_out, None, grad_source

    def _idx(edge_index, node_offset):  # type: ignore[no-untyped-def]
        return edge_index[1] - node_offset

    @staticmethod
    def permute_wigner_inv_edge_to_node(  # type: ignore[no-untyped-def]
        x_message, wigner_inv, edge_index, num_nodes, node_offset: int = 0
    ):
        x_rotated = torch.bmm(wigner_inv, x_message)
        new_embedding = torch.zeros(
            (num_nodes,) + x_rotated.shape[1:],
            dtype=x_rotated.dtype,
            device=x_rotated.device,
        )
        return _SafeIndexAdd.apply(new_embedding, _idx(edge_index, node_offset), x_rotated)

    @staticmethod
    def edge_degree_scatter(  # type: ignore[no-untyped-def]
        x,
        radial_output,
        wigner_inv,
        edge_index,
        m_0_num_coefficients,
        sphere_channels,
        rescale_factor,
        node_offset: int = 0,
    ):
        radial = radial_output.reshape(-1, m_0_num_coefficients, sphere_channels)
        wigner_inv_m0 = wigner_inv[:, :, :m_0_num_coefficients]
        x_edge_embedding = torch.bmm(wigner_inv_m0, radial).to(x.dtype)
        src = x_edge_embedding / rescale_factor
        return _SafeIndexAdd.apply(x, _idx(edge_index, node_offset), src)

    eb.ExecutionBackend.permute_wigner_inv_edge_to_node = permute_wigner_inv_edge_to_node  # type: ignore[method-assign]
    eb.ExecutionBackend.edge_degree_scatter = edge_degree_scatter  # type: ignore[method-assign]
    if hasattr(eb, "UMASFastPytorchBackend"):
        eb.UMASFastPytorchBackend.permute_wigner_inv_edge_to_node = permute_wigner_inv_edge_to_node  # type: ignore[method-assign]
        eb.UMASFastPytorchBackend.edge_degree_scatter = edge_degree_scatter  # type: ignore[method-assign]
    eb.ExecutionBackend._fxpu_scatter_safe_patched = True
    log.info(
        "FXPU scatter patch: SAFE index_add Autograd (FXPU_SCATTER_SAFE_BACKWARD=1, chunk=%s)",
        os.environ.get("FXPU_SCATTER_CHUNK_SIZE", "131072"),
    )
    return "scatterSafe"


def apply_dump_prereduce_forces() -> str | None:
    """Dump per-rank −∂E_part/∂pos before GP force all_reduce.

    ``FXPU_DUMP_PREREDUCE_FORCES=1`` writes
    ``$FXPU_DUMP_PREREDUCE_DIR/prereduce_rank{R}.npy`` (and a summed check).
    Isolates whether Fz bias is already in local autograd or only after reduce.
    """
    if not _dump_prereduce_forces():
        return None
    import numpy as np
    import torch
    from fairchem.core.models.uma import outputs as uma_outputs
    from fairchem.core.models.uma import escn_md as escn_md_mod

    if getattr(uma_outputs.compute_forces, "_fxpu_is_dump_prereduce", False):
        return "dumpPreF"

    dump_dir = os.environ.get("FXPU_DUMP_PREREDUCE_DIR", "").strip()
    if not dump_dir:
        dump_dir = "pbs/out/parity_nacl_18/fix_probes/AI_prereduce"
    dump_dir = str(Path(dump_dir).resolve())
    os.makedirs(dump_dir, exist_ok=True)
    # Sentinel so we can see install even if force autograd never runs.
    Path(dump_dir, "dump_patch_installed.txt").write_text(
        f"pid={os.getpid()} dump_dir={dump_dir}\n", encoding="utf-8"
    )

    _inner = uma_outputs.compute_forces

    def compute_forces(energy_part, pos, training: bool = True):  # type: ignore[no-untyped-def]
        (grad,) = torch.autograd.grad(
            energy_part.sum(),
            pos,
            create_graph=training,
            retain_graph=True,
        )
        forces_local = torch.neg(grad)
        try:
            from fairchem.core.common import gp_utils

            rank = gp_utils.get_gp_rank() if gp_utils.initialized() else 0
        except Exception:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        path = os.path.join(dump_dir, f"prereduce_rank{rank:02d}.npy")
        arr = forces_local.detach().float().cpu().numpy()
        np.save(path, arr)
        log.info("FXPU dump pre-reduce forces rank=%s shape=%s -> %s", rank, arr.shape, path)
        if uma_outputs.gp_utils.initialized():
            forces = uma_outputs.gp_utils.reduce_from_model_parallel_region(forces_local)
        else:
            forces = forces_local
        return forces

    compute_forces._fxpu_is_dump_prereduce = True  # type: ignore[attr-defined]
    compute_forces._fxpu_wraps = _inner  # type: ignore[attr-defined]
    uma_outputs.compute_forces = compute_forces  # type: ignore[assignment]
    escn_md_mod.compute_forces = compute_forces  # type: ignore[assignment]
    # MLP heads may hold a module-level name binding via escn_md.
    try:
        from fairchem.core.models.uma import outputs as _o2

        assert _o2.compute_forces is compute_forces
    except Exception:
        pass
    uma_outputs._fxpu_dump_prereduce_patched = True
    log.info(
        "FXPU force patch: dump pre-reduce forces (FXPU_DUMP_PREREDUCE_FORCES=1) dir=%s",
        dump_dir,
    )
    return "dumpPreF"


def apply_force_no_stress() -> str | None:
    """Pos-only force autograd inside ``compute_forces_and_stress``.

    ``FXPU_FORCE_NO_STRESS=1``: keep stress *keys* in the predictor output (zeros)
    but take ``∂E/∂pos`` only — do not jointly differentiate ``cell``. Isolates
    whether joint pos+cell autograd under GP corrupts Fz (AI5 live path).
    """
    if not _force_no_stress():
        return None
    import torch
    from fairchem.core.models.uma import outputs as uma_outputs
    from fairchem.core.models.uma import escn_md as escn_md_mod

    if getattr(uma_outputs, "_fxpu_force_no_stress_patched", False) and getattr(
        uma_outputs.compute_forces_and_stress, "_fxpu_is_force_no_stress", False
    ):
        return "noStress"

    def compute_forces_and_stress(  # type: ignore[no-untyped-def]
        energy_part, pos, cell, batch, training: bool = True
    ):
        (grad_pos,) = torch.autograd.grad(
            energy_part.sum(),
            pos,
            create_graph=training,
            retain_graph=True,
        )
        forces = torch.neg(grad_pos)
        if uma_outputs.gp_utils.initialized():
            forces = uma_outputs.gp_utils.reduce_from_model_parallel_region(forces)
        # Dummy stress so task keys still exist; not used for parity.
        stress = torch.zeros(
            (cell.shape[0], 9), device=cell.device, dtype=cell.dtype
        )
        return forces, stress

    compute_forces_and_stress._fxpu_is_force_no_stress = True  # type: ignore[attr-defined]
    uma_outputs.compute_forces_and_stress = compute_forces_and_stress  # type: ignore[assignment]
    escn_md_mod.compute_forces_and_stress = compute_forces_and_stress  # type: ignore[assignment]
    for _mod in list(sys.modules.values()):
        if _mod is None:
            continue
        try:
            if getattr(_mod, "compute_forces_and_stress", None) is not None and "fairchem" in getattr(
                _mod, "__name__", ""
            ):
                _mod.compute_forces_and_stress = compute_forces_and_stress  # type: ignore[attr-defined]
        except Exception:
            pass
    uma_outputs._fxpu_force_no_stress_patched = True
    log.info(
        "FXPU force patch: compute_forces_and_stress pos-only (FXPU_FORCE_NO_STRESS=1)"
    )
    return "noStress"


def apply_wigner_gamma_zero() -> str | None:
    """Optionally force Wigner gamma=0 via ``FXPU_WIGNER_GAMMA_ZERO=1``."""
    if not _gamma_zero():
        return None
    import torch
    from fairchem.core.models.uma.common.quaternion import wigner_d_hybrid as wh

    if getattr(wh, "_fxpu_gamma_zero_patched", False):
        return "wigner_g0"

    _orig = wh.axis_angle_wigner_hybrid

    def axis_angle_wigner_hybrid(  # type: ignore[no-untyped-def]
        edge_distance_vec,
        lmax,
        gamma=None,
        coeffs=None,
        U_blocks=None,
        custom_kernels=None,
    ):
        if gamma is None:
            gamma = torch.zeros(
                edge_distance_vec.shape[0],
                dtype=edge_distance_vec.dtype,
                device=edge_distance_vec.device,
            )
        return _orig(
            edge_distance_vec,
            lmax,
            gamma=gamma,
            coeffs=coeffs,
            U_blocks=U_blocks,
            custom_kernels=custom_kernels,
        )

    wh.axis_angle_wigner_hybrid = axis_angle_wigner_hybrid  # type: ignore[assignment]
    # Also patch escn_md import binding if already imported.
    try:
        from fairchem.core.models.uma import escn_md as escn_md_mod

        escn_md_mod.axis_angle_wigner_hybrid = axis_angle_wigner_hybrid  # type: ignore[attr-defined]
    except Exception:
        pass
    wh._fxpu_gamma_zero_patched = True
    log.info("FXPU Wigner patch: gamma forced to 0 (FXPU_WIGNER_GAMMA_ZERO=1)")
    return "wigner_g0"


def apply_all_phase1_patches() -> str:
    """Apply Edgewise + EdgeDegreeEmbedding Phase-1 AC fixes."""
    marker = effective_marker()
    try:
        apply_wigner_gamma_zero()
    except ImportError:
        pass
    try:
        apply_detach_wigner()
    except ImportError:
        pass
    try:
        apply_detach_envelope()
    except ImportError:
        pass
    try:
        apply_detach_edge_dist()
    except ImportError:
        pass
    try:
        apply_so2_radial_outside_ac()
    except ImportError:
        pass
    try:
        apply_n2e_clone()
    except ImportError:
        pass
    try:
        apply_scatter_clone()
    except ImportError:
        pass
    try:
        apply_scatter_safe_backward()
    except ImportError:
        pass
    if _detach_after_mp():
        # Skip Edgewise entirely (zeros) — do not install SeqChunkAC on top.
        try:
            apply_detach_after_mp()
        except ImportError:
            pass
    elif _skip_edgewise():
        log.info(
            "FXPU Phase1: skipping Edgewise SeqChunkAC (FXPU_SKIP_EDGEWISE_PATCH=1); "
            "upstream torch.utils.checkpoint"
        )
        # Still honor maxL1 on upstream Edgewise (previously skipped → false "maxL1 OK").
        try:
            apply_edgewise_max_layers()
        except ImportError:
            pass
    else:
        try:
            apply_edgewise_ckpt_clone()
        except ImportError:
            log.info(
                "FXPU Phase1: FairChem not ready; Edgewise deferred (marker=%s)",
                marker,
            )
        try:
            apply_edgewise_max_layers()
        except ImportError:
            pass
    if _skip_edeg():
        log.info("FXPU Phase1: skipping EdgeDegreeEmbedding patch (FXPU_SKIP_EDEG_PATCH=1)")
        return effective_marker() + "+noEdeg"
    try:
        marker = apply_edge_degree_embedding_ckpt_seq()
    except ImportError:
        log.info(
            "FXPU Phase1: FairChem not ready; EdgeDegree deferred (marker=%s)",
            marker,
        )
    try:
        apply_dump_prereduce_forces()
    except ImportError:
        pass
    try:
        apply_force_no_stress()
    except ImportError:
        pass
    try:
        apply_edgevec_geometry_patches()
    except ImportError:
        pass
    return effective_marker()


def apply_edgevec_geometry_patches() -> str | None:
    """Opt-in safe / alt edge-vector construction for XPU AG diagnostics.

    ``FXPU_EDGEVEC_SAFE_BACKWARD=1``: custom autograd for ``pos[row]-pos[col]``
    with CPU ``numpy.add.at`` backward (diagnostic ground truth).
    ``FXPU_EDGEVEC_INDEX_SELECT=1``: use ``index_select`` instead of advanced idx.
    ``FXPU_PBC_OUT_OF_PLACE=1``: ``vec = vec + shifts`` instead of building with +.
    Patches ``eSCNMDBackbone._generate_graph`` edge-distance branch only.
    """
    if not (
        _edgevec_safe_backward() or _pbc_out_of_place() or _edgevec_index_select() or _edgevec_chunk_size()
    ):
        return None
    import torch
    from fairchem.core.models.uma import escn_md as escn_md_mod

    if getattr(escn_md_mod, "_fxpu_edgevec_geom_patched", False):
        tags = []
        if _edgevec_safe_backward():
            tags.append("edgeSafe")
        if _pbc_out_of_place():
            tags.append("pbcOop")
        if _edgevec_index_select():
            tags.append("idxSel")
        if _edgevec_chunk_size():
            tags.append(f"evChunk{_edgevec_chunk_size()}")
        return "+".join(tags) or "edgeGeom"

    class _SafePosGatherDiff(torch.autograd.Function):
        @staticmethod
        def forward(ctx, pos, row, col):  # type: ignore[no-untyped-def]
            ctx.save_for_backward(row, col)
            ctx.n_atoms = int(pos.shape[0])
            if _edgevec_index_select():
                return torch.index_select(pos, 0, row) - torch.index_select(pos, 0, col)
            return pos[row] - pos[col]

        @staticmethod
        def backward(ctx, grad_diff):  # type: ignore[no-untyped-def]
            import numpy as np

            row, col = ctx.saved_tensors
            # Diagnostic-safe: accumulate on CPU with numpy.add.at, copy back.
            g = grad_diff.detach().float().cpu().numpy()
            row_np = row.detach().cpu().numpy()
            col_np = col.detach().cpu().numpy()
            acc = np.zeros((ctx.n_atoms, 3), dtype=np.float64)
            np.add.at(acc, row_np, g.astype(np.float64, copy=False))
            np.add.at(acc, col_np, -g.astype(np.float64, copy=False))
            grad_pos = torch.tensor(acc, device=grad_diff.device, dtype=grad_diff.dtype)
            return grad_pos, None, None

    _orig = escn_md_mod.eSCNMDBackbone._generate_graph

    def _generate_graph(self, data_dict):  # type: ignore[no-untyped-def]
        # Call original (handles OTF + GP partition). Rebuild ONLY from
        # graph_dict["edge_index"] — data_dict["edge_index"] is often absent/empty
        # under otf_graph and caused E=0 distance vs E>0 embeddings (AX/AY crash).
        graph_dict = _orig(self, data_dict)
        if not (
            _edgevec_safe_backward()
            or _edgevec_index_select()
            or _pbc_out_of_place()
            or _edgevec_chunk_size()
        ):
            return graph_dict
        if "pos" not in data_dict or "edge_index" not in graph_dict:
            return graph_dict
        edge_index = graph_dict["edge_index"]
        if edge_index.numel() == 0 or edge_index.shape[1] == 0:
            return graph_dict
        pos = data_dict["pos"]
        row, col = edge_index[0], edge_index[1]
        n_e = int(row.shape[0])

        # Prefer shifts recovered from the already-correct (PBC) vectors in graph_dict.
        if "edge_distance_vec" in graph_dict:
            with torch.no_grad():
                if _edgevec_index_select():
                    base = torch.index_select(pos, 0, row) - torch.index_select(pos, 0, col)
                else:
                    base = pos[row] - pos[col]
                shifts = (graph_dict["edge_distance_vec"] - base).detach()
        elif "cell_offsets" in data_dict and "cell" in data_dict:
            if len(data_dict["natoms"]) == 1:
                shifts = data_dict["cell_offsets"].to(data_dict["cell"].dtype) @ data_dict[
                    "cell"
                ].squeeze(0)
            else:
                cell_per_edge = data_dict["cell"].repeat_interleave(
                    data_dict["nedges"], dim=0
                )
                shifts = torch.einsum(
                    "ij,ijk->ik",
                    data_dict["cell_offsets"].to(cell_per_edge.dtype),
                    cell_per_edge,
                )
        else:
            shifts = torch.zeros((n_e, 3), device=pos.device, dtype=pos.dtype)

        if _edgevec_safe_backward():
            diff = _SafePosGatherDiff.apply(pos, row, col)
        elif _edgevec_chunk_size():
            chunk = int(_edgevec_chunk_size() or 131072)
            pieces = []
            for s in range(0, n_e, chunk):
                e = min(s + chunk, n_e)
                r, c = row[s:e], col[s:e]
                if _edgevec_index_select():
                    pieces.append(
                        torch.index_select(pos, 0, r) - torch.index_select(pos, 0, c)
                    )
                else:
                    pieces.append(pos[r] - pos[c])
            diff = torch.cat(pieces, dim=0) if pieces else pos.new_zeros((0, 3))
        elif _edgevec_index_select():
            diff = torch.index_select(pos, 0, row) - torch.index_select(pos, 0, col)
        else:
            diff = pos[row] - pos[col]

        edge_distance_vec = diff + shifts
        edge_distance = torch.linalg.norm(edge_distance_vec, dim=-1, keepdim=False)
        graph_dict = dict(graph_dict)
        graph_dict["edge_distance_vec"] = edge_distance_vec
        graph_dict["edge_distance"] = edge_distance
        return graph_dict

    escn_md_mod.eSCNMDBackbone._generate_graph = _generate_graph  # type: ignore[method-assign]
    escn_md_mod._fxpu_edgevec_geom_patched = True
    tags = []
    if _edgevec_safe_backward():
        tags.append("edgeSafe")
    if _pbc_out_of_place():
        tags.append("pbcOop")
    if _edgevec_index_select():
        tags.append("idxSel")
    if _edgevec_chunk_size():
        tags.append(f"evChunk{_edgevec_chunk_size()}")
    marker = "+".join(tags) or "edgeGeom"
    log.info("FXPU edgevec geometry patch: %s", marker)
    return marker


def apply() -> str:
    ensure_project_pythonpath()
    import fairchem_xpu_parallel as fxp  # noqa: F401

    marker = apply_all_phase1_patches()
    global MARKER
    MARKER = marker
    log.info("FXPU Phase1 apply() done (marker=%s)", marker)
    return marker
