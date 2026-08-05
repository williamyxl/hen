"""Phase 4: reduce collective cost under XCCL (odd world-size all_reduce).

Profile (Phase 2/3c, NaCl 20³ FP64 E+F):
  - W≥4: coll_frac already ~0.1–0.6%; all_gather moves ~GB but is fast.
  - W=3: coll_frac ~40%; **all_reduce** dominates (~13.5 s / 36 calls) despite
    ~128 KiB/call. Same call counts/bytes as W=4 (where AR is ~0.01 s total).
  → Odd world-size XCCL/oneCCL all_reduce latency, not FairChem traffic volume.

FairChem InferenceSettings levers (merge_mole, AC off, turbo/compile/tf32) do
not safely cut GP all_reduce traffic under STRICT POLICY (FP64 E+F, no turbo).

This patch: for odd world_size and SUM/AVG, replace native ``dist.all_reduce``
with ``all_gather`` + local sum (in-place). Mathematically equivalent for SUM;
often much faster for small tensors on odd W. Gated by ``FXPU_PHASE4_ODD_AR``
(default **off** — 20³ gate showed W=3 collectives **worse** with this path).
Also times ``reduce_scatter`` (xccl GP backward).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

log = logging.getLogger(__name__)

MARKER = "fxpu_phase4_odd_ar_v1"


def odd_ar_enabled() -> bool:
    """Default off; set FXPU_PHASE4_ODD_AR=1 to enable (not recommended after Phase4 gate)."""
    return os.environ.get("FXPU_PHASE4_ODD_AR", "0").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


def apply_phase4_collective_reductions(*, acc_fn: Callable[..., None]) -> str:
    """Wrap dist collectives after Phase 2 XCCL timers are installed."""
    import torch
    import torch.distributed as dist
    from fairchem.core.common import gp_utils

    if getattr(gp_utils, "_fxpu_phase4_patched", False):
        return MARKER

    # Prefer originals saved by Phase 2; else current (pre-phase4) bindings.
    _orig_ag = getattr(dist, "_fxpu_orig_all_gather", None) or dist.all_gather
    _prev_ar = dist.all_reduce  # may already be Phase2 timed wrapper

    def _nbytes(t: torch.Tensor) -> int:
        try:
            return int(t.numel() * t.element_size())
        except Exception:
            return 0

    def _safe_acc(name: str, dt: float, nbytes: int = 0) -> None:
        """acc_fn assumes fixed keys in sqs_sampling; seed reduce_scatter keys."""
        try:
            acc_fn(name, dt, nbytes)
            return
        except KeyError:
            pass
        # Extend process-local timer dict without editing sqs_sampling source.
        try:
            import fairchem_xpu_parallel as fxp

            with fxp._stats_lock:
                fxp._collective_stats.setdefault(f"{name}_calls", 0)
                fxp._collective_stats.setdefault(f"{name}_s", 0.0)
                fxp._collective_stats.setdefault(f"{name}_bytes", 0)
            acc_fn(name, dt, nbytes)
        except Exception:
            log.debug("Phase4 timer skip for %s", name, exc_info=True)

    def _world(group: Any) -> int:
        if group is None:
            return int(dist.get_world_size())
        return int(dist.get_world_size(group=group))

    def _is_sum_or_avg(op: Any) -> tuple[bool, bool]:
        """Return (ok, is_avg)."""
        if op is None:
            return True, False
        try:
            if op == dist.ReduceOp.SUM:
                return True, False
            if op == dist.ReduceOp.AVG:
                return True, True
        except Exception:
            pass
        return False, False

    def _all_reduce_odd_via_gather(tensor, op=None, group=None, async_op=False):  # type: ignore[no-untyped-def]
        """In-place SUM/AVG via all_gather + local reduce (odd world sizes)."""
        ok, is_avg = _is_sum_or_avg(op)
        world = _world(group)
        use_gather = (
            odd_ar_enabled()
            and not async_op
            and ok
            and world >= 3
            and (world % 2 == 1)
        )
        if not use_gather:
            # Fall through to previous wrapper (Phase2 timed → native).
            return _prev_ar(tensor, op=op, group=group, async_op=async_op)

        t0 = time.perf_counter()
        bufs = [torch.empty_like(tensor) for _ in range(world)]
        # Untimed original gather so bytes/time land under all_reduce only.
        work = _orig_ag(bufs, tensor, group=group, async_op=False)
        if work is not None:
            work.wait()
        tensor.copy_(bufs[0])
        for i in range(1, world):
            tensor.add_(bufs[i])
        if is_avg:
            tensor.div_(world)
        _safe_acc("all_reduce", time.perf_counter() - t0, _nbytes(tensor))
        return None

    dist.all_reduce = _all_reduce_odd_via_gather  # type: ignore[assignment]

    # Time reduce_scatter (FairChem GP backward under xccl); do not alter alg.
    _orig_rs = getattr(dist, "reduce_scatter", None)
    if _orig_rs is not None and not getattr(dist, "_fxpu_phase4_rs_timed", False):

        def _timed_reduce_scatter(output, input_list, op=None, group=None, async_op=False):  # type: ignore[no-untyped-def]
            t0 = time.perf_counter()
            if op is None:
                work = _orig_rs(output, input_list, group=group, async_op=async_op)
            else:
                work = _orig_rs(
                    output, input_list, op=op, group=group, async_op=async_op
                )
            if async_op and work is not None:
                work.wait()
            n = _nbytes(output)
            try:
                n = sum(_nbytes(t) for t in input_list)
            except Exception:
                pass
            _safe_acc("reduce_scatter", time.perf_counter() - t0, n)
            return None if async_op else work

        dist.reduce_scatter = _timed_reduce_scatter  # type: ignore[assignment]
        dist._fxpu_phase4_rs_timed = True  # type: ignore[attr-defined]

    gp_utils._fxpu_phase4_patched = True
    log.info(
        "FXPU Phase4: odd-W all_reduce via all_gather+sum enabled=%s marker=%s",
        odd_ar_enabled(),
        MARKER,
    )
    return MARKER
