"""Phase 2: native XCCL (oneCCL) collectives — skip CPU staging under Ray.

When ``FXPU_DIST_BACKEND=xccl``, keep FairChem's native GP Gather/Reduce
(XPU tensors + reduce_scatter backward) and only wrap ``dist.all_*`` for
stage timers. Do **not** apply the gloo CPU-staging Autograd Functions.

Ray workers are not MPI ranks: use ATL/OFI + TCP (or shm) and the default
internal KVS — never ``CCL_KVS_MODE=mpi``.
"""

from __future__ import annotations


import sys
from pathlib import Path as _Path
_scripts = _Path(__file__).resolve().parents[1] / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

try:
    from fxpu_env_compat import apply_hen_to_fxpu_env_compat
    apply_hen_to_fxpu_env_compat()
except ImportError:
    pass

import logging
import os
import subprocess
import time
from typing import Any

log = logging.getLogger(__name__)

MARKER = "fxpu_xccl_native_v13"
UNEVEN_GATHER_MARKER = "fxpu_xccl_gather_bcast_rs_auto_v16_stripe"

_XCCL_ALIASES = frozenset({"xccl", "ccl", "oneccl"})

# Phase6 starts remote raylets via ``mpiexec -n 1``; actors inherit PMI_* with
# SIZE=1. oneCCL then treats Ray workers as MPI ranks → silent collective
# corruption (W=24 garbage E). Scrub these before xccl init.
_PMI_POISON_PREFIXES = (
    "PMI_",
    "PMIX_",
    "PALS_",
    "I_MPI_",
    "MPIR_",
    "HYDRA_",
    "OMPI_",
    "MPI_",
)


def use_xccl() -> bool:
    return os.environ.get("FXPU_DIST_BACKEND", "gloo").strip().lower() in _XCCL_ALIASES


def maybe_demote_multinode_xccl_to_gloo() -> bool:
    """Opt-in demote of multi-node Ray+XCCL → gloo.

    Historical fails (garbage E / bond0 / cxi) were fixed in ``fxpu_xccl_native_v5``:
    ``CCL_PROCESS_LAUNCHER=none`` + PMI scrub + hsn*-only ``FI_TCP_IFACE``.
    Proven PASS: NaCl 18³ W=24 job ``8729300`` (cos=1 vs W=1).

    Rollback: ``FXPU_DEMOTE_MULTINODE_XCCL=1``. Legacy
    ``FXPU_FORCE_MULTINODE_XCCL=1`` is accepted but no longer required.
    """
    if not use_xccl():
        return False
    if not is_multinode_xccl():
        return False
    if _truthy(os.environ.get("FXPU_DEMOTE_MULTINODE_XCCL")):
        os.environ["FXPU_DIST_BACKEND"] = "gloo"
        log.warning(
            "FXPU Phase2: FXPU_DEMOTE_MULTINODE_XCCL=1 — multi-node xccl demoted to gloo. "
            "marker=%s",
            MARKER,
        )
        return True
    # Keep xccl; ensure Ray-safe launcher defaults are present early.
    os.environ.setdefault("CCL_PROCESS_LAUNCHER", "none")
    if _truthy(os.environ.get("FXPU_FORCE_MULTINODE_XCCL")):
        log.info(
            "FXPU Phase2: multi-node xccl enabled (FXPU_FORCE_MULTINODE_XCCL legacy ok). "
            "marker=%s",
            MARKER,
        )
    return False


def maybe_demote_uneven_xccl_to_gloo(natoms: int, world_size: int) -> bool:
    """Opt-in demote xccl→gloo when ``natoms % world_size != 0``.

    Default **OFF**: uneven GP gathers stay on XCCL via broadcast-based
    ``all_gather`` replacement (``UNEVEN_GATHER_MARKER``). Legacy escape hatch:
    ``FXPU_AUTO_GLOO_UNEVEN=1``.
    """
    if not use_xccl():
        return False
    try:
        w = int(world_size)
        n = int(natoms)
    except (TypeError, ValueError):
        return False
    if w <= 1 or n <= 0 or n % w == 0:
        return False
    if not _truthy(os.environ.get("FXPU_AUTO_GLOO_UNEVEN")):
        return False
    os.environ["FXPU_DIST_BACKEND"] = "gloo"
    log.warning(
        "FXPU Phase2: uneven GP split (natoms=%s %% world=%s = %s) — "
        "xccl demoted to gloo (FXPU_AUTO_GLOO_UNEVEN=1). marker=%s",
        n,
        w,
        n % w,
        MARKER,
    )
    return True


def _truthy(val: str | None) -> bool:
    return (val or "").strip().lower() in ("1", "true", "on", "yes")


def _falsy(val: str | None) -> bool:
    return (val or "").strip().lower() in ("0", "false", "off", "no")


def scrub_mpi_launcher_env() -> list[str]:
    """Remove PMI/MPI launcher vars that poison oneCCL under Ray.

    Returns the keys that were deleted.
    """
    removed: list[str] = []
    for key in list(os.environ):
        if any(key.startswith(p) for p in _PMI_POISON_PREFIXES):
            del os.environ[key]
            removed.append(key)
    # Explicit MPI KVS is fatal under Ray (no real MPI ranks for the PG).
    for key in ("CCL_KVS_MODE",):
        if key in os.environ:
            del os.environ[key]
            removed.append(key)
    return removed


def is_multinode_xccl(world_size: int | None = None) -> bool:
    """True when Phase6 / world > tiles-per-node (cross-host XCCL)."""
    if _truthy(os.environ.get("FXPU_PHASE6_MULTINODE")):
        return True
    try:
        per = max(1, int(os.environ.get("FXPU_TILES_PER_NODE", "12") or 12))
    except ValueError:
        per = 12
    if world_size is not None and int(world_size) > per:
        return True
    try:
        ws = int(os.environ.get("WORLD_SIZE", "0") or 0)
    except ValueError:
        ws = 0
    return ws > per


def default_fi_provider(*, world_size: int | None = None) -> str:
    """OFI provider for Ray+XCCL.

    Default ``tcp`` for both same-node and multi-node Ray: ``cxi`` yields
    ``fi_getinfo -61`` / ATL init failure under Ray (job 8728944). Aurora HSN
    ``cxi`` remains available via ``FXPU_FI_PROVIDER=cxi`` for non-Ray launches.
    Override anytime with ``FXPU_FI_PROVIDER``.
    """
    explicit = (os.environ.get("FXPU_FI_PROVIDER") or "").strip()
    if explicit:
        return explicit
    # Prefer explicit multi-node override knob, else tcp (Ray-safe).
    if is_multinode_xccl(world_size):
        return (os.environ.get("FXPU_FI_PROVIDER_MULTINODE") or "tcp").strip() or "tcp"
    return "tcp"


def _list_local_hsn_ifaces() -> list[tuple[str, str]]:
    """Return ``[(iface, ipv4), ...]`` for local ``hsn*`` devices (10.112 first)."""
    try:
        out = subprocess.check_output(
            ["ip", "-o", "-4", "addr", "show"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    pref: list[tuple[str, str]] = []
    other: list[tuple[str, str]] = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[2] != "inet":
            continue
        name = parts[1].rstrip(":")
        addr = parts[3].split("/", 1)[0]
        if not name.startswith("hsn"):
            continue
        item = (name, addr)
        if addr.startswith("10.112."):
            pref.append(item)
        else:
            other.append(item)
    # Stable order by iface name within each bucket.
    pref.sort(key=lambda x: x[0])
    other.sort(key=lambda x: x[0])
    return pref + other


def pin_ofi_tcp_iface(*, preferred_ip: str | None = None) -> str | None:
    """Pin ``FI_TCP_IFACE`` to a local **HSN** netdev (``hsn*`` only).

    Never accept ``bond*`` / mgmt NICs — job 8729110 pinned ``bond0`` and
    Ray head failed; wrong iface can also silently corrupt XCCL (8728986).
    Override only with ``FXPU_FI_TCP_IFACE``.

    ``FXPU_TCP_IFACE_STRIPE`` (A/B multi-rail, still pinned to hsn*):
      - ``node`` / ``host``: ``hsn[FXPU_NODE_ID % n_hsn]`` (leaders differ by node)
      - ``local`` / ``tile``: ``hsn[CCL_LOCAL_RANK % n_hsn]``
      - ``global``: ``hsn[FXPU_WORKER_ID % n_hsn]``
    Default (unset): first 10.112 hsn* (usually hsn0) — C6e locked path.
    """
    del preferred_ip  # kept for API compat; HSN scan is authoritative
    explicit = (os.environ.get("FXPU_FI_TCP_IFACE") or "").strip()
    if explicit:
        if not explicit.startswith("hsn") and not _truthy(
            os.environ.get("FXPU_ALLOW_NON_HSN_TCP_IFACE")
        ):
            raise RuntimeError(
                f"FXPU Phase2: FXPU_FI_TCP_IFACE={explicit!r} is not hsn* "
                "(set FXPU_ALLOW_NON_HSN_TCP_IFACE=1 to override)"
            )
        os.environ["FI_TCP_IFACE"] = explicit
        return explicit

    hsns = _list_local_hsn_ifaces()
    stripe = (os.environ.get("FXPU_TCP_IFACE_STRIPE") or "").strip().lower()
    if stripe and hsns:
        if stripe in ("node", "host"):
            try:
                idx = int(os.environ.get("FXPU_NODE_ID", "0") or 0)
            except ValueError:
                idx = 0
        elif stripe in ("local", "tile", "rank"):
            try:
                idx = int(
                    os.environ.get("CCL_LOCAL_RANK")
                    or os.environ.get("LOCAL_RANK")
                    or "0"
                )
            except ValueError:
                idx = 0
        elif stripe in ("global", "worker", "world"):
            try:
                idx = int(
                    os.environ.get("FXPU_WORKER_ID")
                    or os.environ.get("RANK")
                    or "0"
                )
            except ValueError:
                idx = 0
        else:
            log.warning(
                "FXPU Phase2: unknown FXPU_TCP_IFACE_STRIPE=%r — using hsn0 path",
                stripe,
            )
            idx = 0
        name, addr = hsns[idx % len(hsns)]
        os.environ["FI_TCP_IFACE"] = name
        log.info(
            "FXPU Phase2: striped FI_TCP_IFACE=%s (ip=%s stripe=%s idx=%s n_hsn=%s)",
            name,
            addr,
            stripe,
            idx,
            len(hsns),
        )
        return name

    if hsns:
        name, addr = hsns[0]
        os.environ["FI_TCP_IFACE"] = name
        log.info(
            "FXPU Phase2: pinned FI_TCP_IFACE=%s (HSN ip=%s) for OFI tcp",
            name,
            addr,
        )
        return name
    log.warning("FXPU Phase2: no local hsn* iface found for FI_TCP_IFACE pin")
    return None


def configure_xccl_env_for_ray(
    local_rank: int | None = None,
    local_size: int | None = None,
    *,
    world_size: int | None = None,
    pin_tcp: bool = True,
) -> dict[str, str]:
    """ATL/OFI env for Ray workers (no PMI / no MPI KVS).

    Returns the env keys we set (for logging / Ray runtime_env).
    """
    removed = scrub_mpi_launcher_env()
    if removed:
        log.info(
            "FXPU Phase2: scrubbed %d PMI/MPI env keys before XCCL (sample=%s)",
            len(removed),
            removed[:8],
        )

    # Prefer OFI; MPI transport needs a real launcher / PMI.
    os.environ.setdefault("CCL_ATL_TRANSPORT", "ofi")
    # Critical for Ray: do not use hydra/pmix defaults (accelerate#2340).
    os.environ["CCL_PROCESS_LAUNCHER"] = "none"
    fi = default_fi_provider(world_size=world_size)
    os.environ["FI_PROVIDER"] = fi
    # FXPU_SKIP_TCP_IFACE_PIN=1: do not pin a single hsn* (A/B multi-rail).
    # Default remains pin-to-hsn* to avoid bond0/mgmt corruption (8729110).
    skip_pin = _truthy(os.environ.get("FXPU_SKIP_TCP_IFACE_PIN"))
    if skip_pin:
        os.environ.pop("FI_TCP_IFACE", None)
        log.info("FXPU Phase2: skipping FI_TCP_IFACE pin (FXPU_SKIP_TCP_IFACE_PIN)")
    elif pin_tcp and (fi == "tcp" or fi.startswith("tcp")):
        pin_ofi_tcp_iface()
    os.environ.setdefault("CCL_ZE_IPC_EXCHANGE", "sockets")
    os.environ.setdefault("CCL_WORKER_COUNT", "1")
    os.environ.setdefault("CCL_LOG_LEVEL", os.environ.get("CCL_LOG_LEVEL", "warn"))
    # Hardened GP collectives (NaCl 21/22/24/31 W=12 NotPresent fixes).
    os.environ.setdefault("FXPU_XCCL_UNEVEN_GATHER", "broadcast")
    os.environ.setdefault("FXPU_XCCL_RS_MODE", "auto")
    os.environ.setdefault("FXPU_XCCL_RS_CHUNK", "0")
    os.environ.setdefault("FXPU_XCCL_PAD_ALIGN", "32")
    os.environ.setdefault("FXPU_AUTO_GLOO_UNEVEN", "0")

    if local_rank is not None:
        os.environ["CCL_LOCAL_RANK"] = str(int(local_rank))
        os.environ["LOCAL_RANK"] = str(int(local_rank))
    if local_size is not None:
        os.environ["CCL_LOCAL_SIZE"] = str(int(local_size))
        os.environ["LOCAL_WORLD_SIZE"] = str(int(local_size))

    keys = (
        "CCL_ATL_TRANSPORT",
        "CCL_PROCESS_LAUNCHER",
        "FI_PROVIDER",
        "FI_TCP_IFACE",
        "CCL_ZE_IPC_EXCHANGE",
        "CCL_WORKER_COUNT",
        "CCL_LOCAL_RANK",
        "CCL_LOCAL_SIZE",
        "CCL_ROOT",
    )
    snap = {k: os.environ[k] for k in keys if k in os.environ}
    log.info(
        "FXPU Phase2 XCCL env for Ray: %s (marker=%s multinode=%s)",
        snap,
        MARKER,
        is_multinode_xccl(world_size),
    )
    return snap


def ray_worker_xccl_env_vars() -> dict[str, str]:
    """Subset of env to inject into Ray actor ``runtime_env``."""
    # Do not pin driver's FI_TCP_IFACE into the returned dict / process for
    # remote actors — each actor pins locally in configure/init hook.
    configure_xccl_env_for_ray(pin_tcp=False)
    # Drop any driver-local iface so it cannot leak via os.environ copy loops.
    os.environ.pop("FI_TCP_IFACE", None)
    out: dict[str, str] = {
        "FXPU_DIST_BACKEND": "xccl",
        "CCL_ATL_TRANSPORT": os.environ.get("CCL_ATL_TRANSPORT", "ofi"),
        "CCL_PROCESS_LAUNCHER": "none",
        "FI_PROVIDER": os.environ.get("FI_PROVIDER", default_fi_provider()),
        "FXPU_FI_PROVIDER": os.environ.get("FXPU_FI_PROVIDER", default_fi_provider()),
        "CCL_ZE_IPC_EXCHANGE": os.environ.get("CCL_ZE_IPC_EXCHANGE", "sockets"),
        "CCL_WORKER_COUNT": os.environ.get("CCL_WORKER_COUNT", "1"),
        "FXPU_XCCL_UNEVEN_GATHER": os.environ.get(
            "FXPU_XCCL_UNEVEN_GATHER", "broadcast"
        ),
        "FXPU_XCCL_RS_MODE": os.environ.get("FXPU_XCCL_RS_MODE", "auto"),
        "FXPU_XCCL_RS_CHUNK": os.environ.get("FXPU_XCCL_RS_CHUNK", "0"),
        "FXPU_XCCL_PAD_ALIGN": os.environ.get("FXPU_XCCL_PAD_ALIGN", "32"),
        "FXPU_AUTO_GLOO_UNEVEN": os.environ.get("FXPU_AUTO_GLOO_UNEVEN", "0"),
    }
    if "CCL_ROOT" in os.environ:
        out["CCL_ROOT"] = os.environ["CCL_ROOT"]
    if "FI_PROVIDER_PATH" in os.environ:
        out["FI_PROVIDER_PATH"] = os.environ["FI_PROVIDER_PATH"]
    if "FXPU_FI_TCP_IFACE" in os.environ:
        out["FXPU_FI_TCP_IFACE"] = os.environ["FXPU_FI_TCP_IFACE"]
    if "LD_LIBRARY_PATH" in os.environ:
        out["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    return out


def _pad_input_contiguous(input: "Any", padded_size: int) -> "Any":
    """Like FairChem ``pad_input`` but always returns a contiguous buffer.

    ``torch.cat`` padding was implicated in XPU ``NotPresent`` crashes on the
    first native XCCL ``all_gather`` for uneven GP shards (NaCl 22³/23³ W=12).
    """
    import torch

    if int(input.shape[0]) == int(padded_size):
        return input if input.is_contiguous() else input.contiguous()
    out = input.new_zeros((int(padded_size),) + tuple(input.shape[1:]))
    n = int(input.shape[0])
    if n > 0:
        out[:n].copy_(input if input.is_contiguous() else input.contiguous())
    return out


def _apply_xccl_uneven_gather_safe(gp_utils: Any, *, acc_fn) -> None:
    """Keep XCCL: replace GP ``all_gather`` with per-rank ``broadcast``.

    Native XCCL ``all_gather`` (and sometimes ``reduce_scatter``) segfaults on
    Aurora for some NaCl sizes at W=12 (``NotPresent``), including equal-split
    N=21 flakes and uneven N=22. Broadcast + ``all_reduce`` backward stay on
    XCCL/XPU and avoid those collectives.

    ``FXPU_XCCL_UNEVEN_GATHER``:
      - ``broadcast`` / ``always`` (default): flat W-wide broadcast gather
      - ``hierarchical`` / ``hier``: intra-node then inter-node broadcast gather
        (needs ``world % FXPU_TILES_PER_NODE == 0``; else falls back to flat)
      - ``uneven``: broadcast only when ``natoms % W != 0``
      - ``native`` / ``off``: stock FairChem all_gather
    """
    import time as _time

    import torch
    import torch.distributed as dist

    mode = (os.environ.get("FXPU_XCCL_UNEVEN_GATHER", "broadcast") or "broadcast").strip().lower()
    if mode in ("0", "off", "false", "no", "native", "all_gather"):
        log.info(
            "FXPU Phase2: XCCL gather safety disabled (FXPU_XCCL_UNEVEN_GATHER=%s)",
            mode,
        )
        return
    if getattr(gp_utils, "_fxpu_xccl_uneven_gather", False):
        return

    always_bcast = mode in ("broadcast", "always", "all", "1", "true", "on", "yes")
    use_hier = mode in ("hierarchical", "hier", "tree")
    if use_hier:
        always_bcast = True
    uneven_only = mode in ("uneven", "rem", "pad")

    def _nbytes(t: torch.Tensor) -> int:
        try:
            return int(t.numel() * t.element_size())
        except Exception:
            return 0

    def _safe_acc(name: str, dt: float, nbytes: int = 0) -> None:
        """Seed new timer keys without editing sqs_sampling (_acc KeyErrors on miss)."""
        try:
            acc_fn(name, dt, nbytes)
            return
        except KeyError:
            pass
        try:
            import fairchem_xpu_parallel as fxp

            with fxp._stats_lock:
                fxp._collective_stats.setdefault(f"{name}_calls", 0)
                fxp._collective_stats.setdefault(f"{name}_s", 0.0)
                fxp._collective_stats.setdefault(f"{name}_bytes", 0)
            acc_fn(name, dt, nbytes)
        except Exception:
            log.debug("Phase2 timer skip for %s", name, exc_info=True)

    def _tiles_per_node() -> int:
        try:
            return max(1, int(os.environ.get("FXPU_TILES_PER_NODE", "12") or 12))
        except ValueError:
            return 12

    def _ensure_hier_groups(group, world: int, tpn: int) -> dict:
        """Create/cached per-node + leader process groups (all ranks must call)."""
        cache = getattr(gp_utils, "_fxpu_hier_pg_cache", None)
        if cache is None:
            cache = {}
            gp_utils._fxpu_hier_pg_cache = cache
        key = (id(group), int(world), int(tpn))
        hit = cache.get(key)
        if hit is not None:
            return hit
        node_groups = []
        for node in range(world // tpn):
            ranks = list(range(node * tpn, (node + 1) * tpn))
            node_groups.append(dist.new_group(ranks=ranks))
        leader_ranks = list(range(0, world, tpn))
        leader_group = dist.new_group(ranks=leader_ranks)
        # Pair groups for recursive-doubling leader allgather (power-of-2 nodes).
        # All world ranks must call new_group for each pair.
        n_nodes = world // tpn
        pair_groups: dict[tuple[int, int], object] = {}
        if n_nodes >= 2 and (n_nodes & (n_nodes - 1)) == 0:
            log2n = n_nodes.bit_length() - 1
            for s in range(log2n):
                dist_n = 1 << s
                for node in range(n_nodes):
                    partner = node ^ dist_n
                    if node < partner:
                        ra, rb = node * tpn, partner * tpn
                        pair_groups[(node, partner)] = dist.new_group(
                            ranks=[ra, rb]
                        )
        hit = {
            "node_groups": node_groups,
            "leader_group": leader_group,
            "pair_groups": pair_groups,
            "tpn": tpn,
            "world": world,
        }
        cache[key] = hit
        log.info(
            "FXPU Phase2: hierarchical PG ready nodes=%s tpn=%s leaders=%s pairs=%s",
            world // tpn,
            tpn,
            leader_ranks,
            sorted(pair_groups.keys()),
        )
        return hit

    def _bcast_all_gather(input_tensor: torch.Tensor, group) -> tuple:  # type: ignore[no-untyped-def]
        """Equal-sized gather via W broadcasts (XCCL-safe substitute for all_gather)."""
        world = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        x = input_tensor.contiguous()
        t0 = _time.perf_counter()
        outs: list[torch.Tensor] = []
        for src in range(world):
            if src == rank:
                buf = x.clone()
            else:
                buf = torch.empty(x.shape, dtype=x.dtype, device=x.device)
            dist.broadcast(buf, src=src, group=group)
            outs.append(buf)
        dt = _time.perf_counter() - t0
        nb = _nbytes(x) * world
        # Precise bucket only (do not also hit all_gather — that double-counts coll_s).
        _safe_acc("broadcast", dt, nb)
        return tuple(outs)

    def _hier_bcast_all_gather(input_tensor: torch.Tensor, group) -> tuple:  # type: ignore[no-untyped-def]
        """Two-level broadcast gather: intra-node then inter-node leaders.

        Falls back to flat ``_bcast_all_gather`` when W is not a multiple of
        ``FXPU_TILES_PER_NODE`` (or single-node).
        """
        world = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        tpn = _tiles_per_node()
        if world <= tpn or (world % tpn) != 0:
            return _bcast_all_gather(input_tensor, group)

        x = input_tensor.contiguous()
        n_nodes = world // tpn
        node = rank // tpn
        local = rank % tpn
        info = _ensure_hier_groups(group, world, tpn)
        ng = info["node_groups"][node]
        lg = info["leader_group"]

        # dist.broadcast src= is always the *global* rank (even with subgroups).
        leader_global = node * tpn

        t0 = _time.perf_counter()
        # Phase 1 — intra-node allgather (tpn broadcasts on node group).
        t_intra0 = _time.perf_counter()
        local_outs: list[torch.Tensor] = []
        for src_local in range(tpn):
            src_global = leader_global + src_local
            if local == src_local:
                buf = x.clone()
            else:
                buf = torch.empty(x.shape, dtype=x.dtype, device=x.device)
            dist.broadcast(buf, src=src_global, group=ng)
            local_outs.append(buf)
        dt_intra = _time.perf_counter() - t_intra0
        node_payload = torch.cat(local_outs, dim=0).contiguous()

        # Optional chunked inter broadcasts (FXPU_XCCL_HIER_INTER_CHUNKS>1):
        # more smaller broadcasts — can help if large-message path is pathological.
        try:
            n_inter_chunks = max(
                1, int(os.environ.get("FXPU_XCCL_HIER_INTER_CHUNKS", "1") or 1)
            )
        except ValueError:
            n_inter_chunks = 1

        inter_mode = (
            os.environ.get("FXPU_XCCL_HIER_INTER", "broadcast") or "broadcast"
        ).strip().lower()
        use_inter_ag = inter_mode in ("all_gather", "native", "ag")
        t_inter0 = _time.perf_counter()
        if local == 0:
            if use_inter_ag:
                # Equal shards across leaders → stock all_gather (hung on Aurora XCCL;
                # keep behind env for A/B only).
                node_chunks = [
                    torch.empty_like(node_payload) for _ in range(n_nodes)
                ]
                dist.all_gather(node_chunks, node_payload, group=lg)
            elif inter_mode in ("ring", "p2p", "isend"):
                # Ring allgather on leaders only (bidirectional isend/irecv).
                # Avoids sequential per-source broadcast; same O(n*(n-1)*P) volume
                # but can use link better than n root broadcasts.
                # NOTE: hung on Aurora XCCL (C6n) — keep for A/B only.
                node_chunks = [
                    torch.empty_like(node_payload) for _ in range(n_nodes)
                ]
                node_chunks[node] = node_payload.contiguous()
                next_global = ((node + 1) % n_nodes) * tpn
                prev_global = ((node - 1) % n_nodes) * tpn
                send_buf = node_chunks[node]
                for step in range(n_nodes - 1):
                    recv_buf = torch.empty_like(send_buf)
                    # dst/src are global ranks (same rule as broadcast src=).
                    req_send = dist.isend(send_buf, dst=next_global, group=lg)
                    req_recv = dist.irecv(recv_buf, src=prev_global, group=lg)
                    if req_send is not None:
                        req_send.wait()
                    if req_recv is not None:
                        req_recv.wait()
                    src_node = (node - step - 1) % n_nodes
                    node_chunks[src_node] = recv_buf
                    send_buf = recv_buf
            elif inter_mode in ("rd", "recursive_doubling", "doubling"):
                # Recursive doubling among leaders via 2-rank broadcast pairs
                # (no P2P). Power-of-2 n_nodes only; else fall back to flat
                # leader broadcasts.
                pair_groups = info.get("pair_groups") or {}
                if (n_nodes & (n_nodes - 1)) != 0 or not pair_groups:
                    node_chunks = []
                    for src_node in range(n_nodes):
                        src_global = src_node * tpn
                        if node == src_node:
                            buf = node_payload
                        else:
                            buf = torch.empty_like(node_payload)
                        dist.broadcast(buf, src=src_global, group=lg)
                        node_chunks.append(buf)
                else:
                    buf = node_payload.contiguous()
                    log2n = n_nodes.bit_length() - 1
                    for s in range(log2n):
                        dist_n = 1 << s
                        partner = node ^ dist_n
                        a, b = (node, partner) if node < partner else (partner, node)
                        pg = pair_groups[(a, b)]
                        a_g, b_g = a * tpn, b * tpn
                        # Optional chunked pair exchange (same CHUNKS knob as
                        # flat inter); helps when large 2-rank broadcasts are
                        # pathological on TCP/XCCL (C6e lesson).
                        if n_inter_chunks <= 1:
                            buf_a = (
                                buf.clone() if node == a else torch.empty_like(buf)
                            )
                            dist.broadcast(buf_a, src=a_g, group=pg)
                            buf_b = (
                                buf.clone() if node == b else torch.empty_like(buf)
                            )
                            dist.broadcast(buf_b, src=b_g, group=pg)
                            buf = torch.cat([buf_a, buf_b], dim=0).contiguous()
                        else:
                            parts = list(buf.tensor_split(n_inter_chunks, dim=0))
                            got_a: list[torch.Tensor] = []
                            got_b: list[torch.Tensor] = []
                            for part in parts:
                                ba = (
                                    part.clone()
                                    if node == a
                                    else torch.empty_like(part)
                                )
                                dist.broadcast(ba, src=a_g, group=pg)
                                bb = (
                                    part.clone()
                                    if node == b
                                    else torch.empty_like(part)
                                )
                                dist.broadcast(bb, src=b_g, group=pg)
                                got_a.append(ba)
                                got_b.append(bb)
                            buf = torch.cat(
                                [
                                    torch.cat(got_a, dim=0),
                                    torch.cat(got_b, dim=0),
                                ],
                                dim=0,
                            ).contiguous()
                    # Split full gather back into per-node chunks for cat below.
                    node_chunks = list(buf.split(int(node_payload.shape[0]), dim=0))
                    if len(node_chunks) != n_nodes:
                        raise RuntimeError(
                            f"rd gather split mismatch: got {len(node_chunks)} "
                            f"want {n_nodes}"
                        )
            elif n_inter_chunks <= 1:
                # Optional: overlap broadcasts from all leaders (async_op).
                # FXPU_XCCL_HIER_INTER_OVERLAP_SRC=1
                use_ovsrc = (
                    os.environ.get("FXPU_XCCL_HIER_INTER_OVERLAP_SRC", "0") or "0"
                ).strip().lower() in ("1", "true", "yes", "on")
                if use_ovsrc:
                    works = []
                    node_chunks = []
                    for src_node in range(n_nodes):
                        src_global = src_node * tpn
                        if node == src_node:
                            buf = node_payload.clone()
                        else:
                            buf = torch.empty_like(node_payload)
                        works.append(
                            dist.broadcast(
                                buf, src=src_global, group=lg, async_op=True
                            )
                        )
                        node_chunks.append(buf)
                    for w in works:
                        if w is not None:
                            w.wait()
                else:
                    node_chunks = []
                    for src_node in range(n_nodes):
                        src_global = src_node * tpn
                        if node == src_node:
                            buf = node_payload
                        else:
                            buf = torch.empty_like(node_payload)
                        dist.broadcast(buf, src=src_global, group=lg)
                        node_chunks.append(buf)
            else:
                # Chunk along dim0; broadcast each slice from each leader.
                # FXPU_XCCL_HIER_INTER_ASYNC=1 — pipeline chunks within one source.
                # FXPU_XCCL_HIER_INTER_OVERLAP_SRC=1 — for each chunk, all sources
                # in flight (better link fill than serial roots).
                use_async = (
                    os.environ.get("FXPU_XCCL_HIER_INTER_ASYNC", "0") or "0"
                ).strip().lower() in ("1", "true", "yes", "on")
                use_ovsrc = (
                    os.environ.get("FXPU_XCCL_HIER_INTER_OVERLAP_SRC", "0") or "0"
                ).strip().lower() in ("1", "true", "yes", "on")
                parts = list(node_payload.tensor_split(n_inter_chunks, dim=0))
                if use_ovsrc:
                    # FXPU_XCCL_HIER_INTER_PIPELINE_DEPTH=D (default 1): keep up to
                    # D chunk-waves in flight before waiting (same math; may fill
                    # TCP better on 2-node). D=1 matches prior wait-per-wave.
                    try:
                        pipe_depth = max(
                            1,
                            int(
                                os.environ.get(
                                    "FXPU_XCCL_HIER_INTER_PIPELINE_DEPTH", "1"
                                )
                                or 1
                            ),
                        )
                    except ValueError:
                        pipe_depth = 1
                    got_by_src: list[list[torch.Tensor | None]] = [
                        [None] * len(parts) for _ in range(n_nodes)
                    ]
                    inflight: list[tuple[list, int]] = []
                    for ci, part in enumerate(parts):
                        works = []
                        bufs: list[torch.Tensor] = []
                        for src_node in range(n_nodes):
                            src_global = src_node * tpn
                            if node == src_node:
                                buf = part.clone()
                            else:
                                buf = torch.empty_like(part)
                            works.append(
                                dist.broadcast(
                                    buf, src=src_global, group=lg, async_op=True
                                )
                            )
                            bufs.append(buf)
                        inflight.append((works, ci))
                        for src_node, buf in enumerate(bufs):
                            got_by_src[src_node][ci] = buf
                        while len(inflight) >= pipe_depth:
                            wlist, _ = inflight.pop(0)
                            for w in wlist:
                                if w is not None:
                                    w.wait()
                    for wlist, _ in inflight:
                        for w in wlist:
                            if w is not None:
                                w.wait()
                    node_chunks = [
                        torch.cat(g, dim=0).contiguous()  # type: ignore[arg-type]
                        for g in got_by_src
                    ]
                else:
                    node_chunks = []
                    for src_node in range(n_nodes):
                        src_global = src_node * tpn
                        got: list[torch.Tensor] = []
                        if use_async and n_inter_chunks > 1:
                            works = []
                            for part in parts:
                                if node == src_node:
                                    buf = part.clone()
                                else:
                                    buf = torch.empty_like(part)
                                works.append(
                                    dist.broadcast(
                                        buf, src=src_global, group=lg, async_op=True
                                    )
                                )
                                got.append(buf)
                            for w in works:
                                if w is not None:
                                    w.wait()
                        else:
                            for part in parts:
                                if node == src_node:
                                    buf = part
                                else:
                                    buf = torch.empty_like(part)
                                dist.broadcast(buf, src=src_global, group=lg)
                                got.append(buf)
                        node_chunks.append(torch.cat(got, dim=0).contiguous())
            full = torch.cat(node_chunks, dim=0).contiguous()
        else:
            full = torch.empty(
                (world * int(x.shape[0]),) + tuple(x.shape[1:]),
                dtype=x.dtype,
                device=x.device,
            )
        dt_inter = _time.perf_counter() - t_inter0

        # Phase 3 — node leader scatters full gather to local ranks.
        t_scat0 = _time.perf_counter()
        dist.broadcast(full, src=leader_global, group=ng)
        dt_scat = _time.perf_counter() - t_scat0

        outs = list(full.split(int(x.shape[0]), dim=0))
        if len(outs) != world:
            raise RuntimeError(
                f"hier gather split mismatch: got {len(outs)} want {world}"
            )
        dt = _time.perf_counter() - t0
        nb = _nbytes(x) * world
        _safe_acc("broadcast", dt, nb)
        _safe_acc("broadcast_intra", dt_intra, _nbytes(x) * tpn)
        _safe_acc("broadcast_inter", dt_inter, _nbytes(node_payload) * n_nodes)
        _safe_acc("broadcast_scatter", dt_scat, _nbytes(full))
        return tuple(outs)

    def _gather_forward(input_tensor: torch.Tensor, group) -> tuple:  # type: ignore[no-untyped-def]
        if use_hier:
            return _hier_bcast_all_gather(input_tensor, group)
        return _bcast_all_gather(input_tensor, group)

    def _rs_chunk() -> int:
        # 0 = disabled (preferred). Chunking increased NotPresent rate (broke N=21).
        try:
            return max(0, int(os.environ.get("FXPU_XCCL_RS_CHUNK", "0") or 0))
        except ValueError:
            return 0

    def _pad_align() -> int:
        try:
            return max(1, int(os.environ.get("FXPU_XCCL_PAD_ALIGN", "32") or 32))
        except ValueError:
            return 32

    def _reduce_scatter_safe(pieces, group, rank):  # type: ignore[no-untyped-def]
        """SumGrad backward without native reduce_scatter (NotPresent on some sizes).

        Default ``reduce``: for dest in 0..W-1, ``dist.reduce(pieces[dest] → dest)``.
        Equivalent to reduce_scatter; uses all-reduce-family collectives (sockets IPC OK).

        ``FXPU_XCCL_RS_MODE``:
          - ``auto`` (default): native reduce_scatter if padded < FXPU_XCCL_RS_REDUCE_MIN
            (19000); else reduce-loop (covers N=31 @ W=12)
          - ``reduce_scatter`` / ``native``: always FairChem stock reduce_scatter
          - ``reduce`` / ``loop``: always dist.reduce loop (safe, slow at large W)
          - ``all_to_all``: needs CCL_ZE_IPC_EXCHANGE=drmfd|pidfd
        """
        mode_rs = (os.environ.get("FXPU_XCCL_RS_MODE", "auto") or "auto").strip().lower()
        local = pieces[rank]
        padded = int(local.shape[0])
        chunk = _rs_chunk()

        def _one(sub_pieces):  # type: ignore[no-untyped-def]
            # auto: native reduce_scatter (fast). N=21/22/24 OK; N=31 needs reduce loop.
            use_native = mode_rs in ("reduce_scatter", "rs", "native") or (
                mode_rs in ("auto", "") and padded < int(os.environ.get("FXPU_XCCL_RS_REDUCE_MIN", "19000") or 19000)
            )
            if mode_rs in ("reduce", "loop"):
                use_native = False
            if use_native:
                from torch.distributed.nn.functional import reduce_scatter

                o = torch.empty_like(sub_pieces[rank])
                return reduce_scatter(o, sub_pieces, group=group), "reduce_scatter"
            if mode_rs in ("all_to_all", "a2a"):
                world = dist.get_world_size(group=group)
                send = [p.contiguous() for p in sub_pieces]
                recv = [torch.empty_like(send[0]) for _ in range(world)]
                dist.all_to_all(recv, send, group=group)
                out = recv[0]
                for t in recv[1:]:
                    out = out + t
                return out, "all_to_all"
            # reduce loop (safe for large uneven shards e.g. N=31 @ W=12)
            world = dist.get_world_size(group=group)
            out = None
            for dest in range(world):
                buf = sub_pieces[dest].contiguous().clone()
                dist.reduce(buf, dst=dest, op=dist.ReduceOp.SUM, group=group)
                if dest == rank:
                    out = buf
            assert out is not None
            return out, "reduce_loop"

        if chunk <= 0 or padded <= chunk:
            t0 = _time.perf_counter()
            out, kind = _one(pieces)
            # Label by algorithm (was wrongly always "all_reduce").
            _safe_acc(kind, _time.perf_counter() - t0, _nbytes(local))
            return out
        parts = []
        t0 = _time.perf_counter()
        nbytes = 0
        kind_last = "reduce_loop"
        for start in range(0, padded, chunk):
            end = min(start + chunk, padded)
            sub = [p[start:end].contiguous() for p in pieces]
            part, kind_last = _one(sub)
            parts.append(part)
            nbytes += _nbytes(sub[rank])
        _safe_acc(kind_last, _time.perf_counter() - t0, nbytes)
        return torch.cat(parts, dim=0)

    class _BcastGatherGradPadded(torch.autograd.Function):
        _fxpu_marker = UNEVEN_GATHER_MARKER

        @staticmethod
        @torch.compiler.disable
        def forward(ctx, input: torch.Tensor):  # type: ignore[no-untyped-def]
            ctx.rank = gp_utils.get_gp_rank()
            ctx.group = gp_utils.get_gp_group()
            return _gather_forward(input, ctx.group)

        @staticmethod
        @torch.compiler.disable
        def backward(ctx, *grad_outputs):  # type: ignore[no-untyped-def]
            return grad_outputs[ctx.rank]

    class _BcastGatherSumGradPadded(torch.autograd.Function):
        """Broadcast gather forward + chunked XCCL reduce_scatter backward."""

        _fxpu_marker = UNEVEN_GATHER_MARKER

        @staticmethod
        @torch.compiler.disable
        def forward(ctx, input: torch.Tensor):  # type: ignore[no-untyped-def]
            ctx.rank = gp_utils.get_gp_rank()
            ctx.group = gp_utils.get_gp_group()
            ctx.shape = tuple(input.shape)
            ctx.device = input.device
            ctx.dtype = input.dtype
            return _gather_forward(input, ctx.group)

        @staticmethod
        @torch.compiler.disable
        def backward(ctx, *grad_outputs):  # type: ignore[no-untyped-def]
            padded = ctx.shape[0]
            for h in grad_outputs:
                if h is not None:
                    padded = int(h.shape[0])
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
            out = _reduce_scatter_safe(pieces, ctx.group, ctx.rank)
            if out.shape[0] != ctx.shape[0]:
                return out[: ctx.shape[0]].contiguous()
            return out

    # Replace Autograd classes when always-broadcast so direct .apply is covered.
    _stock_sum = gp_utils.GatherFromModelParallelRegionSumGradPadded
    _stock_grad = gp_utils.GatherFromModelParallelRegionGradPadded
    if always_bcast:
        gp_utils.GatherFromModelParallelRegionSumGradPadded = _BcastGatherSumGradPadded
        gp_utils.GatherFromModelParallelRegionGradPadded = _BcastGatherGradPadded

    def _gather_impl(input, natoms, *, sum_grad: bool):  # type: ignore[no-untyped-def]
        world_size = gp_utils.get_gp_world_size()
        size_list = gp_utils.size_list_fn(int(natoms), world_size)
        padded_size = int(natoms) // world_size + (
            1 if int(natoms) % world_size != 0 else 0
        )
        align = _pad_align()
        if padded_size % align:
            padded_size += align - (padded_size % align)
        padded = _pad_input_contiguous(input, padded_size)
        uneven = int(natoms) % world_size != 0
        use_bcast = always_bcast or (uneven_only and uneven)
        if use_bcast:
            gather_cls = (
                _BcastGatherSumGradPadded if sum_grad else _BcastGatherGradPadded
            )
        else:
            gather_cls = _stock_sum if sum_grad else _stock_grad
        tensor_list_w_padding = gather_cls.apply(padded)
        return torch.cat(
            [
                t.narrow(0, 0, s) if t.shape[0] != s else t
                for t, s in zip(tensor_list_w_padding, size_list)
            ],
            dim=0,
        )

    def _gather(input, natoms):  # type: ignore[no-untyped-def]
        return _gather_impl(input, natoms, sum_grad=False)

    def _gather_sum_grad(input, natoms):  # type: ignore[no-untyped-def]
        return _gather_impl(input, natoms, sum_grad=True)

    gp_utils.pad_input = _pad_input_contiguous  # type: ignore[assignment]
    gp_utils.gather_from_model_parallel_region = _gather  # type: ignore[assignment]
    gp_utils.gather_from_model_parallel_region_sum_grad = (  # type: ignore[assignment]
        _gather_sum_grad
    )
    gp_utils._fxpu_xccl_uneven_gather = True
    log.info(
        "FXPU Phase2: GP gather → XCCL %s + %s SumGrad bwd "
        "(marker=%s mode=%s always=%s hier=%s rs_chunk=%s pad_align=%s)",
        "hierarchical-broadcast" if use_hier else "broadcast",
        (os.environ.get("FXPU_XCCL_RS_MODE", "auto") or "auto"),
        UNEVEN_GATHER_MARKER,
        mode,
        always_bcast,
        use_hier,
        _rs_chunk(),
        _pad_align(),
    )


def apply_native_xccl_collectives(*, acc_fn, force_patch_fn) -> str:
    """Skip CPU staging; time native XPU collectives; keep FairChem GP Autograd.

    ``acc_fn(name, dt, nbytes=0)`` and ``force_patch_fn()`` come from the
    shimmed ``fairchem_xpu_parallel`` module.
    """
    import torch
    import torch.distributed as dist
    from fairchem.core.common import gp_utils

    if getattr(gp_utils, "_fxpu_xccl_patched", False):
        force_patch_fn()
        return MARKER

    _orig_all_reduce = dist.all_reduce
    _orig_all_gather = dist.all_gather

    def _nbytes(t: torch.Tensor) -> int:
        try:
            return int(t.numel() * t.element_size())
        except Exception:
            return 0

    def _timed_all_reduce(tensor, op=None, group=None, async_op=False):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        work = _orig_all_reduce(tensor, op=op, group=group, async_op=async_op)
        if async_op and work is not None:
            work.wait()
        acc_fn("all_reduce", time.perf_counter() - t0, _nbytes(tensor))
        return None if async_op else work

    def _timed_all_gather(tensor_list, tensor, group=None, async_op=False):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        work = _orig_all_gather(tensor_list, tensor, group=group, async_op=async_op)
        if async_op and work is not None:
            work.wait()
        n = _nbytes(tensor) * max(len(tensor_list), 1)
        acc_fn("all_gather", time.perf_counter() - t0, n)
        return None if async_op else work

    dist._fxpu_orig_all_reduce = _orig_all_reduce  # type: ignore[attr-defined]
    dist._fxpu_orig_all_gather = _orig_all_gather  # type: ignore[attr-defined]
    dist.all_reduce = _timed_all_reduce  # type: ignore[assignment]
    dist.all_gather = _timed_all_gather  # type: ignore[assignment]

    # Uneven pad+all_gather → XCCL broadcast gather; equal splits keep native all_gather.
    try:
        _apply_xccl_uneven_gather_safe(gp_utils, acc_fn=acc_fn)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("FXPU Phase2 uneven-gather patch skipped: %s", exc)

    force_patch_fn()

    gp_utils._fxpu_xccl_patched = True
    # Prevent accidental double-application if something checks the gloo flag.
    gp_utils._fxpu_gloo_xpu_patched = True
    log.info(
        "FXPU Phase2: native XCCL collectives (no CPU staging); marker=%s",
        MARKER,
    )

    # Phase 4: odd-world all_reduce workaround + reduce_scatter timers.
    try:
        from patches import phase4_collectives as p4

        p4.apply_phase4_collective_reductions(acc_fn=acc_fn)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("FXPU Phase4 collective patch skipped: %s", exc)

    return MARKER


def install_init_process_group_hook() -> None:
    """Before xccl init: ATL env + device_id=xpu:0."""
    import torch
    import torch.distributed as dist

    if getattr(dist.init_process_group, "_fxpu_phase2_hook", False):
        return

    _real = dist.init_process_group

    def _init(*args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
        backend = kwargs.get("backend")
        if backend is None and args:
            backend = args[0]
        be = str(backend or "").strip().lower()
        if be in _XCCL_ALIASES:
            rank = int(kwargs.get("rank", 0))
            world = int(kwargs.get("world_size", 1))
            # Multi-node: CCL local_* is per-node tile count, not global world.
            try:
                per = max(1, int(os.environ.get("FXPU_TILES_PER_NODE", "12")))
            except ValueError:
                per = 12
            multi = world > per or os.environ.get(
                "FXPU_PHASE6_MULTINODE", ""
            ).strip().lower() in ("1", "true", "on", "yes")
            if multi:
                local_rank = rank % per
                local_size = per
            else:
                local_rank = rank
                local_size = world
            # Scrub PMI + force launcher=none immediately before init
            # (Ray actors inherit PMI_SIZE=1 from mpiexec -n 1 ray start).
            configure_xccl_env_for_ray(
                local_rank=local_rank,
                local_size=local_size,
                world_size=world,
                pin_tcp=True,
            )
            os.environ["CCL_PROCESS_LAUNCHER"] = "none"
            # Keep ZE mask on local tile through XCCL init (stock may set global).
            os.environ["ZE_AFFINITY_MASK"] = str(local_rank)
            if "device_id" not in kwargs:
                kwargs["device_id"] = torch.device("xpu:0")
            if not dist.is_xccl_available():
                raise RuntimeError(
                    "FXPU_DIST_BACKEND=xccl but torch.distributed.is_xccl_available() is False"
                )
            mask = os.environ.get("ZE_AFFINITY_MASK", "")
            try:
                mask_i = int(mask)
            except ValueError as exc:
                raise RuntimeError(
                    f"FXPU XCCL: ZE_AFFINITY_MASK={mask!r} not an int (rank={rank})"
                ) from exc
            if multi and not (0 <= mask_i < per):
                raise RuntimeError(
                    f"FXPU XCCL: ZE_AFFINITY_MASK={mask_i} out of tile range "
                    f"[0,{per}) on multinode rank={rank} — refusing poisoned init"
                )
        return _real(*args, **kwargs)

    _init._fxpu_phase2_hook = True  # type: ignore[attr-defined]
    dist.init_process_group = _init  # type: ignore[assignment]
    log.info("FXPU Phase2: hooked dist.init_process_group for XCCL device_id/ATL")
