"""Phase 6: multi-node Ray + local-tile affinity for W > tiles/node.

``energy.uma_predict_unit`` (sqs_sampling, frozen) sets
``num_workers_per_node=max(n_workers, 12)``, and Phase 3 bumps per-node up to
``num_workers`` — both force a single-node cluster. This patch (outside
sqs_sampling) restores true multi-node launch when ``num_workers`` exceeds
``FXPU_TILES_PER_NODE`` (default 12):

1. Cap ``num_workers_per_node`` at tiles/node
2. ``ZE_AFFINITY_MASK = worker_id % tiles_per_node``
3. Boot Ray head/workers across ``PBS_NODEFILE`` hosts
4. Per-node ``CCL_LOCAL_RANK`` / ``CCL_LOCAL_SIZE`` for XCCL

Enable: ``FXPU_PHASE6_MULTINODE=1`` (PBS 6b) or auto when ``num_workers > tiles``.
Disable: ``FXPU_PHASE6_MULTINODE=0``.
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
import math
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MARKER = "fxpu_phase6_multinode_v15"
_PHASE6_APPLIED = False


def tiles_per_node() -> int:
    return max(1, int(os.environ.get("FXPU_TILES_PER_NODE", "12")))


def multinode_enabled(num_workers: int | None = None) -> bool:
    flag = os.environ.get("FXPU_PHASE6_MULTINODE", "auto").strip().lower()
    if flag in ("0", "false", "off", "no"):
        return False
    if flag in ("1", "true", "on", "yes"):
        return True
    # auto
    if num_workers is None:
        return False
    return int(num_workers) > tiles_per_node()


def _pbs_hosts() -> list[str]:
    nf = os.environ.get("PBS_NODEFILE")
    if not nf or not Path(nf).is_file():
        return []
    hosts: list[str] = []
    for line in Path(nf).read_text(encoding="utf-8").splitlines():
        h = line.strip()
        if h and h not in hosts:
            hosts.append(h)
    return hosts


def _local_tile(worker_id: int) -> int:
    return int(worker_id) % tiles_per_node()


def _runtime_env_vars(worker_id: int, sqs_path: str) -> dict[str, str]:
    from patches import phase2_xccl as p2

    tile = _local_tile(worker_id)
    per = tiles_per_node()
    node_id = int(worker_id) // per
    env: dict[str, str] = {
        "ZE_FLAT_DEVICE_HIERARCHY": "FLAT",
        "ZE_AFFINITY_MASK": str(tile),
        "PYTHONPATH": os.environ.get("PYTHONPATH", sqs_path),
        "FXPU_PHASE3_LAUNCH": "1",
        "FXPU_PHASE6_MULTINODE": "1",
        "FXPU_TILES_PER_NODE": str(per),
        "FXPU_EDGEWISE_AC_BISECT": os.environ.get("FXPU_EDGEWISE_AC_BISECT", "2"),
        "FXPU_NODE_ID": str(node_id),
        "FXPU_WORKER_ID": str(int(worker_id)),
        "CCL_LOCAL_RANK": str(tile),
        "LOCAL_RANK": str(tile),
        "CCL_LOCAL_SIZE": str(per),
        "LOCAL_WORLD_SIZE": str(per),
    }
    if p2.use_xccl():
        env.update(p2.ray_worker_xccl_env_vars())
        env["CCL_LOCAL_RANK"] = str(tile)
        env["LOCAL_RANK"] = str(tile)
        env["CCL_LOCAL_SIZE"] = str(per)
        env["LOCAL_WORLD_SIZE"] = str(per)
    for key in (
        "FXPU_DIST_BACKEND",
        "FXPU_FI_PROVIDER",
        "FI_PROVIDER",
        "FXPU_FI_TCP_IFACE",
        # Do NOT propagate FI_TCP_IFACE — each node must pin its own HSN iface.
        "CCL_ATL_TRANSPORT",
        "CCL_PROCESS_LAUNCHER",
        "CCL_ZE_IPC_EXCHANGE",
        "CCL_WORKER_COUNT",
        "CCL_ROOT",
        "LD_LIBRARY_PATH",
        "PATH",
        "CONDA_PREFIX",
        "RAY_TMPDIR",
        "TMPDIR",
    ):
        val = os.environ.get(key)
        if val:
            env[key] = val
    # Force non-MPI launcher for XCCL under Ray (actors inherit PMI from
    # mpiexec -n 1 ray start otherwise).
    if p2.use_xccl():
        env["CCL_PROCESS_LAUNCHER"] = "none"
        env.pop("CCL_KVS_MODE", None)
        for k in list(env):
            if any(
                k.startswith(p)
                for p in (
                    "PMI_",
                    "PMIX_",
                    "PALS_",
                    "I_MPI_",
                    "MPIR_",
                    "HYDRA_",
                    "OMPI_",
                    "MPI_",
                )
            ):
                env.pop(k, None)
    # Match Phase3: propagate all FXPU_* so Wigner/SKIP/Phase1 knobs reach actors.
    for key, val in os.environ.items():
        if key.startswith("FXPU_") and val:
            env[key] = val
    # Propagate RxM / universe size / TCP knobs for large-message A/B (C8/C11).
    # Never propagate FI_TCP_IFACE — per-node HSN pin is set locally.
    for key, val in os.environ.items():
        if not val:
            continue
        if key.startswith("FI_OFI_RXM_") or key in ("FI_UNIVERSE_SIZE", "FI_TCP_NODELAY"):
            env[key] = val
        elif key.startswith("FI_TCP_") and key != "FI_TCP_IFACE":
            env[key] = val
    # C7b: force per-node HSN pin LAST so remotes cannot silently stay on hsn0
    # (C7a only logged driver idx=0; result was flat vs C6e).
    # Skip if PBS already set a global FXPU_FI_TCP_IFACE override.
    stripe = (os.environ.get("FXPU_TCP_IFACE_STRIPE") or "").strip().lower()
    if stripe in ("node", "host") and not (os.environ.get("FXPU_FI_TCP_IFACE") or "").strip():
        try:
            mod = max(1, int(os.environ.get("FXPU_TCP_IFACE_STRIPE_MOD", "8") or 8))
        except ValueError:
            mod = 8
        env["FXPU_FI_TCP_IFACE"] = f"hsn{node_id % mod}"
        log.info(
            "Phase6: worker=%s node=%s FXPU_FI_TCP_IFACE=%s (stripe=node)",
            worker_id,
            node_id,
            env["FXPU_FI_TCP_IFACE"],
        )
    return env


def _ray_bin() -> str:
    return shutil.which("ray") or "ray"


def _node_ip(host: str) -> str:
    """Resolve a routable IP for Ray (prefer HSN hostname as given)."""
    try:
        return socket.gethostbyname(host)
    except OSError:
        return host


def _start_ray_cluster(hosts: list[str], per_node: int, ray_tmp: Path) -> str:
    """Start Ray head on hosts[0], workers on remaining; return address.

    Remote starts use ``mpiexec`` (not SSH) so processes stay in the PBS/XPU
    cgroup — SSH to compute nodes often yields ``XPU device count is zero``.
    """
    import shlex

    if not hosts:
        raise RuntimeError("Phase6 multi-node requires PBS_NODEFILE hosts")

    head = hosts[0]
    head_ip = _node_ip(head)
    port = int(os.environ.get("FXPU_RAY_PORT", "6379"))
    address = f"{head_ip}:{port}"
    ray_bin = _ray_bin()
    res = f'{{"xpu_tile": {float(per_node)}}}'
    user = os.environ.get("USER", "hen")[:8]
    mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")

    def _run_on_host(host: str, cmd: str, *, log_path: str | None = None) -> None:
        short = host.split(".")[0]
        here = socket.gethostname().split(".")[0]
        if short == here:
            log.info("Phase6 local: %s", cmd)
            r = subprocess.run(cmd, shell=True)
            if r.returncode != 0:
                detail = ""
                if log_path and Path(log_path).is_file():
                    detail = Path(log_path).read_text(encoding="utf-8", errors="replace")[-4000:]
                raise RuntimeError(
                    f"Phase6 local cmd failed rc={r.returncode}: {cmd}\n{detail}"
                )
            return
        if not mpiexec:
            raise RuntimeError(
                "Phase6 multi-node needs mpiexec (SSH hides XPUs on Aurora)"
            )
        # Stay inside PBS allocation cgroup for Level Zero / XPU visibility.
        # Unset ZE_AFFINITY_MASK in the remote shell even if --envall exported it
        # (PBS scripts often set MASK=0 for the driver tile).
        wrapped = (
            f"{mpiexec} --hosts {host} -n 1 --ppn 1 --envall "
            f"bash -lc {shlex.quote(cmd)}"
        )
        log.info("Phase6 mpiexec %s: %s", host, cmd)
        r = subprocess.run(wrapped, shell=True)
        if r.returncode != 0:
            detail = ""
            if log_path:
                dump = (
                    f"{mpiexec} --hosts {host} -n 1 --ppn 1 --envall "
                    f"bash -lc {shlex.quote(f'cat {log_path} 2>/dev/null | tail -c 8000')}"
                )
                try:
                    d = subprocess.run(
                        dump, shell=True, capture_output=True, text=True, timeout=60
                    )
                    detail = (d.stdout or "") + (d.stderr or "")
                except Exception as exc:  # noqa: BLE001
                    detail = f"(failed to fetch {log_path}: {exc})"
            raise RuntimeError(
                f"Phase6 mpiexec on {host} failed rc={r.returncode}: {cmd}\n{detail}"
            )

    def _truthy(name: str) -> bool:
        return (os.environ.get(name) or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    skip_prestop = _truthy("FXPU_RAY_SKIP_PRESTOP")
    # Parallel mpiexec prestop + worker starts (D1). Default off; set
    # FXPU_RAY_PARALLEL_BRINGUP=1 for cold A/B. Keeps prestop (D0 skip hurt load).
    parallel_bringup = _truthy("FXPU_RAY_PARALLEL_BRINGUP")

    if skip_prestop:
        log.info(
            "Phase6: skipping pre-start ray stop on %d hosts (FXPU_RAY_SKIP_PRESTOP)",
            len(hosts),
        )
    else:
        stop_cmd = f"unset ZE_AFFINITY_MASK; {ray_bin} stop >/dev/null 2>&1 || true"

        def _prestop_one(h: str) -> None:
            try:
                _run_on_host(h, stop_cmd)
            except Exception as exc:  # noqa: BLE001
                log.debug("ray stop on %s: %s", h, exc)

        if parallel_bringup and len(hosts) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            log.info(
                "Phase6: parallel pre-start ray stop on %d hosts (FXPU_RAY_PARALLEL_BRINGUP)",
                len(hosts),
            )
            with ThreadPoolExecutor(max_workers=len(hosts)) as ex:
                futs = [ex.submit(_prestop_one, h) for h in hosts]
                for fut in as_completed(futs):
                    fut.result()
        else:
            for h in hosts:
                _prestop_one(h)

        time.sleep(1.0)

    # Raylets must see all tiles; actors set ZE_AFFINITY_MASK per local tile later.
    export_ze = "unset ZE_AFFINITY_MASK; export ZE_FLAT_DEVICE_HIERARCHY=FLAT; "

    head_log = f"/tmp/fxpu_ray_head_{user}.log"
    head_cmd = (
        f"{export_ze}{ray_bin} start --head --node-ip-address={head_ip} --port={port} "
        f"--num-cpus={per_node} --resources='{res}' "
        f"--temp-dir={ray_tmp} --disable-usage-stats "
        f"--dashboard-host=127.0.0.1 > {head_log} 2>&1"
    )
    _run_on_host(head, head_cmd, log_path=head_log)

    # Wait until GCS accepts status (3s sleep was flaky across allocations).
    ready = False
    for attempt in range(40):
        chk = subprocess.run(
            f"{ray_bin} status --address={address}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if chk.returncode == 0:
            ready = True
            log.info("Phase6 Ray head ready address=%s attempt=%s", address, attempt)
            break
        time.sleep(2.0)
    if not ready:
        detail = ""
        if Path(head_log).is_file():
            detail = Path(head_log).read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(
            f"Phase6 Ray head not ready at {address} after wait\n{detail}"
        )

    def _start_one_worker(h: str) -> None:
        hip = _node_ip(h)
        wtmp = f"/tmp/r{user}-ray-{os.getpid()}-{h.split('.')[0]}"
        wlog = f"/tmp/fxpu_ray_worker_{user}.log"
        wcmd = (
            f"{export_ze}mkdir -p {wtmp} && "
            f"{ray_bin} start --address={address} --node-ip-address={hip} "
            f"--num-cpus={per_node} --resources='{res}' "
            f"--temp-dir={wtmp} --disable-usage-stats "
            f"> {wlog} 2>&1"
        )
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                _run_on_host(h, wcmd, log_path=wlog)
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning(
                    "Phase6 worker ray start attempt %s on %s failed: %s",
                    attempt + 1,
                    h,
                    exc,
                )
                time.sleep(3.0)
        assert last_exc is not None
        raise last_exc

    worker_hosts = hosts[1:]
    if parallel_bringup and len(worker_hosts) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        log.info(
            "Phase6: parallel ray worker start on %d hosts (FXPU_RAY_PARALLEL_BRINGUP)",
            len(worker_hosts),
        )
        with ThreadPoolExecutor(max_workers=len(worker_hosts)) as ex:
            futs = {ex.submit(_start_one_worker, h): h for h in worker_hosts}
            for fut in as_completed(futs):
                fut.result()
    else:
        for h in worker_hosts:
            _start_one_worker(h)

    time.sleep(3.0)
    os.environ["RAY_ADDRESS"] = address
    log.info("Phase6 Ray cluster address=%s hosts=%s marker=%s", address, hosts, MARKER)
    return address


def apply_phase6_multinode_launch() -> str:
    """Replace Phase-3 ParallelMLIPPredictUnit XPU init with multi-node-aware path."""
    global _PHASE6_APPLIED
    if _PHASE6_APPLIED:
        return MARKER

    import copy
    import sys

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    from fairchem.core.units.mlip_unit.predict import (
        MLIPWorkerLocal,
        ParallelMLIPPredictUnit,
    )

    from patches import phase2_xccl as p2
    from patches import phase3_launch as p3

    # Ensure FXPU_PHASE6_MULTINODE is visible before demote check.
    os.environ.setdefault("FXPU_PHASE6_MULTINODE", "1")
    p2.maybe_demote_multinode_xccl_to_gloo()

    _prev_init = ParallelMLIPPredictUnit.__init__

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
            return _prev_init(
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

        per = tiles_per_node()
        # energy.py passes max(n_workers, 12) — cap for true multi-node.
        if multinode_enabled(num_workers):
            num_workers_per_node = per
        else:
            # Single-node: never keep the max(W,12) floor (W=2 must be 2).
            num_workers_per_node = int(num_workers)

        if not multinode_enabled(num_workers):
            return _prev_init(
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

        os.environ.setdefault("ZE_FLAT_DEVICE_HIERARCHY", "FLAT")
        os.environ["ZE_AFFINITY_MASK"] = "0"
        p3._pin_worker_cpus(0)

        project_root = Path(__file__).resolve().parents[1]
        shim_path = str(project_root / "shim")
        fxpu_path = str(project_root)
        sqs_path = str(project_root / "sqs_sampling")
        parts = [p for p in os.environ.get("PYTHONPATH", "").split(":") if p]
        for p in (sqs_path, fxpu_path, shim_path):
            if p in parts:
                parts.remove(p)
            parts.insert(0, p)
        os.environ["PYTHONPATH"] = ":".join(parts)

        if inference_settings is None:
            inference_settings = InferenceSettings()
        self.inference_settings = inference_settings
        self._dataset_to_tasks = {}
        self._validate_atoms_data_fn = None

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

        ray_tmp = p3._short_ray_tmpdir()
        if p2.use_xccl():
            # Configure ATL/OFI but do NOT pin FI_TCP_IFACE on the driver —
            # a wrong (bond0) pin previously broke Ray head start (8729110).
            # Each rank pins its own hsn* inside init_process_group.
            p2.configure_xccl_env_for_ray(
                local_rank=0, local_size=per, pin_tcp=False
            )
            os.environ.pop("FI_TCP_IFACE", None)
            os.environ["CCL_PROCESS_LAUNCHER"] = "none"
            p2.scrub_mpi_launcher_env()

        hosts = _pbs_hosts()
        need_hosts = math.ceil(int(num_workers) / per)
        if len(hosts) < need_hosts:
            raise RuntimeError(
                f"Phase6: need ≥{need_hosts} PBS hosts for W={num_workers} "
                f"(tiles/node={per}); got {len(hosts)}: {hosts}"
            )

        address = _start_ray_cluster(hosts[:need_hosts], per, ray_tmp)
        if not ray.is_initialized():
            os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
            ray.init(
                address=address,
                logging_level=log_level,
                ignore_reinit_error=True,
            )

        alive = [n for n in ray.nodes() if n.get("Alive")]
        log.info(
            "Phase6 ray.nodes alive=%s resources=%s",
            len(alive),
            [(n.get("NodeManagerHostname") or n.get("NodeManagerAddress"), n.get("Resources")) for n in alive],
        )
        if len(alive) < need_hosts:
            raise RuntimeError(
                f"Phase6: Ray cluster has {len(alive)} alive nodes, need {need_hosts}. "
                f"Second-node ray start likely failed (check /tmp/fxpu_ray_worker_*.log)."
            )

        self.atomic_data_on_device = None
        num_nodes = math.ceil(num_workers / num_workers_per_node)
        num_workers_on_node_array = [num_workers_per_node] * num_nodes
        if num_workers % num_workers_per_node > 0:
            num_workers_on_node_array[-1] = num_workers % num_workers_per_node

        # Pin ranks to nodes explicitly. STRICT_SPREAD PG with asymmetric
        # bundles (11+12 for in-process rank0) can place the 12-actor bundle
        # on the head → 13/11 host split and tile-0 double-book (hang).
        head_node_id = ray.get_runtime_context().get_node_id()
        other_node_ids: list[str] = []
        for n in ray.nodes():
            if not n.get("Alive"):
                continue
            nid = n.get("NodeID")
            if nid and nid != head_node_id:
                other_node_ids.append(str(nid))
        ray_node_ids = [str(head_node_id)] + other_node_ids
        if len(ray_node_ids) < num_nodes:
            raise RuntimeError(
                f"Phase6: need {num_nodes} Ray node IDs for affinity, got {len(ray_node_ids)}"
            )
        log.info(
            "Phase6 node affinity head=%s others=%s marker=%s",
            head_node_id,
            other_node_ids[: max(0, num_nodes - 1)],
            MARKER,
        )

        # Multi-node worker actor: local-tile affinity (not global worker_id).
        @ray.remote
        class FxpuPhase6XPUMLIPWorker(MLIPWorkerLocal):
            def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                worker_id = int(args[0]) if args else int(kwargs.get("worker_id", 0))
                tile = _local_tile(worker_id)
                # Scrub PMI inherited from ``mpiexec -n 1 ray start`` before
                # any XCCL / oneCCL import path can see PMI_SIZE=1.
                from patches import phase2_xccl as _p2

                _p2.scrub_mpi_launcher_env()
                os.environ["CCL_PROCESS_LAUNCHER"] = "none"
                os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "FLAT"
                os.environ["ZE_AFFINITY_MASK"] = str(tile)
                os.environ["CCL_LOCAL_RANK"] = str(tile)
                os.environ["LOCAL_RANK"] = str(tile)
                os.environ["CCL_LOCAL_SIZE"] = str(tiles_per_node())
                os.environ["LOCAL_WORLD_SIZE"] = str(tiles_per_node())
                # Fail fast with a clear message if affinity hid all tiles.
                try:
                    import torch

                    n_xpu = int(torch.xpu.device_count())
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"Phase6 worker {worker_id} tile={tile}: torch.xpu probe failed: {exc}"
                    ) from exc
                if n_xpu < 1:
                    raise RuntimeError(
                        f"Phase6 worker {worker_id} tile={tile}: XPU device count is 0 "
                        f"(ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK')}). "
                        "Raylet likely inherited MASK=0; unset it at ray start."
                    )
                p3._pin_worker_cpus(tile)
                import fairchem_xpu_parallel as fxp

                fxp.ensure_worker_patches()
                super().__init__(*args, **kwargs)
                logging.getLogger(__name__).info(
                    "FXPU Phase6 worker %s local_tile=%s mask=%s n_xpu=%s marker=%s",
                    worker_id,
                    tile,
                    os.environ.get("ZE_AFFINITY_MASK"),
                    n_xpu,
                    MARKER,
                )

            def ensure_distributed_setup(self) -> str:
                if not self.is_setup:
                    self._distributed_setup()
                return MARKER

        def _actor_opts(worker_id: int, node_id: str):  # type: ignore[no-untyped-def]
            return FxpuPhase6XPUMLIPWorker.options(
                num_cpus=1,
                num_gpus=0,
                resources={"xpu_tile": 1.0},
                runtime_env={"env_vars": _runtime_env_vars(worker_id, sqs_path)},
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=str(node_id),
                    soft=False,
                ),
            )

        self.workers = []
        # Rank0 uses tile 0; clear any inherited mask before local XPU load.
        os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "FLAT"
        os.environ["ZE_AFFINITY_MASK"] = "0"
        self.local_rank0 = MLIPWorkerLocal(
            worker_id=0,
            world_size=num_workers,
            predictor_config=predict_unit_config,
        )
        master_addr, master_port = self.local_rank0.get_master_address_and_port()
        logging.info(
            "FXPU Phase6: rank0 in-process %s:%s W=%s per_node=%s hosts=%s marker=%s",
            master_addr,
            master_port,
            num_workers,
            num_workers_per_node,
            hosts[:need_hosts],
            MARKER,
        )

        worker_id = 0
        for node_idx, n_on_node in enumerate(num_workers_on_node_array):
            node_id = ray_node_ids[node_idx]
            for i in range(n_on_node):
                if node_idx == 0 and i == 0:
                    worker_id += 1
                    continue
                actor = _actor_opts(worker_id, node_id).remote(
                    worker_id,
                    num_workers,
                    predict_unit_config,
                    master_port,
                    master_addr,
                )
                self.workers.append(actor)
                worker_id += 1

        setup_futures = [w.ensure_distributed_setup.remote() for w in self.workers]
        import fairchem_xpu_parallel as fxp

        fxp.ensure_worker_patches()
        self.local_rank0._distributed_setup()
        if setup_futures:
            ray.get(setup_futures)

        pu = self.local_rank0.predict_unit
        self._dataset_to_tasks = copy.deepcopy(pu.dataset_to_tasks)
        self._validate_atoms_data_fn = pu.model.module.validate_atoms_data

        log.info(
            "FXPU Phase6 Parallel XPU ready: workers=%s per_node=%s marker=%s",
            num_workers,
            num_workers_per_node,
            MARKER,
        )

    ParallelMLIPPredictUnit.__init__ = _parallel_init  # type: ignore[method-assign]
    _PHASE6_APPLIED = True
    log.info("FXPU Phase6 multi-node launch patches applied (%s)", MARKER)
    return MARKER
