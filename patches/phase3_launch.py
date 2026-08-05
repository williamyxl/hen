"""Phase 3: Ray launch / affinity / process-model improvements (no sqs_sampling edits).

Goals (vs Phase 2):
  - ZE_AFFINITY_MASK set before first torch.xpu use in each worker
  - Avoid driver CPU model double-load and discarded rank-0 Ray actor
  - Eager distributed setup during calculator load (cuts warmup_EF_s)
  - Optional CPU pinning; short /tmp Ray TMPDIR
  - Keep Ray (document Ray-must-stay): XCCL + FairChem GP already work under Ray;
    full mpiexec rewrite deferred unless warmup gate fails

Enable: applied whenever ``patch_fairchem_xpu_parallel`` runs (Phase 3 shim).
Disable: ``FXPU_PHASE3_LAUNCH=0``.
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
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MARKER = "fxpu_launch_eager_v1"
_PHASE3_APPLIED = False
_FxpuPhase3Worker = None


def phase3_enabled() -> bool:
    return os.environ.get("FXPU_PHASE3_LAUNCH", "1").strip().lower() not in (
        "0",
        "false",
        "off",
        "no",
    )


def _pin_worker_cpus(worker_id: int, cpus_per_worker: int | None = None) -> list[int] | None:
    """Bind process to a disjoint CPU set (best-effort)."""
    if os.environ.get("FXPU_PHASE3_CPU_PIN", "1").strip().lower() in (
        "0",
        "false",
        "off",
        "no",
    ):
        return None
    try:
        n = int(os.cpu_count() or 1)
        per = int(cpus_per_worker or os.environ.get("FXPU_CPUS_PER_WORKER", "8"))
        per = max(1, min(per, n))
        start = (int(worker_id) * per) % n
        cores = [(start + k) % n for k in range(per)]
        os.sched_setaffinity(0, cores)
        return cores
    except Exception as exc:  # noqa: BLE001
        log.debug("CPU pin skipped for worker %s: %s", worker_id, exc)
        return None


def _short_ray_tmpdir() -> Path:
    """Keep AF_UNIX paths short (PBS $TMPDIR is often too long)."""
    user = os.environ.get("USER", "hen")[:8]
    # Prefer existing PBS/harness RAY_TMPDIR if already under /tmp and short.
    existing = os.environ.get("RAY_TMPDIR") or os.environ.get("TMPDIR")
    if existing and existing.startswith("/tmp/") and len(existing) < 80:
        p = Path(existing)
        p.mkdir(parents=True, exist_ok=True)
        return p
    p = Path("/tmp") / f"r{user}-{os.getpid()}"
    p.mkdir(parents=True, exist_ok=True)
    os.environ["RAY_TMPDIR"] = str(p)
    os.environ["TMPDIR"] = str(p)
    return p


def _runtime_env_vars(worker_id: int, sqs_path: str) -> dict[str, str]:
    """Env injected at Ray worker *process start* (before imports)."""
    from patches import phase2_xccl as p2

    env: dict[str, str] = {
        "ZE_FLAT_DEVICE_HIERARCHY": "FLAT",
        "ZE_AFFINITY_MASK": str(int(worker_id)),
        "PYTHONPATH": os.environ.get("PYTHONPATH", sqs_path),
        "FXPU_PHASE3_LAUNCH": "1",
        "FXPU_EDGEWISE_AC_BISECT": os.environ.get("FXPU_EDGEWISE_AC_BISECT", "2"),
    }
    # Propagate XCCL / Phase2 knobs into actors.
    if p2.use_xccl():
        env.update(p2.ray_worker_xccl_env_vars())
        env["CCL_LOCAL_RANK"] = str(int(worker_id))
        env["LOCAL_RANK"] = str(int(worker_id))
    for key in (
        "FXPU_DIST_BACKEND",
        "FXPU_FI_PROVIDER",
        "FI_PROVIDER",
        "CCL_ATL_TRANSPORT",
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
    # Propagate all FXPU_* diagnostic/force knobs (MAX_LAYERS, DUMP_*, ONE_CHUNK, …).
    # Without this, Ray runtime_env can omit them and Phase1 probes silently no-op.
    for key, val in os.environ.items():
        if key.startswith("FXPU_") and val:
            env[key] = val
    return env


def _get_phase3_worker_actor():
    """Ray actor: affinity + CPU pin before ensure_worker_patches (torch)."""
    global _FxpuPhase3Worker
    if _FxpuPhase3Worker is not None:
        return _FxpuPhase3Worker

    import ray
    from fairchem.core.units.mlip_unit.predict import MLIPWorkerLocal

    @ray.remote
    class FxpuPhase3XPUMLIPWorker(MLIPWorkerLocal):
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            worker_id = int(args[0]) if args else int(kwargs.get("worker_id", 0))
            # Belt-and-suspenders: runtime_env already set these at process start.
            os.environ.setdefault("ZE_FLAT_DEVICE_HIERARCHY", "FLAT")
            os.environ["ZE_AFFINITY_MASK"] = str(worker_id)
            pinned = _pin_worker_cpus(worker_id)
            # Import on worker after affinity — ensure_worker_patches pulls in torch.xpu.
            import fairchem_xpu_parallel as fxp

            fxp.ensure_worker_patches()
            super().__init__(*args, **kwargs)
            logging.getLogger(__name__).info(
                "FXPU Phase3 worker %s start mask=%s pin=%s marker=%s",
                worker_id,
                os.environ.get("ZE_AFFINITY_MASK"),
                pinned,
                MARKER,
            )

        def ensure_distributed_setup(self) -> str:
            if not self.is_setup:
                self._distributed_setup()
            return MARKER

    _FxpuPhase3Worker = FxpuPhase3XPUMLIPWorker
    return _FxpuPhase3Worker


def apply_launch_patches() -> str:
    """Replace ParallelMLIPPredictUnit XPU __init__ with Phase-3 launch path."""
    global _PHASE3_APPLIED
    if not phase3_enabled():
        log.info("FXPU Phase3 launch disabled (FXPU_PHASE3_LAUNCH=0)")
        return "disabled"
    if _PHASE3_APPLIED:
        return MARKER

    import copy

    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    from fairchem.core.units.mlip_unit.predict import (
        MLIPWorkerLocal,
        ParallelMLIPPredictUnit,
    )

    from patches import phase2_xccl as p2

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
        elif int(num_workers_per_node) > int(num_workers):
            # energy.py passes max(W,12); W=2 must schedule 2 tiles, not 12.
            num_workers_per_node = int(num_workers)

        os.environ.setdefault("ZE_FLAT_DEVICE_HIERARCHY", "FLAT")
        os.environ["ZE_AFFINITY_MASK"] = "0"
        _pin_worker_cpus(0)

        # Keep shim first on PYTHONPATH for Ray workers.
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
        # Defer dataset_to_tasks / validate_atoms_data until after rank0 XPU load
        # (avoids driver CPU MLIPPredictUnit double-load).
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

        ray_tmp = _short_ray_tmpdir()
        if p2.use_xccl():
            p2.configure_xccl_env_for_ray(local_rank=0, local_size=int(num_workers))

        n_remote = max(int(num_workers) - 1, 0)
        if not ray.is_initialized():
            os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
            # W=12 on Aurora can be slow to register the raylet after several
            # prior W subprocesses; 120s was flaky in phase3 gate.
            os.environ["RAY_raylet_start_wait_time_s"] = os.environ.get(
                "RAY_raylet_start_wait_time_s", "300"
            )
            # Parallel single-node batteries: connect to a pre-started head
            # (unique FXPU_RAY_ADDRESS / RAY_ADDRESS per hypothesis).
            pre_addr = (
                os.environ.get("FXPU_RAY_ADDRESS", "").strip()
                or os.environ.get("RAY_ADDRESS", "").strip()
            )
            # If ray start logs leaked into the env, take the last host:port token.
            if pre_addr and ("\n" in pre_addr or " " in pre_addr):
                import re as _re

                toks = _re.findall(
                    r"(?:\d{1,3}(?:\.\d{1,3}){3}|localhost|[\w.-]+):\d{2,5}",
                    pre_addr,
                )
                if toks:
                    pre_addr = toks[-1]
            if pre_addr and pre_addr.lower() not in ("auto", "local"):
                log.info(
                    "FXPU Phase3: connecting to existing Ray at %s (parallel battery)",
                    pre_addr,
                )
                ray.init(
                    address=pre_addr,
                    ignore_reinit_error=True,
                    logging_level=log_level,
                )
            else:
                # Custom resource budget = remote actors only (rank0 is in-process).
                init_kwargs = dict(
                    logging_level=log_level,
                    num_cpus=max(num_workers_per_node, num_workers),
                    resources={"xpu_tile": float(max(n_remote, 1))},
                    _temp_dir=str(ray_tmp),
                    include_dashboard=False,
                )
                # Force a brand-new local cluster when requested (still one-per-process).
                if os.environ.get("FXPU_RAY_FORCE_LOCAL", "").strip().lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                ):
                    init_kwargs["address"] = "local"
                last_exc: Exception | None = None
                for attempt in range(1, 4):
                    try:
                        ray.init(**init_kwargs)
                        last_exc = None
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_exc = exc
                        log.warning(
                            "FXPU Phase3 ray.init attempt %s/3 failed: %s", attempt, exc
                        )
                        try:
                            ray.shutdown()
                        except Exception:  # noqa: BLE001
                            pass
                        # Fresh short temp dir per retry (stale GCS sockets).
                        ray_tmp = Path("/tmp") / f"r{os.environ.get('USER', 'hen')[:8]}-{os.getpid()}-a{attempt}"
                        ray_tmp.mkdir(parents=True, exist_ok=True)
                        os.environ["RAY_TMPDIR"] = str(ray_tmp)
                        os.environ["TMPDIR"] = str(ray_tmp)
                        init_kwargs["_temp_dir"] = str(ray_tmp)
                        import time as _time

                        _time.sleep(2.0 * attempt)
                if last_exc is not None:
                    raise last_exc

        self.atomic_data_on_device = None
        num_nodes = math.ceil(num_workers / num_workers_per_node)
        num_workers_on_node_array = [num_workers_per_node] * num_nodes
        if num_workers % num_workers_per_node > 0:
            num_workers_on_node_array[-1] = num_workers % num_workers_per_node

        # Placement: remote workers only (no discarded rank0 actor / no tile-0 contention).
        placement_groups = []
        remote_counts = list(num_workers_on_node_array)
        if remote_counts:
            remote_counts[0] = max(remote_counts[0] - 1, 0)
        for n_w in remote_counts:
            if n_w <= 0:
                placement_groups.append(None)
                continue
            pg = placement_group(
                [{"CPU": n_w, "xpu_tile": float(n_w)}], strategy="STRICT_PACK"
            )
            placement_groups.append(pg)
        for pg in placement_groups:
            if pg is not None:
                ray.get(pg.ready())
                break

        FxpuWorker = _get_phase3_worker_actor()

        def _actor_opts(worker_id: int, pg):  # type: ignore[no-untyped-def]
            return FxpuWorker.options(
                num_cpus=1,
                num_gpus=0,
                resources={"xpu_tile": 1.0},
                runtime_env={"env_vars": _runtime_env_vars(worker_id, sqs_path)},
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=0,
                    placement_group_capture_child_tasks=True,
                ),
            )

        self.workers = []
        self.local_rank0 = MLIPWorkerLocal(
            worker_id=0,
            world_size=num_workers,
            predictor_config=predict_unit_config,
        )
        master_addr, master_port = self.local_rank0.get_master_address_and_port()
        logging.info(
            "FXPU Phase3: rank0 in-process on %s:%s (no CPU preload, no discard actor) marker=%s",
            master_addr,
            master_port,
            MARKER,
        )

        worker_id = 0
        for pg_idx, pg in enumerate(placement_groups):
            # Original per-node count includes rank0 on node 0.
            workers_on_node = num_workers_on_node_array[pg_idx]
            for i in range(workers_on_node):
                if pg_idx == 0 and i == 0:
                    worker_id += 1
                    continue
                assert pg is not None
                actor = _actor_opts(worker_id, pg).remote(
                    worker_id,
                    num_workers,
                    predict_unit_config,
                    master_port,
                    master_addr,
                )
                self.workers.append(actor)
                worker_id += 1

        # Eager distributed setup during load_s (not deferred into warmup_EF_s).
        # Fire remotes first so init_process_group can rendezvous with rank0.
        setup_futures = [w.ensure_distributed_setup.remote() for w in self.workers]
        # Rank0: affinity already 0; patches applied on driver via patch_fairchem_xpu_parallel.
        import fairchem_xpu_parallel as fxp

        fxp.ensure_worker_patches()
        self.local_rank0._distributed_setup()
        if setup_futures:
            ray.get(setup_futures)

        pu = self.local_rank0.predict_unit
        self._dataset_to_tasks = copy.deepcopy(pu.dataset_to_tasks)
        self._validate_atoms_data_fn = pu.model.module.validate_atoms_data

        log.info(
            "FXPU Phase3 Parallel XPU unit ready: workers=%s tiles=0..%s marker=%s",
            num_workers,
            num_workers - 1,
            MARKER,
        )

    ParallelMLIPPredictUnit.__init__ = _parallel_init  # type: ignore[method-assign]
    _PHASE3_APPLIED = True
    log.info("FXPU Phase3 launch patches applied (%s)", MARKER)
    return MARKER
