#!/usr/bin/env python3
"""12-tile Aurora XPU resource manager + monitor for async cell_opt tasks.

One process owns the node. It maintains a free-list of 12 GPU tiles (with the
ALCF Example-4 CPU / NUMA bindings), pulls structures from a queue, and launches
``relax_one.py`` on the next free tile. When a worker exits, its tile is
returned to the pool and the next queued structure starts immediately.

Typical use (from a PBS job on one Aurora node)::

  python xpu_tile_pool.py \\
    --sqs-run .../sqs_sampling/runs/mc_sqs_20260730_032035 \\
    --out-dir  .../sqs_evaluation/workflow_out/cell_opt_<stamp>

Does **not** submit PBS itself. Safe to ``--dry-run`` for a launch plan.

Monitoring
----------
* Live one-line progress on stdout (every ``--poll`` seconds).
* Atomic ``status.json`` in ``--out-dir`` (queue / running / done / failed).
* Per-task dirs: ``tasks/<task_id>/{worker.log,cell_opt.log,result.json,relaxed.extxyz}``.
* Resume: tasks with ``result.json`` + ``relaxed.extxyz`` are skipped unless
  ``--force``.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Aurora single-node tile map (FLAT hierarchy). CPU lists from ALCF Example 4;
# cores 0 and 52 reserved for the OS.
TILE_BINDINGS: list[dict[str, Any]] = [
    {"tile": 0, "cpus": "1-8", "membind": 0},
    {"tile": 1, "cpus": "9-16", "membind": 0},
    {"tile": 2, "cpus": "17-24", "membind": 0},
    {"tile": 3, "cpus": "25-32", "membind": 0},
    {"tile": 4, "cpus": "33-40", "membind": 0},
    {"tile": 5, "cpus": "41-48", "membind": 0},
    {"tile": 6, "cpus": "53-60", "membind": 1},
    {"tile": 7, "cpus": "61-68", "membind": 1},
    {"tile": 8, "cpus": "69-76", "membind": 1},
    {"tile": 9, "cpus": "77-84", "membind": 1},
    {"tile": 10, "cpus": "85-92", "membind": 1},
    {"tile": 11, "cpus": "93-100", "membind": 1},
]

HERE = Path(__file__).resolve().parent
RELAX_ONE = HERE / "relax_one.py"


@dataclass
class Task:
    task_id: str
    structure: str
    out_dir: str
    state: str = "queued"  # queued | running | done | failed | skipped
    tile: int | None = None
    pid: int | None = None
    returncode: int | None = None
    t_submit: float | None = None
    t_start: float | None = None
    t_end: float | None = None
    error: str | None = None


@dataclass
class Slot:
    tile: int
    cpus: str
    membind: int
    proc: subprocess.Popen | None = None
    task_id: str | None = None


@dataclass
class PoolState:
    created_utc: str
    out_dir: str
    sqs_run: str
    n_tiles: int
    n_tasks: int
    n_queued: int = 0
    n_running: int = 0
    n_done: int = 0
    n_failed: int = 0
    n_skipped: int = 0
    free_tiles: list[int] = field(default_factory=list)
    running: list[dict[str, Any]] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    updated_utc: str = ""
    wall_s: float = 0.0
    finished: bool = False


def discover_structures(sqs_run: Path) -> list[Path]:
    """All ``tile_XX/sqs_YYY.extxyz`` under an mc_sqs run directory."""
    files = sorted(sqs_run.glob("tile_*/sqs_*.extxyz"))
    files = [p for p in files if p.name.startswith("sqs_") and p.suffix == ".extxyz"]
    if not files:
        raise FileNotFoundError(f"No tile_*/sqs_*.extxyz under {sqs_run}")
    return files


def task_id_for(path: Path, sqs_run: Path) -> str:
    rel = path.relative_to(sqs_run)
    # tile_00/sqs_000.extxyz → tile_00__sqs_000
    return f"{rel.parent.name}__{path.stem}"


def is_complete(task_dir: Path) -> bool:
    return (task_dir / "result.json").is_file() and (task_dir / "relaxed.extxyz").is_file()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def build_worker_cmd(
    *,
    python: str,
    structure: Path,
    out_dir: Path,
    task_id: str,
    fmax: float,
    steps: int,
    relax_cell: bool,
    uma_model: str,
    uma_task: str,
    dtype: str,
) -> list[str]:
    cmd = [
        python,
        str(RELAX_ONE),
        "--structure",
        str(structure),
        "--out-dir",
        str(out_dir),
        "--task-id",
        task_id,
        "--device",
        "xpu",
        "--dtype",
        dtype,
        "--uma-model",
        uma_model,
        "--uma-task",
        uma_task,
        "--fmax",
        str(fmax),
        "--steps",
        str(steps),
    ]
    if relax_cell:
        cmd.append("--relax-cell")
    else:
        cmd.append("--no-relax-cell")
    return cmd


def launch_on_tile(
    slot: Slot,
    task: Task,
    *,
    python: str,
    fmax: float,
    steps: int,
    relax_cell: bool,
    uma_model: str,
    uma_task: str,
    dtype: str,
    dry_run: bool,
) -> None:
    out = Path(task.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    worker_log = out / "worker.log"
    cmd = build_worker_cmd(
        python=python,
        structure=Path(task.structure),
        out_dir=out,
        task_id=task.task_id,
        fmax=fmax,
        steps=steps,
        relax_cell=relax_cell,
        uma_model=uma_model,
        uma_task=uma_task,
        dtype=dtype,
    )
    # numactl binds CPU/memory; ZE_AFFINITY_MASK selects the XPU tile.
    wrapped = [
        "numactl",
        f"--physcpubind={slot.cpus}",
        f"--membind={slot.membind}",
        *cmd,
    ]
    env = os.environ.copy()
    env["ZE_AFFINITY_MASK"] = str(slot.tile)
    env["ZE_FLAT_DEVICE_HIERARCHY"] = "FLAT"
    env["ZE_ENABLE_PCI_ID_DEVICE_ORDER"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    meta = {
        "task_id": task.task_id,
        "tile": slot.tile,
        "cpus": slot.cpus,
        "membind": slot.membind,
        "cmd": wrapped,
        "env_ZE_AFFINITY_MASK": env["ZE_AFFINITY_MASK"],
        "launched_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "launch.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if dry_run:
        task.state = "skipped"
        task.error = "dry_run"
        task.t_end = time.time()
        print(f"[dry-run] tile={slot.tile:02d} {task.task_id}")
        print("          " + " ".join(wrapped))
        return

    log_f = open(worker_log, "w", encoding="utf-8")  # noqa: SIM115 — kept for Popen lifetime
    log_f.write(" ".join(wrapped) + "\n")
    log_f.flush()
    proc = subprocess.Popen(
        wrapped,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(HERE),
        start_new_session=True,  # detach from manager signals except explicit kill
    )
    slot.proc = proc
    slot.task_id = task.task_id
    task.state = "running"
    task.tile = slot.tile
    task.pid = proc.pid
    task.t_start = time.time()
    # stash handle so we can close log on reap
    proc._hen_log_f = log_f  # type: ignore[attr-defined]


def reap_slot(slot: Slot, tasks: dict[str, Task]) -> bool:
    """Return True if a running worker finished (tile now free)."""
    if slot.proc is None or slot.task_id is None:
        return False
    rc = slot.proc.poll()
    if rc is None:
        return False
    task = tasks[slot.task_id]
    task.returncode = rc
    task.t_end = time.time()
    log_f = getattr(slot.proc, "_hen_log_f", None)
    if log_f is not None:
        try:
            log_f.close()
        except OSError:
            pass
    if rc == 0 and is_complete(Path(task.out_dir)):
        task.state = "done"
        task.error = None
    else:
        task.state = "failed"
        task.error = f"returncode={rc}"
    slot.proc = None
    slot.task_id = None
    return True


def summarize(tasks: dict[str, Task], slots: list[Slot], t0: float, **extra: Any) -> dict[str, Any]:
    counts = {"queued": 0, "running": 0, "done": 0, "failed": 0, "skipped": 0}
    for t in tasks.values():
        counts[t.state] = counts.get(t.state, 0) + 1
    running = [
        {
            "task_id": s.task_id,
            "tile": s.tile,
            "pid": tasks[s.task_id].pid if s.task_id else None,
            "elapsed_s": (
                time.time() - tasks[s.task_id].t_start
                if s.task_id and tasks[s.task_id].t_start
                else None
            ),
        }
        for s in slots
        if s.proc is not None and s.task_id is not None
    ]
    free = [s.tile for s in slots if s.proc is None]
    st = PoolState(
        created_utc=extra.get("created_utc", ""),
        out_dir=extra["out_dir"],
        sqs_run=extra["sqs_run"],
        n_tiles=len(slots),
        n_tasks=len(tasks),
        n_queued=counts["queued"],
        n_running=counts["running"],
        n_done=counts["done"],
        n_failed=counts["failed"],
        n_skipped=counts["skipped"],
        free_tiles=free,
        running=running,
        failed=[t.task_id for t in tasks.values() if t.state == "failed"],
        updated_utc=datetime.now(timezone.utc).isoformat(),
        wall_s=time.time() - t0,
        finished=extra.get("finished", False),
    )
    payload = asdict(st)
    payload["tasks"] = {tid: asdict(t) for tid, t in tasks.items()}
    return payload


def progress_line(payload: dict[str, Any]) -> str:
    run = ",".join(
        f"t{r['tile']:02d}:{r['task_id']}" for r in payload.get("running", [])
    ) or "-"
    return (
        f"[{payload['wall_s']:7.0f}s] "
        f"done={payload['n_done']}/{payload['n_tasks']} "
        f"run={payload['n_running']} fail={payload['n_failed']} "
        f"queue={payload['n_queued']} free={payload['free_tiles']} "
        f"active=[{run}]"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Async 12-tile XPU pool manager for cell_opt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--sqs-run",
        type=Path,
        required=True,
        help="mc_sqs run dir containing tile_XX/sqs_YYY.extxyz",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--n-tiles",
        type=int,
        default=12,
        help="Number of XPU tiles to use (1–12)",
    )
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--fmax", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--relax-cell", action="store_true", default=True)
    ap.add_argument("--no-relax-cell", action="store_false", dest="relax_cell")
    ap.add_argument(
        "--uma-model",
        default="/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/uma-cache/uma-s-1p2.pt",
    )
    ap.add_argument("--uma-task", default="omat")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--poll", type=float, default=5.0, help="Monitor poll interval (s)")
    ap.add_argument("--force", action="store_true", help="Re-run tasks that already completed")
    ap.add_argument("--dry-run", action="store_true", help="Print launch plan; do not exec")
    ap.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional cap (smoke / partial runs)",
    )
    args = ap.parse_args()

    if not (1 <= args.n_tiles <= len(TILE_BINDINGS)):
        raise SystemExit(f"--n-tiles must be in 1..{len(TILE_BINDINGS)}")
    if not RELAX_ONE.is_file():
        raise SystemExit(f"missing worker script: {RELAX_ONE}")

    sqs_run = args.sqs_run.resolve()
    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    status_path = out_root / "status.json"
    created_utc = datetime.now(timezone.utc).isoformat()

    structures = discover_structures(sqs_run)
    if args.max_tasks is not None:
        structures = structures[: args.max_tasks]

    tasks: dict[str, Task] = {}
    queue: list[str] = []
    for path in structures:
        tid = task_id_for(path, sqs_run)
        tdir = out_root / "tasks" / tid
        task = Task(
            task_id=tid,
            structure=str(path.resolve()),
            out_dir=str(tdir),
            t_submit=time.time(),
        )
        if not args.force and is_complete(tdir):
            task.state = "skipped"
            task.t_end = time.time()
        else:
            task.state = "queued"
            queue.append(tid)
        tasks[tid] = task

    slots = [
        Slot(tile=b["tile"], cpus=b["cpus"], membind=b["membind"])
        for b in TILE_BINDINGS[: args.n_tiles]
    ]

    stop = {"flag": False}

    def _on_signal(signum: int, _frame: Any) -> None:
        print(f"\nreceived signal {signum}; stopping new launches, waiting for workers…")
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    t0 = time.time()
    print(
        f"xpu_tile_pool: {len(tasks)} tasks, {len(queue)} to run, "
        f"{args.n_tiles} tiles, out={out_root}"
    )
    if args.dry_run:
        # Launch plan only (uses first N free tiles conceptually).
        for i, tid in enumerate(queue):
            slot = slots[i % len(slots)]
            launch_on_tile(
                slot,
                tasks[tid],
                python=args.python,
                fmax=args.fmax,
                steps=args.steps,
                relax_cell=args.relax_cell,
                uma_model=args.uma_model,
                uma_task=args.uma_task,
                dtype=args.dtype,
                dry_run=True,
            )
        payload = summarize(
            tasks,
            slots,
            t0,
            created_utc=created_utc,
            out_dir=str(out_root),
            sqs_run=str(sqs_run),
            finished=True,
        )
        atomic_write_json(status_path, payload)
        print(f"dry-run wrote {status_path}")
        return

    last_print = 0.0
    while True:
        # Reap finished workers.
        for slot in slots:
            if reap_slot(slot, tasks):
                pass

        # Fill free tiles from the queue.
        if not stop["flag"]:
            for slot in slots:
                if slot.proc is not None:
                    continue
                if not queue:
                    break
                tid = queue.pop(0)
                launch_on_tile(
                    slot,
                    tasks[tid],
                    python=args.python,
                    fmax=args.fmax,
                    steps=args.steps,
                    relax_cell=args.relax_cell,
                    uma_model=args.uma_model,
                    uma_task=args.uma_task,
                    dtype=args.dtype,
                    dry_run=False,
                )

        running = any(s.proc is not None for s in slots)
        finished = (not queue) and (not running)

        payload = summarize(
            tasks,
            slots,
            t0,
            created_utc=created_utc,
            out_dir=str(out_root),
            sqs_run=str(sqs_run),
            finished=finished,
        )
        atomic_write_json(status_path, payload)

        now = time.time()
        if now - last_print >= args.poll or finished:
            print(progress_line(payload), flush=True)
            last_print = now

        if finished:
            break
        time.sleep(min(1.0, args.poll))

    # Final task ledger
    ledger = out_root / "tasks_ledger.json"
    atomic_write_json(
        ledger,
        {
            "created_utc": created_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "n_done": payload["n_done"],
            "n_failed": payload["n_failed"],
            "n_skipped": payload["n_skipped"],
            "failed": payload["failed"],
            "tasks": payload["tasks"],
        },
    )

    # Convenience: concatenate all relaxed frames if any succeeded.
    try:
        from ase.io import read as ase_read
        from ase.io import write as ase_write

        frames = []
        for tid, t in sorted(tasks.items()):
            if t.state in ("done", "skipped") and is_complete(Path(t.out_dir)):
                frames.append(ase_read(Path(t.out_dir) / "relaxed.extxyz"))
        if frames:
            ase_write(out_root / "all_relaxed.extxyz", frames)
            print(f"wrote {out_root / 'all_relaxed.extxyz'} ({len(frames)} frames)")
    except Exception as exc:  # noqa: BLE001 — optional convenience
        print(f"WARN: could not write all_relaxed.extxyz: {exc}")

    print(
        f"DONE done={payload['n_done']} skipped={payload['n_skipped']} "
        f"failed={payload['n_failed']} wall_s={payload['wall_s']:.1f}"
    )
    if payload["n_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
