#!/usr/bin/env python3
"""1-node multi-tile UMA FP64 energy + AG forces for any ASE Atoms.

Public API::

    from fxpu_1node_ef_atoms import predict_ef, predict_ef_timed
    energy_eV, forces = predict_ef(atoms, workers=12)
    result = predict_ef_timed(atoms, workers=12, repeats=3)

``__main__``:
  - single W via ``FXPU_WORKERS`` (default 12), or
  - ladder via ``FXPU_LADDER=1,2,4,6,12`` (subprocess per W for clean Ray teardown).

Demo geometry when run as main: NaCl rocksalt N³ (default N=18).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = PROJECT_ROOT / "uma-cache" / "uma-s-1p2.pt"


def _refuse_arch_change() -> None:
    bad = [
        k
        for k in ("FXPU_EDGEWISE_MAX_LAYERS", "FXPU_DETACH_AFTER_MP")
        if os.environ.get(k, "").strip()
    ]
    if bad:
        raise SystemExit(f"architecture change env set ({bad}) — refuse. Full UMA only.")


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _bootstrap() -> None:
    """Shim first so FairChem XPU patches load before sqs_sampling energy."""
    pref = [
        PROJECT_ROOT / "shim",
        PROJECT_ROOT,
        PROJECT_ROOT / "sqs_sampling",
        PROJECT_ROOT / "scripts",
    ]
    for p in reversed(pref):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)
    os.environ.setdefault("ZE_FLAT_DEVICE_HIERARCHY", "FLAT")
    import fairchem_xpu_parallel  # noqa: F401


def build_nacl_rocksalt(
    n: int,
    *,
    a: float = 5.64,
    rattle: float = 0.05,
    seed: int = 0,
):
    """Conventional NaCl rocksalt n×n×n supercell (ASE Atoms)."""
    from ase import Atoms

    na_frac = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    )
    cl_frac = na_frac + 0.5
    symbols: list[str] = []
    scaled: list[list[float]] = []
    for ix in range(n):
        for iy in range(n):
            for iz in range(n):
                off = np.array([ix, iy, iz], dtype=np.float64)
                for row in na_frac:
                    symbols.append("Na")
                    scaled.append(((row + off) / n).tolist())
                for row in cl_frac:
                    symbols.append("Cl")
                    scaled.append(((row + off) / n).tolist())
    cell = np.eye(3) * (a * n)
    atoms = Atoms(symbols=symbols, cell=cell, pbc=True)
    atoms.set_scaled_positions(np.asarray(scaled, dtype=np.float64))
    if rattle and float(rattle) > 0.0:
        rng = np.random.default_rng(int(seed))
        atoms.positions = atoms.positions + float(rattle) * rng.standard_normal(
            atoms.positions.shape
        )
    return atoms


def _make_calc(ckpt: Path, workers: int, task: str, dtype: str):
    from energy import uma_predict_unit
    from fairchem.core import FAIRChemCalculator

    return FAIRChemCalculator(
        uma_predict_unit(
            model=str(ckpt),
            device="xpu",
            dtype=dtype,
            workers=workers,
        ),
        task_name=task,
    )


def predict_ef(
    atoms,
    *,
    workers: int = 12,
    model: str | Path | None = None,
    task: str = "omat",
    dtype: str = "float64",
    apply_phase1: bool = True,
) -> tuple[float, np.ndarray]:
    """UMA FP64 energy + AG forces on ``atoms`` (1-node, W=1…12)."""
    result = predict_ef_timed(
        atoms,
        workers=workers,
        model=model,
        task=task,
        dtype=dtype,
        apply_phase1=apply_phase1,
        repeats=0,
        warmup=False,
    )
    return float(result["energy_eV"]), np.asarray(result["forces"], dtype=np.float64)


def predict_ef_timed(
    atoms,
    *,
    workers: int = 12,
    model: str | Path | None = None,
    task: str = "omat",
    dtype: str = "float64",
    apply_phase1: bool = True,
    repeats: int = 3,
    warmup: bool = True,
) -> dict[str, Any]:
    """Energy + AG forces with optional warmup and timed repeats.

    Returns dict with energy_eV, forces, load_s, warmup_s, ef_mean_s, ef_times_s, wall_s.
    ``repeats=0`` skips timed loop (single E+F after optional warmup).
    """
    _refuse_arch_change()
    _bootstrap()

    ckpt = Path(model) if model is not None else Path(
        os.environ.get("FXPU_UMA_CKPT", str(DEFAULT_CKPT))
    )
    if ckpt.name != "uma-s-1p2.pt":
        raise SystemExit(f"Only uma-s-1p2.pt allowed, got {ckpt.name}")
    if not ckpt.is_file():
        raise SystemExit(f"Missing checkpoint: {ckpt}")

    workers = int(workers)
    if workers < 1 or workers > 12:
        raise SystemExit(f"1-node launcher requires 1<=workers<=12, got {workers}")

    # W=1: energy.uma_predict_unit requires a single visible tile.
    # W>1: parent must leave ZE_AFFINITY_MASK unset so Ray workers pin tiles.
    if workers == 1:
        os.environ["ZE_AFFINITY_MASK"] = "0"
    else:
        os.environ.pop("ZE_AFFINITY_MASK", None)

    if apply_phase1:
        from patches.phase1_force_correctness import apply_all_phase1_patches

        apply_all_phase1_patches()

    from ase import Atoms

    work = atoms.copy() if hasattr(atoms, "copy") else Atoms(atoms)
    work.info.setdefault("charge", 0)
    work.info.setdefault("spin", 0)
    ref_positions = work.get_positions().copy()

    def _sync_xpu() -> None:
        try:
            import torch

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.synchronize()
        except Exception:
            pass

    t_wall0 = time.perf_counter()
    t_load0 = time.perf_counter()
    calc = _make_calc(ckpt, workers, task, dtype)
    work.calc = calc
    load_s = time.perf_counter() - t_load0

    warmup_s = 0.0
    if warmup:
        work.set_positions(ref_positions)
        t0 = time.perf_counter()
        _ = float(work.get_potential_energy())
        _ = np.asarray(work.get_forces(), dtype=np.float64)
        _sync_xpu()
        warmup_s = time.perf_counter() - t0

    ef_times: list[float] = []
    energy = float("nan")
    forces = np.zeros((len(work), 3), dtype=np.float64)
    n_timed = max(0, int(repeats))
    for i in range(n_timed):
        # Bust ASE/FairChem result cache (same pattern as opt_scale_nacl_20).
        work.set_positions(ref_positions)
        work.positions[0, 0] += 1e-6 * ((-1) ** i)
        work.calc = calc
        t0 = time.perf_counter()
        _ = float(work.get_potential_energy())
        _ = np.asarray(work.get_forces(), dtype=np.float64)
        _sync_xpu()
        ef_times.append(time.perf_counter() - t0)

    # Parity snapshot on exact reference geometry
    work.set_positions(ref_positions)
    work.calc = calc
    t0 = time.perf_counter()
    energy = float(work.get_potential_energy())
    forces = np.asarray(work.get_forces(), dtype=np.float64)
    _sync_xpu()
    if n_timed == 0 and not warmup:
        ef_times.append(time.perf_counter() - t0)

    wall_s = time.perf_counter() - t_wall0
    ef_mean_s = float(np.mean(ef_times)) if ef_times else float("nan")
    return {
        "energy_eV": energy,
        "forces": forces,
        "workers": workers,
        "natoms": int(len(work)),
        "load_s": load_s,
        "warmup_s": warmup_s,
        "ef_times_s": ef_times,
        "ef_mean_s": ef_mean_s,
        "wall_s": wall_s,
        "repeats": int(repeats),
        "warmup": bool(warmup),
    }


def _parity_vs_baseline(
    energy: float,
    forces: np.ndarray,
    base_energy: float,
    base_forces: np.ndarray,
) -> dict[str, float]:
    n = forces.shape[0]
    dE = float(energy - base_energy)
    dF = forces - base_forces
    a = base_forces.ravel()
    b = forces.ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))
    return {
        "delta_E_eV": dE,
        "delta_E_meV_per_atom": 1e3 * dE / n,
        "max_abs_dF": float(np.abs(dF).max()),
        "rms_dF": float(np.sqrt((dF**2).mean())),
        "cos": cos,
    }


def _write_worker_result(out: Path, workers: int, atoms, result: dict[str, Any]) -> dict[str, Any]:
    forces = np.asarray(result["forces"], dtype=np.float64)
    forces_path = out / f"forces_w{workers:02d}.npy"
    np.save(forces_path, forces)
    energy = float(result["energy_eV"])
    summary: dict[str, Any] = {
        "probe": "fxpu_1node_ef_atoms",
        "nacl_n": int(os.environ.get("FXPU_NACL_N", "18")),
        "natoms": int(result["natoms"]),
        "workers": workers,
        "rattle": float(os.environ.get("FXPU_RATTLE", "0.05")),
        "seed": int(os.environ.get("FXPU_SEED", "0")),
        "energy_eV": energy,
        "energy_per_atom_eV": energy / len(atoms),
        "Fmax": float(np.abs(forces).max()),
        "Frms": float(np.sqrt((forces**2).mean())),
        "load_s": float(result["load_s"]),
        "warmup_s": float(result["warmup_s"]),
        "ef_mean_s": float(result["ef_mean_s"]),
        "ef_times_s": list(result["ef_times_s"]),
        "wall_s": float(result["wall_s"]),
        "repeats": int(result["repeats"]),
        "forces_npy": str(forces_path),
        "backend": os.environ.get("FXPU_DIST_BACKEND", ""),
        "gather": os.environ.get("FXPU_XCCL_UNEVEN_GATHER", ""),
        "ipc": os.environ.get("CCL_ZE_IPC_EXCHANGE", ""),
    }
    if workers == 1:
        (out / "summary_w01.json").write_text(json.dumps(summary, indent=2) + "\n")
        np.save(out / "forces_w01.npy", forces)
    elif (out / "forces_w01.npy").is_file() and (out / "summary_w01.json").is_file():
        e1 = float(json.loads((out / "summary_w01.json").read_text())["energy_eV"])
        f1 = np.load(out / "forces_w01.npy")
        summary["vs_w01"] = _parity_vs_baseline(energy, forces, e1, f1)
    (out / f"summary_w{workers:02d}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _run_one_worker(workers: int) -> int:
    n = int(os.environ.get("FXPU_NACL_N", "18"))
    rattle = float(os.environ.get("FXPU_RATTLE", "0.05"))
    seed = int(os.environ.get("FXPU_SEED", "0"))
    repeats = int(os.environ.get("FXPU_REPEATS", "3"))
    out = Path(
        os.environ.get(
            "FXPU_OUT",
            str(PROJECT_ROOT / "pbs" / "out" / f"1node_ef_atoms_n{n}"),
        )
    )
    out.mkdir(parents=True, exist_ok=True)

    atoms = build_nacl_rocksalt(n, rattle=rattle, seed=seed)
    print(
        f"NaCl {n}³  natoms={len(atoms)}  workers={workers}  "
        f"rattle={rattle} seed={seed} repeats={repeats}",
        flush=True,
    )
    result = predict_ef_timed(atoms, workers=workers, repeats=repeats, warmup=True)
    summary = _write_worker_result(out, workers, atoms, result)

    vs = summary.get("vs_w01")
    print(
        f"W={workers} E={summary['energy_eV']:.12f}  "
        f"Fmax={summary['Fmax']:.6g}  "
        f"load={summary['load_s']:.2f}s warmup={summary['warmup_s']:.2f}s "
        f"ef_mean={summary['ef_mean_s']:.3f}s wall={summary['wall_s']:.2f}s",
        flush=True,
    )
    if vs:
        print(
            f"  vs W=1: ΔE={vs['delta_E_meV_per_atom']:.3e} meV/atom  "
            f"max|ΔF|={vs['max_abs_dF']:.3e}  cos={vs['cos']:.10f}",
            flush=True,
        )
    return 0


def _finalize_ladder(out: Path) -> None:
    rows = []
    for p in sorted(out.glob("summary_w*.json")):
        rows.append(json.loads(p.read_text()))
    if not rows:
        return
    rows.sort(key=lambda r: int(r["workers"]))
    e1 = next((r for r in rows if r["workers"] == 1), None)
    ladder = {
        "probe": "fxpu_1node_ef_atoms_ladder",
        "out_dir": str(out),
        "rows": rows,
    }
    if e1 is not None:
        for r in rows:
            if "vs_w01" not in r and r["workers"] != 1:
                fpath = Path(r["forces_npy"])
                if fpath.is_file() and (out / "forces_w01.npy").is_file():
                    r["vs_w01"] = _parity_vs_baseline(
                        float(r["energy_eV"]),
                        np.load(fpath),
                        float(e1["energy_eV"]),
                        np.load(out / "forces_w01.npy"),
                    )
    (out / "ladder_summary.json").write_text(json.dumps(ladder, indent=2) + "\n")
    # Markdown table
    lines = [
        f"# 1-node E+F ladder — NaCl {rows[0].get('nacl_n', '?')}³",
        "",
        "| W | E (eV) | Fmax | load_s | warmup_s | ef_mean_s | wall_s | ΔE meV/atom | max\\|ΔF\\| | cos |",
        "|--:|-------:|-----:|-------:|---------:|----------:|-------:|------------:|----------:|----:|",
    ]
    for r in rows:
        vs = r.get("vs_w01") or {}
        lines.append(
            f"| {r['workers']} | {r['energy_eV']:.8f} | {r['Fmax']:.6f} | "
            f"{r['load_s']:.2f} | {r['warmup_s']:.2f} | {r['ef_mean_s']:.3f} | "
            f"{r['wall_s']:.2f} | "
            f"{vs.get('delta_E_meV_per_atom', float('nan')) if vs else float('nan'):.3e} | "
            f"{vs.get('max_abs_dF', float('nan')) if vs else float('nan'):.3e} | "
            f"{vs.get('cos', float('nan')) if vs else float('nan'):.8f} |"
        )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'ladder_summary.json'}  {out / 'REPORT.md'}", flush=True)


def main() -> int:
    _refuse_arch_change()
    ladder = os.environ.get("FXPU_LADDER", "").strip()
    out = Path(
        os.environ.get(
            "FXPU_OUT",
            str(
                PROJECT_ROOT
                / "pbs"
                / "out"
                / f"1node_ef_atoms_n{os.environ.get('FXPU_NACL_N', '18')}"
            ),
        )
    )

    if ladder:
        workers_list = [int(x) for x in ladder.replace(" ", "").split(",") if x]
        # Climb in order; W=1 first for baseline
        workers_list = sorted(set(workers_list))
        env_base = os.environ.copy()
        env_base["FXPU_OUT"] = str(out)
        env_base.pop("FXPU_LADDER", None)
        out.mkdir(parents=True, exist_ok=True)
        for w in workers_list:
            print(f"\n===== ladder workers={w} =====", flush=True)
            env = env_base.copy()
            env["FXPU_WORKERS"] = str(w)
            env["FXPU_ONE_SHOT"] = "1"
            rc = subprocess.run(
                [sys.executable, "-u", str(Path(__file__).resolve())],
                env=env,
                cwd=str(PROJECT_ROOT),
            ).returncode
            if rc != 0:
                return rc
        _finalize_ladder(out)
        return 0

    workers = int(os.environ.get("FXPU_WORKERS", os.environ.get("WORKERS", "12")))
    return _run_one_worker(workers)


if __name__ == "__main__":
    raise SystemExit(main())
