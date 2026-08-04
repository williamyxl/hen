#!/usr/bin/env python3
"""Relax one SQS .extxyz (ionic + optional cell) with UMA on the bound XPU tile.

Intended as the worker process launched by ``xpu_tile_pool.py`` (one structure
per tile). Writes::

  <out-dir>/relaxed.extxyz
  <out-dir>/cell_opt.log
  <out-dir>/result.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import LBFGS

ROOT = Path(__file__).resolve().parents[2]  # sqs_evaluation/
sys.path.insert(0, str(ROOT))

from energy_models import build_energy_model  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structure", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="xpu")
    p.add_argument("--dtype", default="float64")
    p.add_argument(
        "--uma-model",
        default="/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/uma-cache/uma-s-1p2.pt",
    )
    p.add_argument("--uma-task", default="omat")
    p.add_argument("--fmax", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--relax-cell", action="store_true", default=True)
    p.add_argument("--no-relax-cell", action="store_false", dest="relax_cell")
    p.add_argument(
        "--task-id",
        default=None,
        help="Optional id echoed into result.json (e.g. tile_00__sqs_000)",
    )
    args = p.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    atoms0 = read(args.structure, index=0)
    model = build_energy_model(
        {
            "model": "uma",
            "device": args.device,
            "dtype": args.dtype,
            "uma_model": str(args.uma_model),
            "uma_task": args.uma_task,
            "uma_workers": 1,
        }
    )
    atoms = model.prepare_atoms(atoms0)
    atoms.calc = model.build_calculator()

    t0 = time.time()
    opt_atoms = FrechetCellFilter(atoms) if args.relax_cell else atoms
    dyn = LBFGS(opt_atoms, logfile=str(out / "cell_opt.log"))
    converged = bool(dyn.run(fmax=args.fmax, steps=args.steps))
    e = float(atoms.get_potential_energy())
    wall_s = time.time() - t0

    row = {
        "task_id": args.task_id,
        "structure": str(args.structure.resolve()),
        "formula": atoms.get_chemical_formula(),
        "n_atoms": len(atoms),
        "energy_eV": e,
        "energy_model": model.name,
        "uma_task": args.uma_task,
        "fmax": args.fmax,
        "steps_max": args.steps,
        "relax_cell": bool(args.relax_cell),
        "converged": converged,
        "wall_s": wall_s,
        "cell_A": atoms.cell.array.tolist(),
        "volume_A3": float(atoms.get_volume()),
    }
    atoms.info.update(
        {
            "energy_eV": e,
            "energy_model": model.name,
            "cell_opt_converged": converged,
            "source_structure": str(args.structure.resolve()),
        }
    )
    write(out / "relaxed.extxyz", atoms)
    (out / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(
        f"OK task={args.task_id} E={e:.6f} eV converged={converged} "
        f"wall_s={wall_s:.1f} → {out / 'relaxed.extxyz'}"
    )
    if not converged:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
