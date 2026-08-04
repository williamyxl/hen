#!/usr/bin/env python3
"""Equation of state (E–V scan + multi-scheme ASE fits) for one relaxed SQS.

Worker for the 12-tile EOS pool. Isotropic volume scales from the relaxed cell;
single-point UMA energies (no ionic re-relax). All ASE EOS schemes are fit
equally (see ``eos_fit.py``). Writes::

  <out-dir>/result.json
  <out-dir>/eos.json
  <out-dir>/eos_s*.extxyz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[2]  # sqs_evaluation/
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from energy_models import build_energy_model  # noqa: E402
from eos_fit import attach_fits, fits_all_ok, format_fits_table  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structure", type=Path, required=True, help="Relaxed .extxyz")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="xpu")
    p.add_argument("--dtype", default="float64")
    p.add_argument(
        "--uma-model",
        default="/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/uma-cache/uma-s-1p2.pt",
    )
    p.add_argument("--uma-task", default="omat")
    p.add_argument(
        "--volumes-scale",
        type=float,
        nargs="+",
        default=[0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06],
    )
    p.add_argument("--task-id", default=None)
    args = p.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    scales = [float(s) for s in args.volumes_scale]

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
    calc = model.build_calculator()
    base = model.prepare_atoms(read(args.structure, index=0))
    cell0 = np.asarray(base.cell.array, dtype=float)

    t0 = time.time()
    volumes: list[float] = []
    energies: list[float] = []
    for s in scales:
        atoms = base.copy()
        atoms.set_cell(cell0 * (s ** (1.0 / 3.0)), scale_atoms=True)
        atoms = model.prepare_atoms(atoms)
        atoms.calc = calc
        e = float(atoms.get_potential_energy())
        v = float(atoms.get_volume())
        volumes.append(v)
        energies.append(e)
        atoms.info.update(
            {
                "eos_scale": s,
                "energy_eV": e,
                "energy_model": model.name,
                "task_id": args.task_id,
            }
        )
        write(out / f"eos_s{s:.3f}.extxyz", atoms)

    row = attach_fits(
        {
            "task_id": args.task_id,
            "structure": str(args.structure.resolve()),
            "formula": base.get_chemical_formula(),
            "n_atoms": len(base),
            "volumes_scale": scales,
            "volumes_A3": volumes,
            "energies_eV": energies,
            "energy_model": model.name,
            "uma_task": args.uma_task,
            "wall_s": time.time() - t0,
        }
    )
    (out / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    (out / "eos.json").write_text(json.dumps([row], indent=2), encoding="utf-8")
    print(f"OK task={args.task_id} wall_s={row['wall_s']:.1f}")
    print(format_fits_table(row["fits"]))
    if not fits_all_ok(row["fits"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
