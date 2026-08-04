#!/usr/bin/env python3
"""Finite-strain cubic elastic constants for one relaxed SQS structure (UMA/XPU).

Worker for smoke tests and (later) the 12-tile pool. Writes::

  <out-dir>/result.json
  <out-dir>/elastic.json
  <out-dir>/*_uni_*.extxyz / *_shear_*.extxyz
  <out-dir>/worker summary on stdout
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import LBFGS

ROOT = Path(__file__).resolve().parents[2]  # sqs_evaluation/
sys.path.insert(0, str(ROOT))

from energy_models import build_energy_model  # noqa: E402

EV_A3_TO_GPA = 160.21766208


def apply_strain(atoms, eps_matrix):
    a = atoms.copy()
    a.set_cell(atoms.cell.array @ (np.eye(3) + eps_matrix), scale_atoms=True)
    return a


def vrh_cubic(c11: float, c12: float, c44: float) -> dict[str, float]:
    bv = (c11 + 2.0 * c12) / 3.0
    gv = (c11 - c12 + 3.0 * c44) / 5.0
    gr = 5.0 * (c11 - c12) * c44 / (4.0 * c44 + 3.0 * (c11 - c12))
    b, g = bv, 0.5 * (gv + gr)
    e = 9.0 * b * g / (3.0 * b + g)
    nu = (3.0 * b - 2.0 * g) / (2.0 * (3.0 * b + g))
    return {"B": b, "G": g, "E": e, "nu": nu}


def ionic_relax(atoms, model, calc, *, fmax: float, steps: int, logfile):
    atoms = model.prepare_atoms(atoms)
    atoms.calc = calc
    LBFGS(atoms, logfile=logfile).run(fmax=fmax, steps=steps)
    # ASE stress is σ = (1/V) ∂E/∂ε (tension > 0). Do NOT negate — a leading
    # minus flips Cij and yields unphysical negative moduli.
    stress = np.asarray(atoms.get_stress(voigt=False), dtype=float) * EV_A3_TO_GPA
    return atoms, stress


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
    p.add_argument("--fmax", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument(
        "--strains",
        type=float,
        nargs="+",
        default=[-0.01, -0.005, 0.005, 0.01],
    )
    p.add_argument("--task-id", default=None)
    args = p.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    strains = [float(s) for s in args.strains]

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
    base = read(args.structure, index=0)
    base = model.prepare_atoms(base)

    t0 = time.time()
    s_uni, e_uni = [], []
    for delta in strains:
        eps = np.zeros((3, 3))
        eps[0, 0] = delta
        atoms, stress = ionic_relax(
            apply_strain(base, eps),
            model,
            calc,
            fmax=args.fmax,
            steps=args.steps,
            logfile=str(out / f"uni_{delta:+.4f}.log"),
        )
        write(out / f"uni_{delta:+.4f}.extxyz", atoms)
        s_uni.append(stress)
        e_uni.append(delta)
    c11 = float(np.polyfit(e_uni, [s[0, 0] for s in s_uni], 1)[0])
    c12 = float(np.polyfit(e_uni, [s[1, 1] for s in s_uni], 1)[0])

    s_sh, e_sh = [], []
    for gamma in strains:
        eps = np.zeros((3, 3))
        eps[1, 2] = eps[2, 1] = 0.5 * gamma
        atoms, stress = ionic_relax(
            apply_strain(base, eps),
            model,
            calc,
            fmax=args.fmax,
            steps=args.steps,
            logfile=str(out / f"shear_{gamma:+.4f}.log"),
        )
        write(out / f"shear_{gamma:+.4f}.extxyz", atoms)
        s_sh.append(stress)
        e_sh.append(gamma)
    c44 = float(np.polyfit(e_sh, [s[1, 2] for s in s_sh], 1)[0])
    der = vrh_cubic(c11, c12, c44)
    wall_s = time.time() - t0

    row = {
        "task_id": args.task_id,
        "structure": str(args.structure.resolve()),
        "formula": base.get_chemical_formula(),
        "n_atoms": len(base),
        "C11_GPa": c11,
        "C12_GPa": c12,
        "C44_GPa": c44,
        "B_GPa": der["B"],
        "G_GPa": der["G"],
        "E_GPa": der["E"],
        "nu": der["nu"],
        "strains": strains,
        "fmax": args.fmax,
        "steps": args.steps,
        "energy_model": model.name,
        "uma_task": args.uma_task,
        "wall_s": wall_s,
    }
    (out / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    (out / "elastic.json").write_text(json.dumps([row], indent=2), encoding="utf-8")
    print(
        f"OK task={args.task_id} C11={c11:.2f} C12={c12:.2f} C44={c44:.2f} GPa  "
        f"B={der['B']:.2f} G={der['G']:.2f} E={der['E']:.2f} nu={der['nu']:.4f}  "
        f"wall_s={wall_s:.1f}"
    )


if __name__ == "__main__":
    main()
