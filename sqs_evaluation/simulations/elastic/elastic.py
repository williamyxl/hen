#!/usr/bin/env python3
"""ASE + MLIP finite-strain elastic constants for relaxed cubic SQS .extxyz."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write
from ase.optimize import LBFGS

ROOT = Path(__file__).resolve().parents[3]  # hen/
sys.path.insert(0, str(ROOT / "sqs_sampling"))
from energy import (  # noqa: E402
    UMA_DEFAULTS,
    configure_gfn2_threads,
    gfn2_tblite_params,
    uma_predict_unit,
)

EVAL_ENERGY = ("gfn2-xtb", "uma")

EV_A3_TO_GPA = 160.21766208


def apply_strain(atoms, eps_matrix):
    a = atoms.copy()
    a.set_cell(atoms.cell.array @ (np.eye(3) + eps_matrix), scale_atoms=True)
    return a


def make_calc(args: argparse.Namespace):
    if args.energy == "gfn2-xtb":
        from tblite.ase import TBLite

        configure_gfn2_threads()
        return TBLite(**gfn2_tblite_params())
    if args.energy == "uma":
        from fairchem.core import FAIRChemCalculator

        return FAIRChemCalculator(
            uma_predict_unit(
                model=args.uma_model or UMA_DEFAULTS["model"],
                device=args.device,
                dtype=args.dtype,
                workers=1,
            ),
            task_name=args.uma_task,
        )
    raise ValueError(f"Unknown energy method {args.energy!r}; choose from {EVAL_ENERGY}")


def vrh_cubic(c11, c12, c44):
    bv = (c11 + 2.0 * c12) / 3.0
    gv = (c11 - c12 + 3.0 * c44) / 5.0
    gr = 5.0 * (c11 - c12) * c44 / (4.0 * c44 + 3.0 * (c11 - c12))
    b, g = bv, 0.5 * (gv + gr)
    e = 9.0 * b * g / (3.0 * b + g)
    nu = (3.0 * b - 2.0 * g) / (2.0 * (3.0 * b + g))
    from hardness import chen_fields  # noqa: WPS433

    out = {"B": b, "G": g, "E": e, "nu": nu}
    out.update(chen_fields(b, g))
    return out


def ionic_relax(atoms, calc, fmax: float, steps: int):
    atoms = atoms.copy()
    atoms.calc = calc
    LBFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    # ASE stress is σ = (1/V) ∂E/∂ε (tension > 0). Do not negate.
    stress = np.asarray(atoms.get_stress(voigt=False), dtype=float) * EV_A3_TO_GPA
    return atoms, stress


def main() -> None:
    parser = argparse.ArgumentParser(description="ASE MLIP elastic constants")
    parser.add_argument(
        "--structures",
        type=Path,
        default=Path("../cell_opt/relaxed/all_relaxed.extxyz"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("strained"))
    parser.add_argument("--energy", choices=EVAL_ENERGY, required=True)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument(
        "--uma-model",
        default="/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/uma-cache/uma-s-1p2.pt",
    )
    parser.add_argument("--uma-task", default="omat")
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--strains", type=float, nargs="+", default=[-0.01, -0.005, 0.005, 0.01]
    )
    args = parser.parse_args()

    frames = read(args.structures, index=":")
    if not isinstance(frames, list):
        frames = [frames]

    calc = make_calc(args)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for n, base in enumerate(frames):
        s_uni, e_uni = [], []
        for delta in args.strains:
            eps = np.zeros((3, 3))
            eps[0, 0] = delta
            atoms, stress = ionic_relax(apply_strain(base, eps), calc, args.fmax, args.steps)
            write(args.out_dir / f"sqs_{n:03d}_uni_{delta:+.4f}.extxyz", atoms)
            s_uni.append(stress)
            e_uni.append(delta)
        c11 = float(np.polyfit(e_uni, [s[0, 0] for s in s_uni], 1)[0])
        c12 = float(np.polyfit(e_uni, [s[1, 1] for s in s_uni], 1)[0])

        s_sh, e_sh = [], []
        for gamma in args.strains:
            eps = np.zeros((3, 3))
            eps[1, 2] = eps[2, 1] = 0.5 * gamma
            atoms, stress = ionic_relax(apply_strain(base, eps), calc, args.fmax, args.steps)
            write(args.out_dir / f"sqs_{n:03d}_shear_{gamma:+.4f}.extxyz", atoms)
            s_sh.append(stress)
            e_sh.append(gamma)
        c44 = float(np.polyfit(e_sh, [s[1, 2] for s in s_sh], 1)[0])
        der = vrh_cubic(c11, c12, c44)
        row = {
            "frame": n,
            "formula": base.get_chemical_formula(),
            "C11_GPa": c11,
            "C12_GPa": c12,
            "C44_GPa": c44,
            "B_GPa": der["B"],
            "G_GPa": der["G"],
            "E_GPa": der["E"],
            "nu": der["nu"],
            "G_over_B": der["G_over_B"],
            "Hv_Chen_GPa": der["Hv_Chen_GPa"],
            "energy_method": args.energy,
        }
        results.append(row)
        print(
            f"frame {n}: C11={c11:.2f} C12={c12:.2f} C44={c44:.2f} GPa  "
            f"B={der['B']:.2f} G={der['G']:.2f} E={der['E']:.2f} nu={der['nu']:.4f}  "
            f"Hv_Chen={der['Hv_Chen_GPa']:.2f}"
        )

    out_json = args.out_dir / "elastic.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
