#!/usr/bin/env python3
"""ASE + MLIP ionic ± cell relaxation of SQS .extxyz structures.

Target properties: total energy; mixing enthalpy (vs --eref-json).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import FIRE

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sqs_sampling"))
from energy import SUPPORTED, build_calculator  # noqa: E402


def mixing_enthalpy(atoms, energy: float, eref: dict[str, float]) -> float:
    """ΔH_mix per atom = E/N - sum(c_i E_i_ref)."""
    symbols = atoms.get_chemical_symbols()
    n = len(symbols)
    if n == 0:
        raise ValueError("Empty structure")
    missing = sorted(set(symbols) - set(eref))
    if missing:
        raise KeyError(f"Reference energies missing species: {missing}")
    e_ref = sum(eref[s] for s in symbols) / n
    return energy / n - e_ref


def main() -> None:
    parser = argparse.ArgumentParser(description="ASE MLIP relax SQS structures")
    parser.add_argument(
        "--structures",
        type=Path,
        default=Path("../../sqs_sampling/final_sqs/sqs.extxyz"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("relaxed"))
    parser.add_argument("--energy", choices=SUPPORTED, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mace-model", default=None)
    parser.add_argument("--uma-model", default="uma-s-1p2")
    parser.add_argument("--uma-task", default="omat")
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--relax-cell", action="store_true", default=True)
    parser.add_argument("--no-relax-cell", action="store_false", dest="relax_cell")
    parser.add_argument(
        "--eref-json",
        type=Path,
        default=None,
        help='JSON map species -> E_ref (eV/atom), e.g. {"Ti": -x, "N": -y}',
    )
    args = parser.parse_args()

    if not args.structures.is_file():
        raise FileNotFoundError(f"Structures not found: {args.structures.resolve()}")
    if args.steps < 1:
        raise ValueError(f"--steps must be >= 1, got {args.steps}")

    eref = None
    if args.eref_json is not None:
        if not args.eref_json.is_file():
            raise FileNotFoundError(args.eref_json.resolve())
        with open(args.eref_json, encoding="utf-8") as f:
            eref = json.load(f)
        if not isinstance(eref, dict):
            raise ValueError("--eref-json must be a JSON object of species -> float")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = read(args.structures, index=":")
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) == 0:
        raise RuntimeError(f"No frames in {args.structures}")

    calc = build_calculator(
        args.energy,
        device=args.device,
        uma_model=args.uma_model,
        uma_task=args.uma_task,
        mace_model=args.mace_model,
    )

    relaxed = []
    summary = []
    for n, atoms in enumerate(frames):
        atoms = atoms.copy()
        atoms.calc = calc
        opt_atoms = FrechetCellFilter(atoms) if args.relax_cell else atoms
        opt = FIRE(opt_atoms, logfile=str(args.out_dir / f"sqs_{n:03d}.log"))
        opt.run(fmax=args.fmax, steps=args.steps)
        max_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
        if max_force > args.fmax:
            raise RuntimeError(
                f"frame {n}: relax failed (max force={max_force:.4f} > fmax={args.fmax})"
            )
        e = float(atoms.get_potential_energy())
        if not np.isfinite(e):
            raise RuntimeError(f"frame {n}: non-finite energy {e!r}")
        row = {
            "frame": n,
            "formula": atoms.get_chemical_formula(),
            "energy_eV": e,
            "energy_method": args.energy,
        }
        if eref is not None:
            row["delta_h_mix_eV_per_atom"] = mixing_enthalpy(atoms, e, eref)
        summary.append(row)
        out = args.out_dir / f"sqs_{n:03d}_relaxed.extxyz"
        atoms.info.update(row)
        write(out, atoms)
        relaxed.append(atoms)
        print(
            f"frame {n}: E={e:.6f} eV"
            + (
                f"  dH_mix={row['delta_h_mix_eV_per_atom']:.6f} eV/atom"
                if eref is not None
                else ""
            )
        )

    write(args.out_dir / "all_relaxed.extxyz", relaxed)
    with open(args.out_dir / "energies.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out_dir / 'all_relaxed.extxyz'}")
    print(f"wrote {args.out_dir / 'energies.json'}")


if __name__ == "__main__":
    main()
