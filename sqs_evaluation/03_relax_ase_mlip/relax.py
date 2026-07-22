#!/usr/bin/env python3
"""ASE + MLIP ionic ± cell relaxation of SQS .extxyz structures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import LBFGS

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sqs_sampling"))
from energy import SUPPORTED, build_calculator  # noqa: E402


def mixing_enthalpy(atoms, energy: float, eref: dict[str, float]) -> float:
    """ΔH_mix per atom = E/N - sum(c_i E_i_ref)."""
    symbols = atoms.get_chemical_symbols()
    return energy / len(symbols) - sum(eref[s] for s in symbols) / len(symbols)


def main() -> None:
    parser = argparse.ArgumentParser(description="ASE MLIP relax SQS structures")
    parser.add_argument(
        "--structures",
        type=Path,
        default=Path("../../sqs_sampling/final_sqs/sqs.extxyz"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("relaxed"))
    parser.add_argument("--energy", choices=SUPPORTED, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mace-model", default=None)
    parser.add_argument(
        "--uma-model", default="/mnt/d/workdir/uma-cache/uma-s-1p2.pt"
    )
    parser.add_argument("--uma-task", default="omat")
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--relax-cell", action="store_true", default=True)
    parser.add_argument("--no-relax-cell", action="store_false", dest="relax_cell")
    parser.add_argument("--eref-json", type=Path, default=None)
    args = parser.parse_args()

    eref = json.loads(args.eref_json.read_text()) if args.eref_json else None
    frames = read(args.structures, index=":")
    if not isinstance(frames, list):
        frames = [frames]

    calc = build_calculator(
        args.energy,
        device=args.device,
        uma_model=args.uma_model,
        uma_task=args.uma_task,
        mace_model=args.mace_model,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    relaxed, summary = [], []
    for n, atoms in enumerate(frames):
        atoms = atoms.copy()
        atoms.calc = calc
        opt_atoms = FrechetCellFilter(atoms) if args.relax_cell else atoms
        LBFGS(opt_atoms, logfile=str(args.out_dir / f"sqs_{n:03d}.log")).run(
            fmax=args.fmax, steps=args.steps
        )
        e = float(atoms.get_potential_energy())
        row = {"frame": n, "formula": atoms.get_chemical_formula(), "energy_eV": e, "energy_method": args.energy}
        if eref is not None:
            row["delta_h_mix_eV_per_atom"] = mixing_enthalpy(atoms, e, eref)
        summary.append(row)
        atoms.info.update(row)
        write(args.out_dir / f"sqs_{n:03d}_relaxed.extxyz", atoms)
        relaxed.append(atoms)
        msg = f"frame {n}: E={e:.6f} eV"
        if eref is not None:
            msg += f"  dH_mix={row['delta_h_mix_eV_per_atom']:.6f} eV/atom"
        print(msg)

    write(args.out_dir / "all_relaxed.extxyz", relaxed)
    (args.out_dir / "energies.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
