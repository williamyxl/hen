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

ROOT = Path(__file__).resolve().parents[3]  # hen/
sys.path.insert(0, str(ROOT / "sqs_sampling"))
from energy import (  # noqa: E402
    UMA_DEFAULTS,
    configure_gfn2_threads,
    gfn2_tblite_params,
    uma_predict_unit,
)

EVAL_ENERGY = ("gfn2-xtb", "uma")  # FairChem UMA production; GFN2 optional


def _eref_per_atom_map(eref: dict) -> dict[str, float]:
    """Accept either a flat {el: eV} map or elemental_refs.json with per_atom_eV."""
    if "per_atom_eV" in eref and isinstance(eref["per_atom_eV"], dict):
        return {str(k): float(v) for k, v in eref["per_atom_eV"].items()}
    out = {
        str(k): float(v)
        for k, v in eref.items()
        if isinstance(v, (int, float)) and 1 <= len(str(k)) <= 2
    }
    if not out:
        raise ValueError(
            "eref must be {El: eV/atom} or elemental_refs.json with per_atom_eV"
        )
    return out


def mixing_enthalpy(atoms, energy: float, eref: dict[str, float]) -> float:
    """ΔH_mix per atom vs end-member / custom refs: E/N − Σ μ_i / N.

    ``eref`` maps element symbol → reference energy per atom (eV). For rocksalt
    MN end-members, use E(MN)/2 for both M and N (or an equivalent scheme).
    """
    refs = _eref_per_atom_map(eref)
    symbols = atoms.get_chemical_symbols()
    missing = sorted(set(symbols) - set(refs))
    if missing:
        raise KeyError(f"mixing eref missing elements {missing}")
    return energy / len(symbols) - sum(refs[s] for s in symbols) / len(symbols)


def formation_enthalpy(atoms, energy: float, elemental_eref: dict) -> float:
    """Formation enthalpy per atom vs elemental standards (eV/atom).

    ΔH_f / N = E(crystal)/N − Σ_i μ_i / N
    with μ_M from bulk metal and μ_N = E(N₂)/2 (see build_elemental_refs.py).
    """
    refs = _eref_per_atom_map(elemental_eref)
    symbols = atoms.get_chemical_symbols()
    missing = sorted(set(symbols) - set(refs))
    if missing:
        raise KeyError(f"elemental eref missing elements {missing}")
    return energy / len(symbols) - sum(refs[s] for s in symbols) / len(symbols)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="ASE MLIP relax SQS structures")
    parser.add_argument(
        "--structures",
        type=Path,
        default=Path("../../sqs_sampling/final_sqs/sqs.extxyz"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("relaxed"))
    parser.add_argument("--energy", choices=EVAL_ENERGY, required=True)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument(
        "--uma-model",
        default="/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/uma-cache/uma-s-1p2.pt",
    )
    parser.add_argument("--uma-task", default="omat")
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--relax-cell", action="store_true", default=True)
    parser.add_argument("--no-relax-cell", action="store_false", dest="relax_cell")
    parser.add_argument("--eref-json", type=Path, default=None,
                        help="End-member / custom refs for ΔH_mix (flat map or per_atom_eV)")
    parser.add_argument(
        "--elemental-eref-json",
        type=Path,
        default=None,
        help="Elemental refs from build_elemental_refs.py for ΔH_f",
    )
    args = parser.parse_args()

    eref = json.loads(args.eref_json.read_text()) if args.eref_json else None
    elemental_eref = (
        json.loads(args.elemental_eref_json.read_text())
        if args.elemental_eref_json
        else None
    )
    frames = read(args.structures, index=":")
    if not isinstance(frames, list):
        frames = [frames]

    calc = make_calc(args)

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
        if elemental_eref is not None:
            row["delta_h_f_eV_per_atom"] = formation_enthalpy(atoms, e, elemental_eref)
        summary.append(row)
        atoms.info.update(row)
        write(args.out_dir / f"sqs_{n:03d}_relaxed.extxyz", atoms)
        relaxed.append(atoms)
        msg = f"frame {n}: E={e:.6f} eV"
        if eref is not None:
            msg += f"  dH_mix={row['delta_h_mix_eV_per_atom']:.6f} eV/atom"
        if elemental_eref is not None:
            msg += f"  dH_f={row['delta_h_f_eV_per_atom']:.6f} eV/atom"
        print(msg)

    write(args.out_dir / "all_relaxed.extxyz", relaxed)
    (args.out_dir / "energies.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
