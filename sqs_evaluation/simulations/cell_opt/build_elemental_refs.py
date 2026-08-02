#!/usr/bin/env python3
"""Build elemental reference energies for formation enthalpy (ΔH_f).

References (0 K, same energy method as the alloy):
  - Ti, Zr, Hf: bulk hcp metal
  - Nb, Ta: bulk bcc metal
  - N: ½ N₂ molecule in a large vacuum box (μ_N = E(N₂)/2)

Writes JSON consumed by ``relax.py --elemental-eref-json``.

Example::

  python build_elemental_refs.py --energy uma --device xpu --out elemental_refs.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from ase import Atoms
from ase.build import bulk, molecule
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.optimize import LBFGS

ROOT = Path(__file__).resolve().parents[3]  # hen/
sys.path.insert(0, str(ROOT / "sqs_sampling"))
from energy import (  # noqa: E402
    UMA_DEFAULTS,
    UMA_SPIN_OFF,
    configure_gfn2_threads,
    gfn2_tblite_params,
    uma_predict_unit,
)

EVAL_ENERGY = ("gfn2-xtb", "uma")

# Standard-state crystals for HEN metals (ASE bulk prototypes).
METAL_BUILDERS: dict[str, Callable[[], Atoms]] = {
    "Ti": lambda: bulk("Ti", "hcp", cubic=False),
    "Zr": lambda: bulk("Zr", "hcp", cubic=False),
    "Hf": lambda: bulk("Hf", "hcp", cubic=False),
    "Nb": lambda: bulk("Nb", "bcc"),
    "Ta": lambda: bulk("Ta", "bcc"),
}


def make_n2(*, vacuum: float = 8.0) -> Atoms:
    """N₂ in a cubic cell with vacuum; PBC on for MLIP calculators."""
    atoms = molecule("N2")
    atoms.center(vacuum=vacuum)
    atoms.pbc = True
    return atoms


def make_calc(args: argparse.Namespace, *, uma_task: str):
    if args.energy == "gfn2-xtb":
        from tblite.ase import TBLite

        configure_gfn2_threads()
        return TBLite(**gfn2_tblite_params(charge=0))
    if args.energy == "uma":
        from fairchem.core import FAIRChemCalculator

        return FAIRChemCalculator(
            uma_predict_unit(
                model=args.uma_model or UMA_DEFAULTS["model"],
                device=args.device,
                dtype=args.dtype,
                workers=1,
            ),
            task_name=uma_task,
        )
    raise ValueError(f"Unknown energy method {args.energy!r}; choose from {EVAL_ENERGY}")


def prepare_atoms(atoms: Atoms, *, energy: str) -> Atoms:
    atoms = atoms.copy()
    if energy == "uma":
        atoms.info["charge"] = 0
        atoms.info["spin"] = UMA_SPIN_OFF
    return atoms


def relax_and_energy(
    atoms: Atoms,
    calc: Any,
    *,
    energy_method: str,
    fmax: float,
    steps: int,
    relax_cell: bool,
    logfile: Path | None,
) -> tuple[Atoms, float]:
    atoms = prepare_atoms(atoms, energy=energy_method)
    atoms.calc = calc
    if relax_cell and atoms.pbc.any() and len(atoms) > 2:
        opt_atoms = FrechetCellFilter(atoms)
    else:
        # Molecules / tiny cells: ionic only (keep box)
        opt_atoms = atoms
    LBFGS(opt_atoms, logfile=str(logfile) if logfile else "-").run(
        fmax=fmax, steps=steps
    )
    return atoms, float(atoms.get_potential_energy())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Elemental refs for formation enthalpy (metals + ½ N₂)"
    )
    parser.add_argument("--energy", choices=EVAL_ENERGY, required=True)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument(
        "--uma-model",
        default="/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/uma-cache/uma-s-1p2.pt",
    )
    parser.add_argument(
        "--uma-task",
        default="omat",
        help="UMA task for bulk metals (default: omat)",
    )
    parser.add_argument(
        "--uma-task-n2",
        default="omol",
        help="UMA task for N₂ (default: omol; use omat only if omol unavailable)",
    )
    parser.add_argument("--elements", nargs="+", default=list(METAL_BUILDERS))
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--n2-vacuum", type=float, default=8.0)
    parser.add_argument("--out", type=Path, default=Path("elemental_refs.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("elemental_refs"))
    parser.add_argument(
        "--no-relax",
        action="store_true",
        help="Single-point only (no ionic/cell relaxation)",
    )
    args = parser.parse_args()

    for el in args.elements:
        if el not in METAL_BUILDERS:
            raise ValueError(
                f"Unsupported element {el!r}; choose from {sorted(METAL_BUILDERS)}"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    details: dict[str, Any] = {}
    per_atom: dict[str, float] = {}

    # Metals (crystal task)
    metal_calc = make_calc(args, uma_task=args.uma_task)
    for el in args.elements:
        atoms0 = METAL_BUILDERS[el]()
        log = args.out_dir / f"{el}.log"
        if args.no_relax:
            atoms = prepare_atoms(atoms0, energy=args.energy)
            atoms.calc = metal_calc
            e = float(atoms.get_potential_energy())
        else:
            atoms, e = relax_and_energy(
                atoms0,
                metal_calc,
                energy_method=args.energy,
                fmax=args.fmax,
                steps=args.steps,
                relax_cell=True,
                logfile=log,
            )
        e_pa = e / len(atoms)
        per_atom[el] = e_pa
        details[el] = {
            "structure": "hcp" if el in ("Ti", "Zr", "Hf") else "bcc",
            "n_atoms": len(atoms),
            "energy_eV": e,
            "energy_per_atom_eV": e_pa,
            "cell_A": np.asarray(atoms.cell.array, dtype=float).tolist(),
        }
        write(args.out_dir / f"{el}.extxyz", atoms)
        print(f"{el}: E={e:.6f} eV  E/atom={e_pa:.6f} eV  ({details[el]['structure']})")

    # Nitrogen: μ_N = E(N₂)/2
    n2_calc = make_calc(args, uma_task=args.uma_task_n2)
    n2_0 = make_n2(vacuum=args.n2_vacuum)
    if args.no_relax:
        n2 = prepare_atoms(n2_0, energy=args.energy)
        n2.calc = n2_calc
        e_n2 = float(n2.get_potential_energy())
    else:
        n2, e_n2 = relax_and_energy(
            n2_0,
            n2_calc,
            energy_method=args.energy,
            fmax=args.fmax,
            steps=args.steps,
            relax_cell=False,
            logfile=args.out_dir / "N2.log",
        )
    mu_n = e_n2 / 2.0
    per_atom["N"] = mu_n
    details["N"] = {
        "structure": "N2",
        "n_atoms": 2,
        "energy_eV": e_n2,
        "energy_per_atom_eV": mu_n,
        "n_reference": "0.5_N2",
        "bond_length_A": float(n2.get_distance(0, 1)),
        "uma_task": args.uma_task_n2 if args.energy == "uma" else None,
    }
    write(args.out_dir / "N2.extxyz", n2)
    print(f"N: E(N2)={e_n2:.6f} eV  μ_N=E(N2)/2={mu_n:.6f} eV")

    payload = {
        "energy_method": args.energy,
        "device": args.device,
        "dtype": args.dtype,
        "uma_model": args.uma_model if args.energy == "uma" else None,
        "uma_task_metals": args.uma_task if args.energy == "uma" else None,
        "uma_task_n2": args.uma_task_n2 if args.energy == "uma" else None,
        "n_reference": "0.5_N2",
        "definition": (
            "ΔH_f / N_atoms = E(crystal)/N - Σ_i μ_i / N, "
            "with μ_M = E(bulk metal)/atom and μ_N = E(N2)/2"
        ),
        "per_atom_eV": per_atom,
        "details": details,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
