#!/usr/bin/env python3
"""Build elemental reference energies for formation enthalpy (ΔH_f).

References (0 K, same energy method as the alloy):
  - Ti, Zr, Hf: bulk hcp metal
  - Nb, Ta: bulk bcc metal
  - N: ½ N₂ molecule in a large vacuum box (μ_N = E(N₂)/2)

Supports single-species workers (for multi-tile PBS) and a merge step::

  # one tile / one species
  python build_elemental_refs.py --energy uma --species Ti --out-dir refs/uma
  python build_elemental_refs.py --energy uma --species N --out-dir refs/uma

  # after all partials exist
  python build_elemental_refs.py --merge --out-dir refs/uma --out refs/uma/elemental_refs.json

  # serial all-in-one (legacy)
  python build_elemental_refs.py --energy uma --out refs/uma/elemental_refs.json
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
REQUIRED_SPECIES = ("Ti", "Zr", "Hf", "Nb", "Ta", "N")

METAL_BUILDERS: dict[str, Callable[[], Atoms]] = {
    "Ti": lambda: bulk("Ti", "hcp", cubic=False),
    "Zr": lambda: bulk("Zr", "hcp", cubic=False),
    "Hf": lambda: bulk("Hf", "hcp", cubic=False),
    "Nb": lambda: bulk("Nb", "bcc"),
    "Ta": lambda: bulk("Ta", "bcc"),
}


def make_n2(*, vacuum: float = 8.0) -> Atoms:
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
        opt_atoms = atoms
    LBFGS(opt_atoms, logfile=str(logfile) if logfile else "-").run(
        fmax=fmax, steps=steps
    )
    return atoms, float(atoms.get_potential_energy())


def _meta(args: argparse.Namespace) -> dict[str, Any]:
    return {
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
    }


def run_metal(args: argparse.Namespace, el: str) -> dict[str, Any]:
    calc = make_calc(args, uma_task=args.uma_task)
    atoms0 = METAL_BUILDERS[el]()
    log = args.out_dir / f"{el}.log"
    if args.no_relax:
        atoms = prepare_atoms(atoms0, energy=args.energy)
        atoms.calc = calc
        e = float(atoms.get_potential_energy())
    else:
        atoms, e = relax_and_energy(
            atoms0,
            calc,
            energy_method=args.energy,
            fmax=args.fmax,
            steps=args.steps,
            relax_cell=True,
            logfile=log,
        )
    e_pa = e / len(atoms)
    write(args.out_dir / f"{el}.extxyz", atoms)
    print(f"{el}: E={e:.6f} eV  E/atom={e_pa:.6f} eV")
    return {
        "species": el,
        **_meta(args),
        "per_atom_eV": {el: e_pa},
        "details": {
            el: {
                "structure": "hcp" if el in ("Ti", "Zr", "Hf") else "bcc",
                "n_atoms": len(atoms),
                "energy_eV": e,
                "energy_per_atom_eV": e_pa,
                "cell_A": np.asarray(atoms.cell.array, dtype=float).tolist(),
            }
        },
    }


def run_n2(args: argparse.Namespace) -> dict[str, Any]:
    calc = make_calc(args, uma_task=args.uma_task_n2)
    n2_0 = make_n2(vacuum=args.n2_vacuum)
    if args.no_relax:
        n2 = prepare_atoms(n2_0, energy=args.energy)
        n2.calc = calc
        e_n2 = float(n2.get_potential_energy())
    else:
        n2, e_n2 = relax_and_energy(
            n2_0,
            calc,
            energy_method=args.energy,
            fmax=args.fmax,
            steps=args.steps,
            relax_cell=False,
            logfile=args.out_dir / "N2.log",
        )
    mu_n = e_n2 / 2.0
    write(args.out_dir / "N2.extxyz", n2)
    print(f"N: E(N2)={e_n2:.6f} eV  μ_N=E(N2)/2={mu_n:.6f} eV")
    return {
        "species": "N",
        **_meta(args),
        "per_atom_eV": {"N": mu_n},
        "details": {
            "N": {
                "structure": "N2",
                "n_atoms": 2,
                "energy_eV": e_n2,
                "energy_per_atom_eV": mu_n,
                "n_reference": "0.5_N2",
                "bond_length_A": float(n2.get_distance(0, 1)),
                "uma_task": args.uma_task_n2 if args.energy == "uma" else None,
            }
        },
    }


def write_partial(out_dir: Path, payload: dict[str, Any]) -> Path:
    sp = payload["species"]
    path = out_dir / "partials" / f"{sp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return path


def merge_partials(out_dir: Path, out: Path, *, require_all: bool = True) -> dict[str, Any]:
    partial_dir = out_dir / "partials"
    files = sorted(partial_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No partials under {partial_dir}")
    per_atom: dict[str, float] = {}
    details: dict[str, Any] = {}
    meta: dict[str, Any] | None = None
    for f in files:
        part = json.loads(f.read_text(encoding="utf-8"))
        if meta is None:
            meta = {
                k: part.get(k)
                for k in (
                    "energy_method",
                    "device",
                    "dtype",
                    "uma_model",
                    "uma_task_metals",
                    "uma_task_n2",
                    "n_reference",
                    "definition",
                )
            }
        # N partial owns uma_task_n2 (metals may still carry a stale value).
        if part.get("species") == "N" and part.get("uma_task_n2") is not None:
            meta["uma_task_n2"] = part["uma_task_n2"]
        per_atom.update({k: float(v) for k, v in part["per_atom_eV"].items()})
        details.update(part["details"])
    assert meta is not None
    if require_all:
        missing = [s for s in REQUIRED_SPECIES if s not in per_atom]
        if missing:
            raise RuntimeError(f"Missing species in partials: {missing}")
    payload = {**meta, "per_atom_eV": per_atom, "details": details}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"merged {len(per_atom)} species → {out}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Elemental refs for formation enthalpy (metals + ½ N₂)"
    )
    parser.add_argument("--energy", choices=EVAL_ENERGY, default=None)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dtype", default="float64")
    parser.add_argument(
        "--uma-model",
        default="/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/uma-cache/uma-s-1p2.pt",
    )
    parser.add_argument("--uma-task", default="omat")
    # Same task as metals/nitrides so μ_N is on a consistent energy scale.
    parser.add_argument("--uma-task-n2", default="omat")
    parser.add_argument(
        "--species",
        type=str,
        default=None,
        help="Single species worker: Ti|Zr|Hf|Nb|Ta|N (writes partials/<sp>.json)",
    )
    parser.add_argument(
        "--elements",
        nargs="+",
        default=None,
        help="Serial metal list (default: all metals); ignored with --species/--merge",
    )
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--n2-vacuum", type=float, default=8.0)
    parser.add_argument("--out", type=Path, default=Path("elemental_refs.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("elemental_refs"))
    parser.add_argument("--no-relax", action="store_true")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge out-dir/partials/*.json into --out (no XPU work)",
    )
    parser.add_argument(
        "--skip-n2",
        action="store_true",
        help="Serial mode: metals only (no N₂)",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        merge_partials(args.out_dir, args.out)
        return

    if args.energy is None:
        raise SystemExit("--energy is required unless --merge")

    if args.species is not None:
        sp = args.species.strip()
        if sp == "N":
            payload = run_n2(args)
        elif sp in METAL_BUILDERS:
            payload = run_metal(args, sp)
        else:
            raise ValueError(
                f"Unknown --species {sp!r}; choose from {list(METAL_BUILDERS)+['N']}"
            )
        write_partial(args.out_dir, payload)
        return

    # Serial all-in-one
    metals = list(args.elements) if args.elements else list(METAL_BUILDERS)
    for el in metals:
        if el not in METAL_BUILDERS:
            raise ValueError(f"Unsupported element {el!r}")
    per_atom: dict[str, float] = {}
    details: dict[str, Any] = {}
    for el in metals:
        part = run_metal(args, el)
        write_partial(args.out_dir, part)
        per_atom.update(part["per_atom_eV"])
        details.update(part["details"])
    if not args.skip_n2:
        part = run_n2(args)
        write_partial(args.out_dir, part)
        per_atom.update(part["per_atom_eV"])
        details.update(part["details"])
    payload = {**_meta(args), "per_atom_eV": per_atom, "details": details}
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
