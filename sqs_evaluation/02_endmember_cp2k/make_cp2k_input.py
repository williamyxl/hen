#!/usr/bin/env python3
"""Build a complete CP2K CELL_OPT input from an end-member .extxyz."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase.io import read

DEFAULT_POTENTIAL = {
    "Ti": "GTH-PBE-q12",
    "Zr": "GTH-PBE-q12",
    "Hf": "GTH-PBE-q12",
    "Nb": "GTH-PBE-q13",
    "Ta": "GTH-PBE-q13",
    "V": "GTH-PBE-q13",
    "Mo": "GTH-PBE-q14",
    "W": "GTH-PBE-q14",
    "Al": "GTH-PBE-q3",
    "N": "GTH-PBE-q5",
}
DEFAULT_BASIS = "DZVP-MOLOPT-SR-GTH"


def kind_block(symbol: str) -> str:
    if symbol not in DEFAULT_POTENTIAL:
        raise KeyError(
            f"No default CP2K POTENTIAL for {symbol}; extend DEFAULT_POTENTIAL "
            "in make_cp2k_input.py"
        )
    return (
        f"    &KIND {symbol}\n"
        f"      BASIS_SET {DEFAULT_BASIS}\n"
        f"      POTENTIAL {DEFAULT_POTENTIAL[symbol]}\n"
        f"    &END KIND\n"
    )


def render_inp(atoms, project: str, run_type: str = "CELL_OPT") -> str:
    if run_type not in {"CELL_OPT", "GEO_OPT", "ENERGY"}:
        raise ValueError(f"Unsupported RUN_TYPE {run_type!r}")
    cell = atoms.cell.array
    if not np.all(np.isfinite(cell)) or not np.all(np.isfinite(atoms.get_positions())):
        raise RuntimeError("Non-finite cell or coordinates")
    species = sorted(set(atoms.get_chemical_symbols()))
    kinds = "".join(kind_block(s) for s in species)
    coords = "\n".join(
        f"      {s} {x:.10f} {y:.10f} {z:.10f}"
        for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions())
    )
    motion = ""
    if run_type == "CELL_OPT":
        motion = """
&MOTION
  &CELL_OPT
    TYPE DIRECT_CELL_OPT
    OPTIMIZER LBFGS
    MAX_ITER 200
    KEEP_SYMMETRY TRUE
  &END CELL_OPT
&END MOTION
"""
    elif run_type == "GEO_OPT":
        motion = """
&MOTION
  &GEO_OPT
    OPTIMIZER LBFGS
    MAX_ITER 200
  &END GEO_OPT
&END MOTION
"""
    return f"""&GLOBAL
  PROJECT {project}
  RUN_TYPE {run_type}
  PRINT_LEVEL LOW
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep
  STRESS_TENSOR ANALYTICAL
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    POTENTIAL_FILE_NAME GTH_POTENTIALS
    &MGRID
      CUTOFF 600
      REL_CUTOFF 60
    &END MGRID
    &QS
      METHOD GPW
      EPS_DEFAULT 1.0E-12
    &END QS
    &SCF
      SCF_GUESS ATOMIC
      EPS_SCF 1.0E-6
      MAX_SCF 100
      &OT
        PRECONDITIONER FULL_SINGLE_INVERSE
        MINIMIZER DIIS
      &END OT
    &END SCF
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
    &END XC
    &PRINT
      &STRESS_TENSOR ON
      &END STRESS_TENSOR
    &END PRINT
  &END DFT
  &SUBSYS
    &CELL
      A {cell[0, 0]:.10f} {cell[0, 1]:.10f} {cell[0, 2]:.10f}
      B {cell[1, 0]:.10f} {cell[1, 1]:.10f} {cell[1, 2]:.10f}
      C {cell[2, 0]:.10f} {cell[2, 1]:.10f} {cell[2, 2]:.10f}
      PERIODIC XYZ
    &END CELL
    &COORD
{coords}
    &END COORD
{kinds}  &END SUBSYS
&END FORCE_EVAL
{motion}"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Write CP2K input from extxyz")
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument(
        "--run-type",
        choices=["CELL_OPT", "GEO_OPT", "ENERGY"],
        default="CELL_OPT",
    )
    args = parser.parse_args()

    if not args.structure.is_file():
        raise FileNotFoundError(args.structure.resolve())
    atoms = read(args.structure, index=0)
    if len(atoms) == 0:
        raise ValueError("Empty structure")

    project = args.project or f"endmember_{atoms.get_chemical_formula()}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_inp(atoms, project, args.run_type), encoding="utf-8")
    print(f"wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
