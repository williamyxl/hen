#!/usr/bin/env python3
"""Convert SQS .extxyz -> LAMMPS data file (atomic, metal units)."""
from __future__ import annotations

import argparse
from pathlib import Path

from ase.io import read, write


def main() -> None:
    parser = argparse.ArgumentParser(description="extxyz -> LAMMPS data")
    parser.add_argument(
        "--structure",
        type=Path,
        default=Path("../../sqs_sampling/final_sqs/sqs_000.extxyz"),
    )
    parser.add_argument("--out", type=Path, default=Path("data.sqs"))
    parser.add_argument(
        "--type-order",
        nargs="+",
        required=True,
        help="Species order matching pair_coeff, e.g. Ti Zr Hf Nb Ta N",
    )
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()

    atoms = read(args.structure, index=args.frame)
    missing = sorted(set(atoms.get_chemical_symbols()) - set(args.type_order))
    if missing:
        raise ValueError(f"Species {missing} missing from --type-order")

    write(
        args.out,
        atoms,
        format="lammps-data",
        specorder=args.type_order,
        masses=True,
        atom_style="atomic",
    )
    print(f"wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
