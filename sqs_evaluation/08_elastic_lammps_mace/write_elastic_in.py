#!/usr/bin/env python3
"""Write LAMMPS+MACE uniaxial/shear strain inputs."""
from __future__ import annotations

import argparse
from pathlib import Path

UNIAXIAL = """# Generated uniaxial strain job
units           metal
atom_style      atomic
boundary        p p p

read_data       {data_file}

pair_style      mace no_domain_decomposition
pair_coeff      * * {mace_model} {species}

neighbor        1.0 bin
neigh_modify    delay 0 every 1 check yes

variable        delta equal {delta}
change_box      all x scale $(1+v_delta) remap units box

min_style       cg
minimize        1.0e-6 1.0e-8 10000 100000

variable        pxx equal pxx
variable        pyy equal pyy
variable        pzz equal pzz
print           "STRAIN ${{delta}} PXX ${{pxx}} PYY ${{pyy}} PZZ ${{pzz}}"
"""

SHEAR = """# Generated shear strain job
units           metal
atom_style      atomic
boundary        p p p

read_data       {data_file}

pair_style      mace no_domain_decomposition
pair_coeff      * * {mace_model} {species}

neighbor        1.0 bin
neigh_modify    delay 0 every 1 check yes

variable        delta equal {delta}
change_box      all triclinic
change_box      all yz delta $(v_delta*ly) remap units box

min_style       cg
minimize        1.0e-6 1.0e-8 10000 100000

variable        pyz equal pyz
print           "SHEAR ${{delta}} PYZ ${{pyz}}"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Write LAMMPS elastic strain inputs")
    parser.add_argument("--mace-model", type=Path, required=True)
    parser.add_argument("--type-order", nargs="+", required=True)
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("../04_relax_lammps_mace/data.sqs.relaxed"),
    )
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--mode", choices=["uniaxial", "shear"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tmpl = UNIAXIAL if args.mode == "uniaxial" else SHEAR
    args.out.write_text(
        tmpl.format(
            data_file=args.data_file.resolve().as_posix(),
            mace_model=args.mace_model.resolve().as_posix(),
            species=" ".join(args.type_order),
            delta=args.delta,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
