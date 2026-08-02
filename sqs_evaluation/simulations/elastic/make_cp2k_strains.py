#!/usr/bin/env python3
"""Build strained .extxyz + CP2K GEO_OPT inputs for elastic (cp2k_dft backend).

Strain generation is an elastic-property step; CP2K input rendering uses
``energy_models.cp2k.render_inp`` (not a separate CP2K simulation folder).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[2]  # sqs_evaluation/
sys.path.insert(0, str(ROOT))
from energy_models.cp2k import render_inp  # noqa: E402


def apply(atoms, eps):
    a = atoms.copy()
    a.set_cell(atoms.cell.array @ (np.eye(3) + eps), scale_atoms=True)
    return a


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strained cells + CP2K GEO_OPT inputs for elastic (cp2k_dft)"
    )
    parser.add_argument(
        "--structure",
        type=Path,
        default=Path("../cell_opt/relaxed/sqs_000_relaxed.extxyz"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("strained_extxyz"))
    parser.add_argument(
        "--strains", type=float, nargs="+", default=[-0.01, -0.005, 0.005, 0.01]
    )
    args = parser.parse_args()

    base = read(args.structure, index=0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    inp_dir = args.out_dir / "cp2k_inputs"
    inp_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for delta in args.strains:
        for tag, eps in (
            (f"uni_exx_{delta:+.4f}", np.diag([delta, 0.0, 0.0])),
            (
                f"shear_eyz_{delta:+.4f}",
                np.array([[0, 0, 0], [0, 0, 0.5 * delta], [0, 0.5 * delta, 0]]),
            ),
        ):
            atoms = apply(base, eps)
            write(args.out_dir / f"{tag}.extxyz", atoms)
            (inp_dir / f"{tag}.inp").write_text(
                render_inp(atoms, tag, run_type="GEO_OPT"), encoding="utf-8"
            )
            written += 1

    print(f"wrote {written} strained structures under {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
