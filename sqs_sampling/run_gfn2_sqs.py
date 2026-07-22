#!/usr/bin/env python3
"""Single-point GFN2-xTB energy on an SQS initial structure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# GFN2-xTB / TBLite: fixed threading (set before OpenMP libs load)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_STACKSIZE"] = "8G"

import numpy as np
import yaml
from ase.io import read

from energy import build_calculator
from lattice import composition_string
from mc_sqs import build_initial


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-point GFN2-xTB on SQS structure")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path, help="Build initial SQS from YAML")
    src.add_argument("--structure", type=Path, help="Load .extxyz")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if args.seed is not None:
            cfg["seed"] = args.seed
        atoms = build_initial(cfg, np.random.default_rng(int(cfg["seed"])))
        source = str(args.config)
    else:
        atoms = read(args.structure, index=0)
        source = str(args.structure)

    atoms = atoms.copy()
    atoms.calc = build_calculator("gfn2-xtb")
    energy = float(atoms.get_potential_energy())

    summary = {
        "source": source,
        "composition": composition_string(atoms),
        "natoms": len(atoms),
        "energy_eV": energy,
        "energy_eV_per_atom": energy / len(atoms),
        "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
        "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
        "OMP_STACKSIZE": os.environ["OMP_STACKSIZE"],
    }
    print(json.dumps(summary, indent=2))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
