#!/usr/bin/env python3
"""Single-point energy (GFN2-xTB or FairChem UMA on Intel XPU) on an SQS structure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# GFN2-xTB / TBLite: fixed threading (set before OpenMP libs load)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_STACKSIZE"] = "4G"

import numpy as np
import yaml
from ase.io import read

from energy import (
    CBMC_SUPPORTED,
    UMA_DEFAULTS,
    configure_gfn2_threads,
    formal_charge_and_multiplicity,
    gfn2_tblite_params,
    set_uma_spin_charge,
    uma_predict_unit,
)
from lattice import composition_string
from mc_sqs import build_initial


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-point GFN2-xTB or FairChem UMA (XPU) on SQS structure"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", type=Path, help="Build initial SQS from YAML")
    src.add_argument("--structure", type=Path, help="Load .extxyz")
    parser.add_argument(
        "--energy",
        choices=CBMC_SUPPORTED,
        default=None,
        help="Energy method (default: config energy, else uma)",
    )
    parser.add_argument("--device", default=None, help="UMA device (xpu; cpu for debug)")
    parser.add_argument("--dtype", default=None, help="FairChem dtype (float64 required)")
    parser.add_argument("--uma-model", default=None)
    parser.add_argument("--uma-task", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    cfg: dict = {}
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

    method = (
        args.energy
        or str(cfg.get("energy", "uma")).strip().lower()
    )
    if method not in CBMC_SUPPORTED:
        raise ValueError(
            f"energy method {method!r} not supported; choose from {CBMC_SUPPORTED}"
        )

    device = args.device or str(cfg.get("device", "xpu"))
    dtype = args.dtype or str(cfg.get("dtype", "float64"))
    uma_model = args.uma_model or (
        str(cfg["uma_model"]) if cfg.get("uma_model") else UMA_DEFAULTS["model"]
    )
    uma_task = args.uma_task or str(cfg.get("uma_task", "omat"))

    atoms = atoms.copy()
    charge, multiplicity = formal_charge_and_multiplicity(atoms)
    if method == "gfn2-xtb":
        from tblite.ase import TBLite

        configure_gfn2_threads()
        atoms.calc = TBLite(**gfn2_tblite_params(atoms, charge=charge))
    else:
        from fairchem.core import FAIRChemCalculator

        atoms = set_uma_spin_charge(atoms, charge=charge)
        atoms.calc = FAIRChemCalculator(
            uma_predict_unit(model=uma_model, device=device, dtype=dtype, workers=1),
            task_name=uma_task,
        )
    energy = float(atoms.get_potential_energy())

    summary = {
        "source": source,
        "composition": composition_string(atoms),
        "natoms": len(atoms),
        "energy_method": method,
        "formal_charge": charge,
        "formal_multiplicity": multiplicity,
        "energy_eV": energy,
        "energy_eV_per_atom": energy / len(atoms),
    }
    if method == "gfn2-xtb":
        summary.update(
            {
                "tblite_multiplicity": 1,
                "tblite_spin_ignored": True,
                "OMP_NUM_THREADS": os.environ["OMP_NUM_THREADS"],
                "MKL_NUM_THREADS": os.environ["MKL_NUM_THREADS"],
                "OMP_STACKSIZE": os.environ["OMP_STACKSIZE"],
            }
        )
    else:
        summary.update(
            {
                "uma_model": uma_model,
                "uma_task": uma_task,
                "device": device,
                "atoms_info_charge": atoms.info.get("charge"),
                "atoms_info_spin": atoms.info.get("spin"),
            }
        )
    print(json.dumps(summary, indent=2))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
