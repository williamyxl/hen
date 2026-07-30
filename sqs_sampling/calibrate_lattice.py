#!/usr/bin/env python3
"""Cubic rocksalt end-member lattice calibration (TiN…TaN) for SQS sampling.

Optimizes each MN cell (same supercell as SQS) with hydrostatic (cubic) cell
relaxation using the energy backend from the YAML config, then reports a Vegard
conventional-cell a for the target cation composition.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase.filters import FrechetCellFilter
from ase.io import write
from ase.optimize import LBFGS

from energy import (
    CBMC_SUPPORTED,
    UMA_DEFAULTS,
    configure_gfn2_threads,
    formal_charge_and_multiplicity,
    gfn2_tblite_params,
    set_uma_spin_charge,
    uma_predict_unit,
)
from lattice import (
    DEFAULT_ENDMEMBER_CATIONS,
    build_rocksalt_supercell,
    vegard_lattice_constant,
)

log = logging.getLogger("calibrate_lattice")


def calculator_from_config(cfg: dict, *, atoms=None) -> Any:
    method = str(cfg.get("energy", "uma")).strip().lower()
    if method not in CBMC_SUPPORTED:
        raise ValueError(
            f"energy method {method!r} not supported; choose from {CBMC_SUPPORTED}"
        )
    if method == "gfn2-xtb":
        from tblite.ase import TBLite

        configure_gfn2_threads()
        return method, TBLite(**gfn2_tblite_params(atoms))

    from fairchem.core import FAIRChemCalculator

    return method, FAIRChemCalculator(
        uma_predict_unit(
            model=str(cfg.get("uma_model") or UMA_DEFAULTS["model"]),
            device=str(cfg.get("device", "xpu")),
            dtype=str(cfg.get("dtype", "float64")),
            workers=int(cfg.get("uma_workers", 1)),
        ),
        task_name=str(cfg.get("uma_task", "omat")),
    )


def optimize_cubic_rocksalt(
    cation: str,
    anion: str,
    a0: float,
    calc: Any,
    *,
    method: str,
    supercell: tuple[int, int, int],
    fmax: float,
    steps: int,
    logfile: Path | None = None,
) -> tuple[float, float, Any, int, int]:
    """Hydrostatic cell opt (cubic). Returns (a_conv, E, atoms, charge, spin)."""
    nx, ny, nz = supercell
    if nx != ny or ny != nz:
        raise ValueError(f"calibration requires cubic supercell, got {supercell}")

    atoms = build_rocksalt_supercell(anion, a0, supercell, cation=cation)
    formal_q, formal_m = formal_charge_and_multiplicity(atoms)
    if method == "uma":
        atoms = set_uma_spin_charge(atoms, charge=formal_q)
        charge = int(atoms.info["charge"])
        spin = int(atoms.info["spin"])  # always 0 (spin off)
    else:
        charge = formal_q
        spin = 1  # TBLite closed-shell

    atoms.calc = calc
    filt = FrechetCellFilter(atoms, hydrostatic_strain=True)
    opt = LBFGS(filt, logfile=str(logfile) if logfile else None)
    opt.run(fmax=fmax, steps=steps)

    # Enforce exact cubic supercell; report conventional-cell a = L / n
    L = float(np.mean(atoms.cell.lengths()))
    atoms.set_cell(np.eye(3) * L, scale_atoms=True)
    if np.max(np.abs(atoms.cell.lengths() - L)) > 1e-4:
        raise ValueError(f"Cell not cubic after opt: {atoms.cell.lengths()}")
    if np.max(np.abs(atoms.cell.angles() - 90.0)) > 1e-2:
        raise ValueError(f"Cell not cubic after opt: {atoms.cell.angles()}")
    a_conv = L / float(nx)
    energy = float(atoms.get_potential_energy())
    atoms.info["formal_spin"] = formal_m
    return a_conv, energy, atoms, charge, spin


def run_calibration(cfg: dict) -> dict:
    anion = str(cfg.get("anion", "N"))
    cations = tuple(cfg.get("calibrate_cations", DEFAULT_ENDMEMBER_CATIONS))
    a0 = float(cfg.get("a_guess", cfg.get("a", 4.25)))
    fmax = float(cfg.get("calibration_fmax", 0.01))
    steps = int(cfg.get("calibration_steps", 200))
    supercell = tuple(int(x) for x in cfg.get("supercell", [1, 1, 1]))
    out_dir = Path(cfg.get("calibration_dir", "calibration"))
    out_dir.mkdir(parents=True, exist_ok=True)

    method = str(cfg.get("energy", "uma")).strip().lower()
    if method not in CBMC_SUPPORTED:
        raise ValueError(
            f"energy method {method!r} not supported; choose from {CBMC_SUPPORTED}"
        )
    shared_calc = None
    if method != "gfn2-xtb":
        _, shared_calc = calculator_from_config(cfg)

    endmembers: dict[str, dict[str, Any]] = {}
    a_by_cation: dict[str, float] = {}

    for cation in cations:
        formula = f"{cation}{anion}"
        probe = build_rocksalt_supercell(anion, a0, supercell, cation=cation)
        if method == "gfn2-xtb":
            _, calc = calculator_from_config(cfg, atoms=probe)
        else:
            calc = shared_calc
        formal_q, formal_m = formal_charge_and_multiplicity(probe)
        log.info(
            "calibrating %s  supercell=%s  a0=%.4f Å  energy=%s  "
            "formal_charge=%d formal_spin=%d  fmax=%s",
            formula,
            list(supercell),
            a0,
            method,
            formal_q,
            formal_m,
            fmax,
        )
        a_opt, energy, atoms, q_use, m_use = optimize_cubic_rocksalt(
            cation,
            anion,
            a0,
            calc,
            method=method,
            supercell=supercell,
            fmax=fmax,
            steps=steps,
            logfile=out_dir / f"{formula}.opt.log",
        )
        atoms.info.update(
            {
                "energy_eV": energy,
                "energy_method": method,
                "a_A": a_opt,
                "supercell": list(supercell),
                "calibration": "cubic_hydrostatic",
            }
        )
        atoms.calc = None
        write(out_dir / f"{formula}.extxyz", atoms)
        endmembers[formula] = {
            "cation": cation,
            "anion": anion,
            "supercell": list(supercell),
            "natoms": len(atoms),
            "a_A": a_opt,
            "supercell_edge_A": a_opt * supercell[0],
            "energy_eV": energy,
            "energy_eV_per_atom": energy / len(atoms),
            "a0_A": a0,
            "formal_charge": formal_q,
            "formal_spin": formal_m,
            "uma_charge": q_use if method == "uma" else None,
            "uma_spin": m_use if method == "uma" else None,
        }
        a_by_cation[cation] = a_opt
        log.info(
            "  %s  a_conv=%.6f Å  E=%.6f eV (%.6f eV/atom)  uma_spin=%s",
            formula,
            a_opt,
            energy,
            energy / len(atoms),
            m_use if method == "uma" else "n/a",
        )

    composition = dict(cfg.get("cation_composition", {}))
    result: dict[str, Any] = {
        "energy_method": method,
        "device": cfg.get("device"),
        "uma_model": cfg.get("uma_model"),
        "uma_task": cfg.get("uma_task"),
        "anion": anion,
        "supercell": list(supercell),
        "a_guess_A": a0,
        "calibration_fmax": fmax,
        "endmembers": endmembers,
        "a_by_cation_A": a_by_cation,
    }
    if composition:
        vegard = vegard_lattice_constant(composition, a_by_cation)
        result["cation_composition"] = composition
        result["vegard_a_A"] = vegard
        log.info(
            "Vegard a for composition %s: %.6f Å (conventional cell)",
            composition,
            vegard,
        )

    out_json = Path(cfg.get("calibration_file", out_dir / "calibration.json"))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s", out_json)
    result["calibration_file"] = str(out_json)
    return result


def resolve_lattice_constant(cfg: dict, *, force_recalibrate: bool = False) -> float:
    """Return cubic conventional a for SQS: explicit cfg['a'], or Vegard from calibration."""
    if not cfg.get("calibrate_lattice", False):
        return float(cfg["a"])

    cal_path = Path(cfg.get("calibration_file", "calibration/calibration.json"))
    if cal_path.is_file() and not force_recalibrate and not cfg.get(
        "recalibrate", False
    ):
        data = json.loads(cal_path.read_text(encoding="utf-8"))
        want = [int(x) for x in cfg.get("supercell", [1, 1, 1])]
        got = [int(x) for x in data.get("supercell", [])]
        if got and got != want:
            log.warning(
                "%s supercell %s != config %s; recalibrating",
                cal_path,
                got,
                want,
            )
        elif "vegard_a_A" in data:
            log.info("reusing %s  vegard_a=%.6f Å", cal_path, data["vegard_a_A"])
            return float(data["vegard_a_A"])
        else:
            composition = dict(cfg["cation_composition"])
            a = vegard_lattice_constant(composition, data["a_by_cation_A"])
            log.info("reusing %s  vegard_a=%.6f Å", cal_path, a)
            return a

    result = run_calibration(cfg)
    if "vegard_a_A" not in result:
        raise RuntimeError("calibration produced no vegard_a_A; set cation_composition")
    return float(result["vegard_a_A"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cubic rocksalt MN lattice calibration (TiN…TaN)"
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if calibration_file exists",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
    )

    cal_path = Path(cfg.get("calibration_file", "calibration/calibration.json"))
    if cal_path.is_file() and not args.force:
        data = json.loads(cal_path.read_text(encoding="utf-8"))
        log.info("%s exists; pass --force to recalibrate", cal_path)
        print(json.dumps(data, indent=2))
        return

    result = run_calibration(cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
