"""Energy calculators for MC SQS sampling: GFN2-xTB, UMA MLIP, MACE MLIP."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.optimize import LBFGS

SUPPORTED = ("gfn2-xtb", "uma", "mace")

# Required for all GFN2-xTB / TBLite runs
GFN2_MAX_ITERATIONS = 2000
GFN2_OMP_NUM_THREADS = "1"
GFN2_MKL_NUM_THREADS = "1"
GFN2_OMP_STACKSIZE = "8G"


def _configure_gfn2_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = GFN2_OMP_NUM_THREADS
    os.environ["MKL_NUM_THREADS"] = GFN2_MKL_NUM_THREADS
    os.environ["OMP_STACKSIZE"] = GFN2_OMP_STACKSIZE


def build_calculator(
    method: str,
    *,
    device: str = "cpu",
    uma_model: str = "uma-s-1p2",
    uma_task: str = "omat",
    mace_model: str | None = None,
) -> Any:
    """Return an ASE calculator for the requested energy method."""
    key = method.strip().lower()
    if key not in SUPPORTED:
        raise ValueError(f"Unknown energy method {method!r}; choose from {SUPPORTED}")

    if key == "gfn2-xtb":
        _configure_gfn2_threads()
        from tblite.ase import TBLite

        return TBLite(method="GFN2-xTB", max_iterations=GFN2_MAX_ITERATIONS)

    if key == "uma":
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        predictor = pretrained_mlip.get_predict_unit(uma_model, device=device)
        return FAIRChemCalculator(predictor, task_name=uma_task)

    if not mace_model:
        raise ValueError("MACE requires mace_model path (--mace-model or config mace_model)")
    from mace.calculators import MACECalculator

    return MACECalculator(model_paths=str(Path(mace_model)), device=device)


def evaluate_energy(
    atoms: Atoms,
    calc: Any,
    *,
    relax: bool = False,
    fmax: float = 0.05,
    steps: int = 20,
) -> float:
    """Attach calculator, optionally relax ions, return potential energy (eV)."""
    atoms = atoms.copy()
    atoms.calc = calc
    if relax:
        LBFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    return float(atoms.get_potential_energy())
