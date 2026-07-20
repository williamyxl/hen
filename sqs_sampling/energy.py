"""Energy calculators for MC SQS sampling: GFN2-xTB, UMA MLIP, MACE MLIP."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.optimize import FIRE

SUPPORTED = ("gfn2-xtb", "uma", "mace")


def build_calculator(
    method: str,
    *,
    device: str = "cpu",
    uma_model: str = "uma-s-1p2",
    uma_task: str = "omat",
    mace_model: str | None = None,
) -> Any:
    """Return an ASE calculator for the requested energy method.

    Imports the selected backend only. Missing packages raise ImportError.
    """
    key = method.strip().lower()
    if key not in SUPPORTED:
        raise ValueError(f"Unknown energy method {method!r}; choose from {SUPPORTED}")

    if key == "gfn2-xtb":
        from tblite.ase import TBLite

        return TBLite(method="GFN2-xTB")

    if key == "uma":
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        predictor = pretrained_mlip.get_predict_unit(uma_model, device=device)
        return FAIRChemCalculator(predictor, task_name=uma_task)

    if mace_model is None:
        raise ValueError("MACE requires mace_model path (--mace-model or config mace_model)")
    model_path = Path(mace_model)
    if not model_path.is_file():
        raise FileNotFoundError(f"MACE model not found: {model_path.resolve()}")

    from mace.calculators import MACECalculator

    return MACECalculator(model_paths=str(model_path), device=device)


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
        if steps < 1:
            raise ValueError(f"relax_steps must be >= 1, got {steps}")
        FIRE(atoms, logfile=None).run(fmax=fmax, steps=steps)
        max_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
        if max_force > fmax:
            raise RuntimeError(
                f"Ionic relax failed to reach fmax={fmax} eV/Å "
                f"(max force={max_force:.4f} eV/Å, steps={steps})"
            )
    energy = atoms.get_potential_energy()
    if energy is None or not np.isfinite(energy):
        raise RuntimeError(f"Calculator returned non-finite energy: {energy!r}")
    return float(energy)
