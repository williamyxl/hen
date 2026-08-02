"""FairChem UMA energy model (default production backend)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator

from .base import EnergyModel

_HEN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HEN / "sqs_sampling"))
from energy import UMA_DEFAULTS, UMA_SPIN_OFF, uma_predict_unit  # noqa: E402


class UMAModel(EnergyModel):
    name = "uma"
    mode = "ase"

    def __init__(
        self,
        *,
        model: str | None = None,
        device: str = "xpu",
        dtype: str = "float64",
        task: str = "omat",
        workers: int = 1,
    ) -> None:
        self.model = model or UMA_DEFAULTS["model"]
        self.device = device
        self.dtype = dtype
        self.task = task
        self.workers = int(workers)

    def build_calculator(self) -> Calculator:
        from fairchem.core import FAIRChemCalculator

        return FAIRChemCalculator(
            uma_predict_unit(
                model=self.model,
                device=self.device,
                dtype=self.dtype,
                workers=self.workers,
            ),
            task_name=self.task,
        )

    def prepare_atoms(self, atoms: Atoms) -> Atoms:
        a = atoms.copy()
        # Alloys: keep existing charge if set; else neutral for generic cells
        a.info.setdefault("charge", int(a.info.get("charge", 0)))
        a.info["spin"] = UMA_SPIN_OFF
        return a

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update(
            {
                "model": self.model,
                "device": self.device,
                "dtype": self.dtype,
                "task": self.task,
                "workers": self.workers,
            }
        )
        return d
