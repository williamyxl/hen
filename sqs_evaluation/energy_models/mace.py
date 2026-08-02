"""MACE MLIP energy model (optional; not the HEN default)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ase.calculators.calculator import Calculator

from .base import EnergyModel


class MACEModel(EnergyModel):
    name = "mace"
    mode = "ase"

    def __init__(self, *, model: str, device: str = "cpu") -> None:
        if not model:
            raise ValueError("MACEModel requires model= path")
        self.model = str(Path(model))
        self.device = device

    def build_calculator(self) -> Calculator:
        from mace.calculators import MACECalculator

        return MACECalculator(model_paths=self.model, device=self.device)

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({"model": self.model, "device": self.device})
        return d
