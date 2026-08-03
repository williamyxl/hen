"""GFN2-xTB via CP2K (distinct from TBLite) — file-IO backend (scaffold)."""

from __future__ import annotations

from typing import Any

from ase.calculators.calculator import Calculator

from .base import EnergyModel


class GFN2CP2KModel(EnergyModel):
    """CP2K ``METHOD XTBn`` / GFN2-xTB driver (input generation TBD)."""

    name = "gfn2_cp2k"
    mode = "fileio"

    def __init__(self, *, project: str = "fxpu_gfn2_cp2k") -> None:
        self.project = project

    def build_calculator(self) -> Calculator:
        raise NotImplementedError(
            "gfn2_cp2k backend is scaffolded only; wire CP2K GFN2 input/output next"
        )

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({"engine": "cp2k", "method": "GFN2-xTB", "status": "scaffold"})
        return d
