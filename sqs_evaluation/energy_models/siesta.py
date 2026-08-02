"""Siesta DFT energy model — file-IO backend (scaffold)."""

from __future__ import annotations

from typing import Any

from ase.calculators.calculator import Calculator

from .base import EnergyModel


class SiestaModel(EnergyModel):
    """Siesta DFT (XC / basis / k-grid to be configured when wired)."""

    name = "siesta"
    mode = "fileio"

    def __init__(
        self,
        *,
        xc: str = "PBE",
        mesh_cutoff_ry: float = 200.0,
        project: str = "hen_siesta",
    ) -> None:
        self.xc = xc
        self.mesh_cutoff_ry = float(mesh_cutoff_ry)
        self.project = project

    def build_calculator(self) -> Calculator:
        raise NotImplementedError(
            "siesta backend is scaffolded only; wire ASE Siesta or FileIO next"
        )

    @property
    def supports_electronic_dos(self) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update(
            {
                "xc": self.xc,
                "mesh_cutoff_ry": self.mesh_cutoff_ry,
                "status": "scaffold",
            }
        )
        return d
