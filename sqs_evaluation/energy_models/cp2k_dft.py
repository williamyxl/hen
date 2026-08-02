"""CP2K DFT energy model (XC + basis + dispersion) — file-IO backend (scaffold)."""

from __future__ import annotations

from typing import Any

from ase.calculators.calculator import Calculator

from .base import EnergyModel


class CP2KDFTModel(EnergyModel):
    """PBE/DFT in CP2K with configurable XC, basis, and dispersion.

    Input writing lives in ``energy_models/cp2k/`` (same backend family).
    Full ASE FileIOCalculator wiring is still TODO.
    """

    name = "cp2k_dft"
    mode = "fileio"

    def __init__(
        self,
        *,
        xc: str = "PBE",
        basis: str = "DZVP-MOLOPT-SR-GTH",
        dispersion: str = "D3",
        cutoff_ry: float = 600.0,
        project: str = "hen_cp2k",
    ) -> None:
        self.xc = xc
        self.basis = basis
        self.dispersion = dispersion
        self.cutoff_ry = float(cutoff_ry)
        self.project = project

    def build_calculator(self) -> Calculator:
        raise NotImplementedError(
            "cp2k_dft ASE calculator not wired yet; use energy_models/cp2k/"
            "make_cp2k_input.py for inputs, then extend this backend"
        )

    @property
    def supports_electronic_dos(self) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update(
            {
                "xc": self.xc,
                "basis": self.basis,
                "dispersion": self.dispersion,
                "cutoff_ry": self.cutoff_ry,
                "project": self.project,
                "status": "scaffold",
            }
        )
        return d
