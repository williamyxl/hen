"""Energy-model interface for property simulations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from ase import Atoms
from ase.calculators.calculator import Calculator

# Properties that need an energy/force engine
ENERGY_DRIVEN = frozenset(
    {
        "cell_opt",
        "formation_enthalpy",
        "elastic",
        "eos",
        "dos",
    }
)
# Geometry-only (no calculator)
GEOMETRY_ONLY = frozenset({"sro", "lld"})


class EnergyModel(ABC):
    """Backend that supplies energies (and usually forces/stress) for properties."""

    name: str
    # ase = attach ASE Calculator; fileio = write inputs / parse outputs (DFT codes)
    mode: Literal["ase", "fileio"] = "ase"

    @abstractmethod
    def build_calculator(self) -> Calculator:
        """Return an ASE calculator (required for mode='ase')."""

    def prepare_atoms(self, atoms: Atoms) -> Atoms:
        """Attach backend-specific atoms.info (charge/spin, etc.)."""
        return atoms.copy()

    @property
    def supports_forces(self) -> bool:
        return True

    @property
    def supports_stress(self) -> bool:
        return True

    @property
    def supports_electronic_dos(self) -> bool:
        """True only for electronic-structure DFT backends."""
        return False

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "supports_forces": self.supports_forces,
            "supports_stress": self.supports_stress,
            "supports_electronic_dos": self.supports_electronic_dos,
        }
