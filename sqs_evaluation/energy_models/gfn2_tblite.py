"""GFN2-xTB via TBLite ASE (optional cheap path; distinct from CP2K GFN2)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator

from .base import EnergyModel

_HEN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_HEN / "sqs_sampling"))
from energy import configure_gfn2_threads, gfn2_tblite_params  # noqa: E402


class GFN2TBLiteModel(EnergyModel):
    name = "gfn2_tblite"
    mode = "ase"

    def build_calculator(self) -> Calculator:
        from tblite.ase import TBLite

        configure_gfn2_threads()
        return TBLite(**gfn2_tblite_params())

    def prepare_atoms(self, atoms: Atoms) -> Atoms:
        return atoms.copy()

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["engine"] = "tblite"
        return d
