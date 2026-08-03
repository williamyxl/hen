"""Registry: map config ``energy.model`` → EnergyModel instance."""

from __future__ import annotations

from typing import Any

from .base import EnergyModel
from .cp2k_dft import CP2KDFTModel
from .gfn2_cp2k import GFN2CP2KModel
from .gfn2_tblite import GFN2TBLiteModel
from .mace import MACEModel
from .siesta import SiestaModel
from .uma import UMAModel

# Default production backend for HEN evaluation
DEFAULT_ENERGY_MODEL = "uma"

_REGISTRY = {
    "uma": UMAModel,
    "mace": MACEModel,
    "gfn2_tblite": GFN2TBLiteModel,
    "gfn2_cp2k": GFN2CP2KModel,
    "cp2k_dft": CP2KDFTModel,
    "siesta": SiestaModel,
}


def list_energy_models() -> list[str]:
    return sorted(_REGISTRY)


def build_energy_model(cfg: dict[str, Any] | None = None) -> EnergyModel:
    """Build from a workflow config ``energy:`` block (default: UMA)."""
    cfg = dict(cfg or {})
    name = str(cfg.get("model", DEFAULT_ENERGY_MODEL)).strip().lower()
    # aliases
    aliases = {
        "fairchem": "uma",
        "fairchem_uma": "uma",
        "gfn2": "gfn2_tblite",
        "gfn2-xtb": "gfn2_tblite",
        "tblite": "gfn2_tblite",
        "cp2k": "cp2k_dft",
    }
    name = aliases.get(name, name)
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown energy model {name!r}; choose from {list_energy_models()}"
        )

    cls = _REGISTRY[name]
    if name == "uma":
        return cls(
            model=cfg.get("uma_model") or cfg.get("model_path"),
            device=str(cfg.get("device", "xpu")),
            dtype=str(cfg.get("dtype", "float64")),
            task=str(cfg.get("uma_task", "omat")),
            workers=int(cfg.get("uma_workers", 1)),
        )
    if name == "mace":
        path = cfg.get("mace_model") or cfg.get("model_path")
        if not path:
            raise ValueError("energy.mace_model (or model_path) is required for mace")
        return cls(model=str(path), device=str(cfg.get("device", "cpu")))
    if name == "gfn2_tblite":
        return cls()
    if name == "gfn2_cp2k":
        return cls(project=str(cfg.get("project", "fxpu_gfn2_cp2k")))
    if name == "cp2k_dft":
        return cls(
            xc=str(cfg.get("xc", "PBE")),
            basis=str(cfg.get("basis", "DZVP-MOLOPT-SR-GTH")),
            dispersion=str(cfg.get("dispersion", "D3")),
            cutoff_ry=float(cfg.get("cutoff_ry", 600.0)),
            project=str(cfg.get("project", "fxpu_cp2k")),
        )
    if name == "siesta":
        return cls(
            xc=str(cfg.get("xc", "PBE")),
            mesh_cutoff_ry=float(cfg.get("mesh_cutoff_ry", 200.0)),
            project=str(cfg.get("project", "fxpu_siesta")),
        )
    raise RuntimeError(f"unhandled model {name}")
