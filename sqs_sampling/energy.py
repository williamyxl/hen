"""Energy helpers for MC SQS: GFN2-xTB (TBLite) and FairChem UMA (Intel XPU)."""

from __future__ import annotations

import os
from collections import Counter
from copy import deepcopy
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms

SUPPORTED = ("gfn2-xtb", "uma", "mace")
CBMC_SUPPORTED = ("gfn2-xtb", "uma")

GFN2_OMP_NUM_THREADS = "1"
GFN2_MKL_NUM_THREADS = "1"
GFN2_OMP_STACKSIZE = "4G"

# TBLite ASE knobs (see tblite.ase.TBLite.default_parameters)
GFN2_TBLITE: dict[str, Any] = {
    "method": "GFN2-xTB",
    "charge": 0,
    "multiplicity": 1,
    "accuracy": 1.0,
    "guess": "sad",
    "max_iterations": 1000,
    "mixer": "broyden",
    "mixer_memory": 0,
    "mixer_damping": 0.4,
    "annealing": None,
    "electric_field": None,
    "spin_polarization": None,
    "solvation": None,
    "electronic_temperature": 300.0,
    "cache_api": True,
    "verbosity": 1,
    "xtb_config": None,
}

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_UMA_CHECKPOINTS = frozenset({"uma-s-1p2.pt"})
UMA_DEFAULTS: dict[str, Any] = {
    "model": str(_PROJECT_ROOT / "uma-cache" / "uma-s-1p2.pt"),
    "task": "omat",
    "device": "xpu",
    "dtype": "float64",
}

UMA_SPIN_OFF = 0

UNPAIRED_M3 = {
    "Ti": 1,
    "Zr": 1,
    "Hf": 1,
    "V": 2,
    "Nb": 2,
    "Ta": 2,
    "Cr": 3,
    "Mo": 3,
    "W": 3,
    "Al": 0,
    "N": 0,
}


def resolve_torch_dtype(dtype: str | Any | None = None) -> Any:
    """Map config/CLI dtype name to torch.dtype; FXPU FairChem path requires float64."""
    import torch

    if dtype is None:
        dtype = UMA_DEFAULTS["dtype"]
    if isinstance(dtype, torch.dtype):
        resolved = dtype
    else:
        key = str(dtype).strip().lower()
        mapping = {
            "float64": torch.float64,
            "fp64": torch.float64,
            "double": torch.float64,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if key not in mapping:
            raise ValueError(f"Unsupported dtype {dtype!r}")
        resolved = mapping[key]
    if resolved != torch.float64:
        raise ValueError(
            f"FXPU FairChem inference requires float64/fp64; got {dtype!r}"
        )
    return resolved


def assert_allowed_uma_checkpoint(model: str | Path) -> Path:
    """Only allow uma-s-1p2.pt for this deployment."""
    path = Path(model)
    if path.suffix == ".pt" and path.name not in ALLOWED_UMA_CHECKPOINTS:
        raise ValueError(
            f"Checkpoint {path.name!r} is not allowed; "
            f"use one of {sorted(ALLOWED_UMA_CHECKPOINTS)}"
        )
    return path


def enforce_module_float64(module: Any) -> None:
    """Cast parameters/buffers to FP64 and verify."""
    import torch

    if hasattr(module, "to"):
        module.to(dtype=torch.float64)
    bad: list[tuple[str, str]] = []
    named_parameters = getattr(module, "named_parameters", None)
    if callable(named_parameters):
        for name, param in named_parameters():
            if param.dtype != torch.float64:
                bad.append((name, str(param.dtype)))
    named_buffers = getattr(module, "named_buffers", None)
    if callable(named_buffers):
        for name, buf in named_buffers():
            if torch.is_floating_point(buf) and buf.dtype != torch.float64:
                bad.append((name, str(buf.dtype)))
    if bad:
        raise RuntimeError(f"Non-FP64 tensors after cast: {bad[:8]}")


def configure_gfn2_threads() -> None:
    os.environ["OMP_NUM_THREADS"] = GFN2_OMP_NUM_THREADS
    os.environ["MKL_NUM_THREADS"] = GFN2_MKL_NUM_THREADS
    os.environ["OMP_STACKSIZE"] = GFN2_OMP_STACKSIZE


def formal_charge_and_multiplicity(atoms: Atoms) -> tuple[int, int]:
    """Net charge and spin multiplicity for formal M^{3+} / N^{3-} rocksalt nitrides."""
    counts = Counter(atoms.get_chemical_symbols())
    n_n = int(counts.get("N", 0))
    metals = {s: n for s, n in counts.items() if s != "N"}
    charge = 3 * sum(metals.values()) - 3 * n_n

    unpaired = 0
    for sym, n in metals.items():
        if sym not in UNPAIRED_M3:
            raise KeyError(
                f"No M3+ unpaired-electron count for {sym}; extend UNPAIRED_M3"
            )
        unpaired += UNPAIRED_M3[sym] * n
    return int(charge), int(unpaired + 1)


def gfn2_tblite_params(
    atoms: Atoms | None = None,
    *,
    charge: int | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Merge GFN2_TBLITE defaults and formal charge. Spin multiplicity is ignored."""
    overrides.pop("multiplicity", None)
    overrides.pop("spin_polarization", None)

    params = deepcopy(GFN2_TBLITE)
    params.update(overrides)

    if atoms is not None:
        formal_q, _ = formal_charge_and_multiplicity(atoms)
        if charge is None:
            charge = formal_q
    if charge is not None:
        params["charge"] = charge

    params["multiplicity"] = 1
    params["spin_polarization"] = None

    if params.get("charge") is None:
        raise ValueError("GFN2 requires charge (via atoms or knobs)")

    return params


def set_uma_spin_charge(
    atoms: Atoms,
    *,
    charge: int | None = None,
) -> Atoms:
    """Set FairChem ``atoms.info`` charge; spin is turned off (``spin=0``)."""
    atoms = atoms.copy()
    formal_q, formal_m = formal_charge_and_multiplicity(atoms)
    atoms.info["formal_charge"] = int(formal_q)
    atoms.info["formal_spin"] = int(formal_m)
    atoms.info["charge"] = int(formal_q if charge is None else charge)
    atoms.info["spin"] = UMA_SPIN_OFF
    return atoms


def assert_single_tile_xpu() -> None:
    """Require FLAT + one visible tile for single-tile UMA on XPU."""
    import torch

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("torch.xpu is not available for single-tile UMA")
    n = int(torch.xpu.device_count())
    if n != 1:
        raise RuntimeError(
            f"Single-tile UMA expects 1 visible XPU (ZE_AFFINITY_MASK); got {n}. "
            "Set ZE_FLAT_DEVICE_HIERARCHY=FLAT and ZE_AFFINITY_MASK=<tile>."
        )


def uma_predict_unit(
    *,
    model: str | Path | None = None,
    device: str | None = None,
    dtype: str | Any | None = None,
    workers: int = 1,
) -> Any:
    """Return a FairChem ``MLIPPredictUnit`` (W=1) or ``ParallelMLIPPredictUnit`` (W≥2).

    Callers wrap with ``FAIRChemCalculator(unit, task_name=...)``.
    """
    from dataclasses import replace

    from fairchem.core.units.mlip_unit.api.inference import guess_inference_settings
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit

    model_path = assert_allowed_uma_checkpoint(model or UMA_DEFAULTS["model"])
    if not model_path.is_file():
        raise FileNotFoundError(f"UMA checkpoint not found: {model_path}")
    device = device or UMA_DEFAULTS["device"]
    device_key = str(device).strip().lower()
    if device_key.startswith("cuda"):
        raise ValueError(
            "FXPU path is Intel XPU–only; refuse device="
            f"{device!r}. Use device='xpu' (or 'cpu' for debug)."
        )
    torch_dtype = resolve_torch_dtype(dtype)
    n_workers = int(workers)
    if n_workers < 1:
        raise ValueError(f"workers must be >= 1, got {n_workers}")

    settings = guess_inference_settings("default")
    settings = replace(
        settings,
        base_precision_dtype=torch_dtype,
        tf32=False,
        compile=False,
    )

    if n_workers == 1:
        if device_key.startswith("xpu"):
            from fairchem_xpu_parallel import patch_fairchem_xpu_device

            assert_single_tile_xpu()
            patch_fairchem_xpu_device()
        unit = MLIPPredictUnit(
            str(model_path),
            device=device,
            inference_settings=settings,
        )
    else:
        from fairchem.core.units.mlip_unit.predict import ParallelMLIPPredictUnit
        from fairchem_xpu_parallel import patch_fairchem_xpu_parallel

        if device_key.startswith("xpu"):
            patch_fairchem_xpu_parallel()
        unit = ParallelMLIPPredictUnit(
            str(model_path),
            device=device,
            inference_settings=settings,
            num_workers=n_workers,
            num_workers_per_node=max(n_workers, 12),
        )

    for attr in ("model", "module", "_module"):
        mod = getattr(unit, attr, None)
        if mod is not None:
            enforce_module_float64(mod)
            break
    return unit


def atoms_to_payload(atoms: Atoms) -> tuple:
    """Picklable snapshot for multiprocess GFN2 workers."""
    params = gfn2_tblite_params(atoms)
    return (
        list(atoms.get_chemical_symbols()),
        np.asarray(atoms.get_positions(), dtype=float),
        np.asarray(atoms.cell.array, dtype=float),
        np.asarray(atoms.pbc, dtype=bool),
        np.asarray(atoms.get_tags(), dtype=int),
        params,
    )


def gfn2_energy_payload(payload: tuple) -> float:
    """Worker entry: one TBLite GFN2-xTB single-point (1 OMP thread)."""
    from tblite.ase import TBLite

    symbols, positions, cell, pbc, tags, params = payload
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
    atoms.set_tags(tags)
    configure_gfn2_threads()
    atoms.calc = TBLite(**params)
    return float(atoms.get_potential_energy())


def evaluate_gfn2_parallel(atoms_list: list[Atoms], pool: Pool) -> list[float]:
    """Concurrent GFN2 energies via Pool.map_async."""
    if not atoms_list:
        return []
    async_result = pool.map_async(
        gfn2_energy_payload, [atoms_to_payload(a) for a in atoms_list]
    )
    return list(async_result.get())


def evaluate_energies(
    atoms_list: list[Atoms],
    method: str,
    *,
    pool: Pool | None = None,
    calc: Any | None = None,
) -> list[float]:
    """Dispatch energy evaluations for CBMC / smoke tests."""
    key = method.strip().lower()
    if key == "gfn2-xtb":
        if pool is None:
            raise ValueError("gfn2-xtb parallel eval requires a multiprocessing Pool")
        return evaluate_gfn2_parallel(atoms_list, pool)
    if key == "uma":
        if calc is None:
            raise ValueError("uma eval requires a shared FAIRChemCalculator")
        energies: list[float] = []
        for atoms in atoms_list:
            a = set_uma_spin_charge(atoms)
            a.calc = calc
            energies.append(float(a.get_potential_energy()))
        return energies
    raise ValueError(f"Unknown energy method {method!r}; choose from {CBMC_SUPPORTED}")
