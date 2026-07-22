"""Energy calculators for MC SQS sampling: GFN2-xTB, UMA, nvalchemi-UMA, MACE."""

from __future__ import annotations

import os
from collections import Counter
from copy import deepcopy
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import LBFGS
from ase.stress import full_3x3_to_voigt_6_stress

SUPPORTED = ("gfn2-xtb", "uma", "nvalchemi-uma", "mace")
# CBMC: GFN2 pool workers, or shared UMA / nvalchemi-UMA calculator
CBMC_SUPPORTED = ("gfn2-xtb", "uma", "nvalchemi-uma")

# ---------------------------------------------------------------------------
# Threading (process env; set before TBLite runs)
# ---------------------------------------------------------------------------
GFN2_OMP_NUM_THREADS = "1"
GFN2_MKL_NUM_THREADS = "1"
GFN2_OMP_STACKSIZE = "4G"

# ---------------------------------------------------------------------------
# Full TBLite ASE knobs (see tblite.ase.TBLite.default_parameters)
# Edit here; build_gfn2_calculator(**overrides) can override any key.
# mixer: "broyden" | "broyden:oda" | "broyden:mesa"  (no DIIS)
# guess: "sad" | "eeq" | "eeqbc"
# mixer_memory: 0 means use max_iterations
# annealing: None | T0_K | (T0_K, hold, cycles)
# spin_polarization: ignored (closed-shell TBLite path)
# ---------------------------------------------------------------------------
GFN2_TBLITE: dict[str, Any] = {
    "method": "GFN2-xTB",
    "charge": 0,  # filled from formal M3+/N3- unless overridden
    "multiplicity": 1,  # always closed-shell for TBLite (spin ignored)
    "accuracy": 1.0,
    "guess": "sad",
    "max_iterations": 1000,
    "mixer": "broyden",
    "mixer_memory": 0,
    "mixer_damping": 0.4,
    "annealing": None,
    "electric_field": None,
    "spin_polarization": None,  # kept off; TBLite path ignores open-shell spin
    "solvation": None,
    "electronic_temperature": 300.0,
    "cache_api": True,
    "verbosity": 1,
    "xtb_config": None,
}

# ---------------------------------------------------------------------------
# UMA / FairChem defaults (edit here or override via config / CLI)
# ---------------------------------------------------------------------------
UMA_DEFAULTS: dict[str, Any] = {
    "model": "/mnt/d/workdir/uma-cache/uma-s-1p2.pt",
    "task": "omat",
    "device": "cuda",
    # nvalchemi UMAWrapper: "default" | "turbo" (torch.compile; fixed composition)
    "inference_settings": "default",
    # batched multi-structure forwards (high VRAM); default off
    "batch": False,
}

# Formal M^{3+} / N^{3-} high-spin d counts (unpaired e^{-} per ion)
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


def _configure_gfn2_threads() -> None:
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
    # Drop spin knobs if callers pass them; TBLite path is closed-shell only
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

    # Force closed shell regardless of formal high-spin estimate
    params["multiplicity"] = 1
    params["spin_polarization"] = None

    if params.get("charge") is None:
        raise ValueError("GFN2 requires charge (via atoms or knobs)")

    return params


# FairChem materials default: spin off (see fairchem DEFAULT_SPIN = 0)
UMA_SPIN_OFF = 0


def set_uma_spin_charge(
    atoms: Atoms,
    *,
    charge: int | None = None,
) -> Atoms:
    """Set FairChem ``atoms.info`` charge; spin is turned off (``spin=0``).

    Formal high-spin multiplicity is recorded as ``formal_spin`` but not used.
    """
    atoms = atoms.copy()
    formal_q, formal_m = formal_charge_and_multiplicity(atoms)
    atoms.info["formal_charge"] = int(formal_q)
    atoms.info["formal_spin"] = int(formal_m)
    atoms.info["charge"] = int(formal_q if charge is None else charge)
    atoms.info["spin"] = UMA_SPIN_OFF
    return atoms


def build_gfn2_calculator(
    atoms: Atoms | None = None,
    *,
    charge: int | None = None,
    **overrides: Any,
) -> Any:
    """TBLite GFN2-xTB using GFN2_TBLITE (+ optional overrides). Spin ignored."""
    _configure_gfn2_threads()
    from tblite.ase import TBLite

    return TBLite(**gfn2_tblite_params(atoms, charge=charge, **overrides))


def build_uma_calculator(
    *,
    model: str | None = None,
    task: str | None = None,
    device: str | None = None,
) -> Any:
    """FairChem UMA ASE calculator from a local .pt path or pretrained name."""
    from fairchem.core import FAIRChemCalculator

    model = model or UMA_DEFAULTS["model"]
    task = task or UMA_DEFAULTS["task"]
    device = device or UMA_DEFAULTS["device"]
    return FAIRChemCalculator.from_model_checkpoint(
        model, task_name=task, device=device
    )


class NvalchemiUMACalculator(Calculator):
    """ASE calculator over NVIDIA ``nvalchemi.models.uma.UMAWrapper``.

    Uses tensor-native UMA inference from
    https://github.com/NVIDIA/nvalchemi-toolkit (built on
    https://github.com/NVIDIA/nvalchemi-toolkit-ops). CBMC calls
    :meth:`energies`; batching is off by default (``batch=False``).
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        wrapper: Any,
        *,
        device: str = "cuda",
        batch: bool = False,
    ) -> None:
        super().__init__()
        self.wrapper = wrapper
        self.device = device
        self.batch = bool(batch)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        from nvalchemi.data import AtomicData, Batch

        if properties is None:
            properties = ["energy"]
        Calculator.calculate(self, atoms, properties, system_changes)
        assert atoms is not None
        atoms = set_uma_spin_charge(atoms)
        data = AtomicData.from_atoms(atoms)
        batch = Batch.from_data_list([data])
        if hasattr(batch, "to"):
            batch = batch.to(self.device)

        # Request only what ASE asked for (energy-only SPE avoids force autograd)
        want = set(properties)
        active = {"energy"}
        if "forces" in want:
            active.add("forces")
        if "stress" in want or any(k.startswith("stress") for k in want):
            active.add("stress")
        prev = self.wrapper.model_config.active_outputs
        self.wrapper.model_config.active_outputs = frozenset(active)
        try:
            out = self.wrapper(batch)
        finally:
            self.wrapper.model_config.active_outputs = prev

        energy = float(out["energy"].detach().cpu().reshape(-1)[0].item())
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        if "forces" in out:
            self.results["forces"] = (
                out["forces"].detach().cpu().numpy().astype(float)
            )
        if "stress" in out:
            stress = out["stress"].detach().cpu().numpy().reshape(3, 3)
            self.results["stress"] = full_3x3_to_voigt_6_stress(stress)

    def energies(self, atoms_list: list[Atoms]) -> list[float]:
        """Single-points; batched only if ``self.batch`` is True."""
        if not atoms_list:
            return []
        if not self.batch:
            out: list[float] = []
            for atoms in atoms_list:
                a = atoms.copy()
                a.calc = self
                out.append(float(a.get_potential_energy()))
            return out

        from nvalchemi.data import AtomicData, Batch

        datas = [AtomicData.from_atoms(set_uma_spin_charge(a)) for a in atoms_list]
        batch = Batch.from_data_list(datas)
        if hasattr(batch, "to"):
            batch = batch.to(self.device)
        prev = self.wrapper.model_config.active_outputs
        self.wrapper.model_config.active_outputs = frozenset({"energy"})
        try:
            pred = self.wrapper(batch)
        finally:
            self.wrapper.model_config.active_outputs = prev
        return [
            float(x)
            for x in pred["energy"].detach().cpu().reshape(-1).tolist()
        ]


def build_nvalchemi_uma_calculator(
    *,
    model: str | None = None,
    task: str | None = None,
    device: str | None = None,
    inference_settings: str | None = None,
    batch: bool | None = None,
) -> NvalchemiUMACalculator:
    """NVIDIA nvalchemi ``UMAWrapper`` ASE calculator (batching off by default)."""
    from nvalchemi.models.uma import UMAWrapper

    model = model or UMA_DEFAULTS["model"]
    task = task or UMA_DEFAULTS["task"]
    device = device or UMA_DEFAULTS["device"]
    inference_settings = inference_settings or UMA_DEFAULTS["inference_settings"]
    if batch is None:
        batch = bool(UMA_DEFAULTS["batch"])
    wrapper = UMAWrapper.from_checkpoint(
        model,
        task_name=task,
        device=device,
        inference_settings=inference_settings,
    )
    return NvalchemiUMACalculator(wrapper, device=device, batch=batch)


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
    symbols, positions, cell, pbc, tags, params = payload
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=pbc)
    atoms.set_tags(tags)
    _configure_gfn2_threads()
    from tblite.ase import TBLite

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


def evaluate_with_calculator(
    atoms_list: list[Atoms],
    calc: Any,
    *,
    method: str = "uma",
) -> list[float]:
    """Single-points with a shared calculator (UMA / nvalchemi-UMA / MACE)."""
    key = method.strip().lower()
    if key == "nvalchemi-uma":
        if not isinstance(calc, NvalchemiUMACalculator):
            raise TypeError("nvalchemi-uma requires NvalchemiUMACalculator")
        return calc.energies(atoms_list)

    energies: list[float] = []
    for atoms in atoms_list:
        a = set_uma_spin_charge(atoms) if key == "uma" else atoms.copy()
        a.calc = calc
        energies.append(float(a.get_potential_energy()))
    return energies


def evaluate_energies(
    atoms_list: list[Atoms],
    method: str,
    *,
    pool: Pool | None = None,
    calc: Any | None = None,
) -> list[float]:
    """Dispatch energy evaluations for CBMC / smoke tests.

    gfn2-xtb: multiprocess Pool (requires ``pool``); closed-shell (spin ignored).
    uma: shared FairChem ASE calculator (``spin=0``).
    nvalchemi-uma: NVIDIA UMAWrapper (``spin=0``; batching off unless enabled).
    mace: shared ASE calculator (no spin tags).
    """
    key = method.strip().lower()
    if key == "gfn2-xtb":
        if pool is None:
            raise ValueError("gfn2-xtb parallel eval requires a multiprocessing Pool")
        return evaluate_gfn2_parallel(atoms_list, pool)
    if key in ("uma", "nvalchemi-uma", "mace"):
        if calc is None:
            raise ValueError(f"{key} eval requires a shared ASE calculator")
        return evaluate_with_calculator(atoms_list, calc, method=key)
    raise ValueError(f"Unknown energy method {method!r}; choose from {SUPPORTED}")


def build_calculator(
    method: str,
    *,
    atoms: Atoms | None = None,
    device: str = "cuda",
    uma_model: str = "/mnt/d/workdir/uma-cache/uma-s-1p2.pt",
    uma_task: str = "omat",
    inference_settings: str = "default",
    nvalchemi_batch: bool = False,
    mace_model: str | None = None,
    **gfn2_overrides: Any,
) -> Any:
    """Return an ASE calculator for the requested energy method."""
    key = method.strip().lower()
    if key not in SUPPORTED:
        raise ValueError(f"Unknown energy method {method!r}; choose from {SUPPORTED}")

    if key == "gfn2-xtb":
        gfn2_overrides.pop("multiplicity", None)
        return build_gfn2_calculator(atoms=atoms, **gfn2_overrides)

    if key == "uma":
        return build_uma_calculator(model=uma_model, task=uma_task, device=device)

    if key == "nvalchemi-uma":
        return build_nvalchemi_uma_calculator(
            model=uma_model,
            task=uma_task,
            device=device,
            inference_settings=inference_settings,
            batch=nvalchemi_batch,
        )

    if not mace_model:
        raise ValueError("MACE requires mace_model path (--mace-model or config mace_model)")
    from mace.calculators import MACECalculator

    return MACECalculator(model_paths=str(Path(mace_model)), device=device)


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
        LBFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    return float(atoms.get_potential_energy())
