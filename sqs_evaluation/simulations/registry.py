"""Property runners: cell_opt, formation_enthalpy, sro, lld, elastic, eos, dos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ase import Atoms
from ase.io import read, write

from energy_models.base import ENERGY_DRIVEN, GEOMETRY_ONLY, EnergyModel

PropertyFn = Callable[..., dict[str, Any]]


def _load_frames(structures: list[Path] | Path) -> list[Atoms]:
    paths = [structures] if isinstance(structures, Path) else list(structures)
    frames: list[Atoms] = []
    for p in paths:
        got = read(p, index=":")
        if isinstance(got, list):
            frames.extend(got)
        else:
            frames.append(got)
    if not frames:
        raise FileNotFoundError(f"No structures loaded from {paths}")
    return frames


def run_cell_opt(
    frames: list[Atoms],
    model: EnergyModel,
    *,
    out_dir: Path,
    fmax: float = 0.01,
    steps: int = 500,
    relax_cell: bool = True,
    **_kwargs: Any,
) -> dict[str, Any]:
    from ase.filters import FrechetCellFilter
    from ase.optimize import LBFGS

    if model.mode != "ase":
        raise NotImplementedError(f"cell_opt requires ASE backend; got {model.name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    calc = model.build_calculator()
    relaxed: list[Atoms] = []
    rows: list[dict[str, Any]] = []
    for n, atoms0 in enumerate(frames):
        atoms = model.prepare_atoms(atoms0)
        atoms.calc = calc
        opt_atoms = FrechetCellFilter(atoms) if relax_cell else atoms
        LBFGS(opt_atoms, logfile=str(out_dir / f"cell_opt_{n:03d}.log")).run(
            fmax=fmax, steps=steps
        )
        e = float(atoms.get_potential_energy())
        row = {
            "frame": n,
            "formula": atoms.get_chemical_formula(),
            "energy_eV": e,
            "energy_model": model.name,
        }
        atoms.info.update(row)
        write(out_dir / f"sqs_{n:03d}_relaxed.extxyz", atoms)
        relaxed.append(atoms)
        rows.append(row)
    write(out_dir / "all_relaxed.extxyz", relaxed)
    (out_dir / "energies.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return {"property": "cell_opt", "n_frames": len(rows), "out_dir": str(out_dir), "results": rows}


def run_formation_enthalpy(
    frames: list[Atoms],
    model: EnergyModel | None,
    *,
    out_dir: Path,
    elemental_eref: Path | dict | None = None,
    energies: list[float] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """ΔH_f / atom using elemental refs; energies from frames or single-point."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent / "cell_opt"))
    from relax import formation_enthalpy, _eref_per_atom_map  # noqa: WPS433

    if elemental_eref is None:
        raise ValueError("formation_enthalpy requires elemental_eref path or dict")
    if isinstance(elemental_eref, Path):
        eref = json.loads(Path(elemental_eref).read_text(encoding="utf-8"))
    else:
        eref = elemental_eref
    _eref_per_atom_map(eref)  # validate

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    calc = None
    if energies is None:
        if model is None or model.mode != "ase":
            raise ValueError("Need energies= or an ASE energy model for single-points")
        calc = model.build_calculator()

    for n, atoms0 in enumerate(frames):
        atoms = atoms0.copy()
        if energies is not None:
            e = float(energies[n])
        else:
            assert model is not None and calc is not None
            atoms = model.prepare_atoms(atoms)
            atoms.calc = calc
            e = float(atoms.get_potential_energy())
        dhf = formation_enthalpy(atoms, e, eref)
        rows.append(
            {
                "frame": n,
                "formula": atoms.get_chemical_formula(),
                "energy_eV": e,
                "delta_h_f_eV_per_atom": dhf,
                "energy_model": None if model is None else model.name,
            }
        )
    out = out_dir / "formation_enthalpy.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return {"property": "formation_enthalpy", "n_frames": len(rows), "out": str(out), "results": rows}


def run_sro(
    frames: list[Atoms],
    model: EnergyModel | None = None,
    *,
    out_dir: Path,
    cutoff: float = 3.5,
    shell_edges: list[float] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent / "sro"))
    from analyze_sro import pair_correlation, warren_cowley  # noqa: WPS433

    del model  # geometry-only
    shell_edges = list(shell_edges or [0.0, 2.6, 3.5])
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for n, atoms in enumerate(frames):
        payload = {
            "frame": n,
            "formula": atoms.get_chemical_formula(),
            "warren_cowley": warren_cowley(atoms, cutoff, shell_edges),
            "pair_correlation": pair_correlation(atoms, cutoff),
        }
        out = out_dir / f"sro_frame_{n:03d}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(str(out))
    return {"property": "sro", "n_frames": len(written), "files": written}


def run_lld(
    frames: list[Atoms],
    model: EnergyModel | None = None,
    *,
    out_dir: Path,
    ideal: Path | list[Atoms],
    a_ref: float,
    cutoff: float = 3.2,
    bond_pairs: list[str] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent / "lld"))
    from analyze_lld import bond_length_stats, displacements, local_strain  # noqa: WPS433

    del model
    if isinstance(ideal, Path):
        ideal_atoms = read(ideal, index=":")
        if not isinstance(ideal_atoms, list):
            ideal_atoms = [ideal_atoms]
    else:
        ideal_atoms = list(ideal)
    if len(ideal_atoms) == 1 and len(frames) > 1:
        ideal_atoms = ideal_atoms * len(frames)
    if len(ideal_atoms) != len(frames):
        raise ValueError(
            f"lld needs one ideal per frame (or a single shared ideal); "
            f"got {len(ideal_atoms)} ideals for {len(frames)} frames"
        )
    pairs = [tuple(p.split("-", 1)) for p in (bond_pairs or ["Ti-N", "Ti-Ti"])]
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for n, (atoms, ide) in enumerate(zip(frames, ideal_atoms)):
        norms, rmsd = displacements(atoms, ide)
        payload = {
            "frame": n,
            "formula": atoms.get_chemical_formula(),
            "bonds": bond_length_stats(atoms, pairs, cutoff),
            "rmsd_A": rmsd,
            "mean_abs_disp_A": float(norms.mean()),
            "local_strain": local_strain(norms, a_ref),
        }
        out = out_dir / f"lld_frame_{n:03d}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(str(out))
    return {"property": "lld", "n_frames": len(written), "files": written}


def run_elastic(
    frames: list[Atoms],
    model: EnergyModel,
    *,
    out_dir: Path,
    strains: list[float] | None = None,
    fmax: float = 0.01,
    steps: int = 200,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Cubic finite-strain elastic constants (ASE + energy model)."""
    import numpy as np
    from ase.optimize import LBFGS

    if model.mode != "ase":
        raise NotImplementedError(f"elastic requires ASE backend; got {model.name}")

    EV_A3_TO_GPA = 160.21766208
    strains = list(strains or [-0.01, -0.005, 0.005, 0.01])
    out_dir.mkdir(parents=True, exist_ok=True)
    calc = model.build_calculator()

    def apply_strain(atoms, eps_matrix):
        a = atoms.copy()
        a.set_cell(atoms.cell.array @ (np.eye(3) + eps_matrix), scale_atoms=True)
        return a

    def ionic_relax(atoms):
        atoms = model.prepare_atoms(atoms)
        atoms.calc = calc
        LBFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
        # ASE stress is σ = (1/V) ∂E/∂ε (tension > 0). Do not negate.
        stress = np.asarray(atoms.get_stress(voigt=False), dtype=float) * EV_A3_TO_GPA
        return atoms, stress

    def vrh(c11, c12, c44):
        bv = (c11 + 2.0 * c12) / 3.0
        gv = (c11 - c12 + 3.0 * c44) / 5.0
        gr = 5.0 * (c11 - c12) * c44 / (4.0 * c44 + 3.0 * (c11 - c12))
        b, g = bv, 0.5 * (gv + gr)
        e = 9.0 * b * g / (3.0 * b + g)
        nu = (3.0 * b - 2.0 * g) / (2.0 * (3.0 * b + g))
        return b, g, e, nu

    results = []
    for n, base0 in enumerate(frames):
        base = model.prepare_atoms(base0)
        s_uni, e_uni = [], []
        for delta in strains:
            eps = np.zeros((3, 3))
            eps[0, 0] = delta
            atoms, stress = ionic_relax(apply_strain(base, eps))
            write(out_dir / f"sqs_{n:03d}_uni_{delta:+.4f}.extxyz", atoms)
            s_uni.append(stress)
            e_uni.append(delta)
        c11 = float(np.polyfit(e_uni, [s[0, 0] for s in s_uni], 1)[0])
        c12 = float(np.polyfit(e_uni, [s[1, 1] for s in s_uni], 1)[0])
        s_sh, e_sh = [], []
        for gamma in strains:
            eps = np.zeros((3, 3))
            eps[1, 2] = eps[2, 1] = 0.5 * gamma
            atoms, stress = ionic_relax(apply_strain(base, eps))
            write(out_dir / f"sqs_{n:03d}_shear_{gamma:+.4f}.extxyz", atoms)
            s_sh.append(stress)
            e_sh.append(gamma)
        c44 = float(np.polyfit(e_sh, [s[1, 2] for s in s_sh], 1)[0])
        b, g, e, nu = vrh(c11, c12, c44)
        row = {
            "frame": n,
            "formula": base.get_chemical_formula(),
            "C11_GPa": c11,
            "C12_GPa": c12,
            "C44_GPa": c44,
            "B_GPa": b,
            "G_GPa": g,
            "E_GPa": e,
            "nu": nu,
            "energy_model": model.name,
        }
        results.append(row)
    out = out_dir / "elastic.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return {"property": "elastic", "n_frames": len(results), "out": str(out), "results": results}


def run_eos(
    frames: list[Atoms],
    model: EnergyModel,
    *,
    out_dir: Path,
    volumes_scale: list[float] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Equation of state: E(V) scans + equal multi-scheme ASE fits."""
    import sys

    import numpy as np

    _eos_dir = Path(__file__).resolve().parent / "eos"
    if str(_eos_dir) not in sys.path:
        sys.path.insert(0, str(_eos_dir))
    from eos_fit import attach_fits  # noqa: WPS433

    if model.mode != "ase":
        raise NotImplementedError(f"eos requires ASE backend; got {model.name}")

    scales = list(volumes_scale or [0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06])
    out_dir.mkdir(parents=True, exist_ok=True)
    calc = model.build_calculator()
    results = []
    for n, atoms0 in enumerate(frames):
        base = model.prepare_atoms(atoms0)
        volumes, energies = [], []
        cell0 = np.asarray(base.cell.array, dtype=float)
        for s in scales:
            atoms = base.copy()
            atoms.set_cell(cell0 * (s ** (1.0 / 3.0)), scale_atoms=True)
            atoms = model.prepare_atoms(atoms)
            atoms.calc = calc
            e = float(atoms.get_potential_energy())
            volumes.append(float(atoms.get_volume()))
            energies.append(e)
            write(out_dir / f"sqs_{n:03d}_eos_s{s:.3f}.extxyz", atoms)
        row = attach_fits(
            {
                "frame": n,
                "formula": base.get_chemical_formula(),
                "volumes_scale": scales,
                "volumes_A3": volumes,
                "energies_eV": energies,
                "energy_model": model.name,
            }
        )
        results.append(row)
    out = out_dir / "eos.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return {"property": "eos", "n_frames": len(results), "out": str(out), "results": results}


def run_dos(
    frames: list[Atoms],
    model: EnergyModel,
    *,
    out_dir: Path,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Electronic DOS — requires a DFT backend with supports_electronic_dos."""
    if not model.supports_electronic_dos:
        raise NotImplementedError(
            f"dos requires an electronic-structure backend (cp2k_dft/siesta); "
            f"got {model.name} (supports_electronic_dos=False)"
        )
    raise NotImplementedError(
        f"dos runner not implemented yet for {model.name}; scaffold only"
    )


_RUNNERS: dict[str, PropertyFn] = {
    "cell_opt": run_cell_opt,
    "formation_enthalpy": run_formation_enthalpy,
    "sro": run_sro,
    "lld": run_lld,
    "elastic": run_elastic,
    "eos": run_eos,
    "dos": run_dos,
}


def list_properties() -> list[str]:
    return sorted(_RUNNERS)


def run_property(
    name: str,
    frames: list[Atoms],
    model: EnergyModel | None,
    *,
    out_dir: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    key = name.strip().lower()
    if key not in _RUNNERS:
        raise ValueError(f"Unknown property {name!r}; choose from {list_properties()}")
    if key in ENERGY_DRIVEN and model is None:
        raise ValueError(f"property {key} requires an energy model")
    if key in ENERGY_DRIVEN and model is not None and key == "dos":
        pass
    return _RUNNERS[key](frames, model, out_dir=out_dir, **kwargs)
