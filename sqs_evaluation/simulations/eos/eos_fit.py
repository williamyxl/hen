#!/usr/bin/env python3
"""Multi-scheme E–V equation-of-state fits (ASE), all schemes treated equally.

ASE schemes (same set as ``ase.eos.eos_names``)::

  sj, taylor, murnaghan, birch, birchmurnaghan, pouriertarantola,
  vinet, anton-schmidt, p3

Materials Project documents overlapping forms (Birch, Murnaghan, Vinet,
Poirier–Tarantola, …); see
https://docs.materialsproject.org/methodology/materials-methodology/equations-of-state

Also usable offline on an existing ``result.json`` (re-fit only, no XPU)::

  python eos_fit.py --from-result path/to/result.json [--write]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from ase.eos import EquationOfState, eos_names

EV_A3_TO_GPA = 160.21766208

# Equal treatment: full ASE set, stable order.
EOS_SCHEMES: tuple[str, ...] = tuple(eos_names)


def _rms(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(d * d)))


def _predict_energies(eos: EquationOfState, volumes: np.ndarray) -> np.ndarray | None:
    """Best-effort E(V) from a fitted ASE ``EquationOfState``."""
    if hasattr(eos, "func") and hasattr(eos, "eos_parameters"):
        return np.asarray(eos.func(volumes, *eos.eos_parameters), dtype=float)
    # sj: cubic poly in V^{-1/3} stored as ``fit0``
    if eos.eos_string == "sj" and getattr(eos, "fit0", None) is not None:
        return np.asarray(eos.fit0(volumes ** (-1.0 / 3.0)), dtype=float)
    return None


def fit_one_eos(
    volumes: list[float] | np.ndarray,
    energies: list[float] | np.ndarray,
    scheme: str,
) -> dict[str, Any]:
    """Fit a single ASE EOS scheme. Always returns a dict with ``ok`` flag."""
    v = np.asarray(volumes, dtype=float)
    e = np.asarray(energies, dtype=float)
    out: dict[str, Any] = {
        "scheme": scheme,
        "ok": False,
        "V0_A3": None,
        "E0_eV": None,
        "B_eV_A3": None,
        "B_GPa": None,
        "rms_eV": None,
        "error": None,
    }
    try:
        eos = EquationOfState(v, e, eos=scheme)
        v0, e0, B = eos.fit()
        e_pred = _predict_energies(eos, v)
        rms = _rms(e, e_pred) if e_pred is not None else None
        out.update(
            {
                "ok": True,
                "V0_A3": float(v0),
                "E0_eV": float(e0),
                "B_eV_A3": float(B),
                "B_GPa": float(B * EV_A3_TO_GPA),
                "rms_eV": rms,
                "error": None,
            }
        )
        if not (
            math.isfinite(out["V0_A3"])
            and math.isfinite(out["E0_eV"])
            and math.isfinite(out["B_GPa"])
            and out["V0_A3"] > 0.0
            and out["B_GPa"] > 0.0
        ):
            out["ok"] = False
            out["error"] = "non-finite or non-positive V0/B"
    except Exception as exc:  # noqa: BLE001 — collect per-scheme failures
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def fit_all_eos(
    volumes: list[float] | np.ndarray,
    energies: list[float] | np.ndarray,
    schemes: tuple[str, ...] | list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fit every scheme; returns ``{scheme: fit_dict}``."""
    names = tuple(schemes) if schemes is not None else EOS_SCHEMES
    return {name: fit_one_eos(volumes, energies, name) for name in names}


def fits_all_ok(fits: dict[str, dict[str, Any]]) -> bool:
    if not fits:
        return False
    return all(bool(f.get("ok")) for f in fits.values())


def attach_fits(row: dict[str, Any], schemes: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Add/replace equal-scheme fits on a result row that has volumes + energies.

    Drops privileged single-scheme top-level V0/E0/B keys if present.
    """
    volumes = row["volumes_A3"]
    energies = row["energies_eV"]
    fits = fit_all_eos(volumes, energies, schemes=schemes)
    row = dict(row)
    for k in ("V0_A3", "E0_eV", "B_eV_A3", "B_GPa"):
        row.pop(k, None)
    row["eos_schemes"] = list(fits.keys())
    row["fits"] = fits
    row["n_fits_ok"] = sum(1 for f in fits.values() if f.get("ok"))
    row["n_fits"] = len(fits)
    return row


def format_fits_table(fits: dict[str, dict[str, Any]]) -> str:
    lines = [
        f"{'scheme':20s} {'ok':3s} {'V0_A3':>12s} {'E0_eV':>14s} {'B_GPa':>10s} {'rms_eV':>10s}"
    ]
    for name, f in fits.items():
        if f.get("ok"):
            rms = f.get("rms_eV")
            rms_s = f"{rms:10.4e}" if isinstance(rms, (int, float)) else f"{'n/a':>10s}"
            lines.append(
                f"{name:20s} {'Y':3s} {f['V0_A3']:12.3f} {f['E0_eV']:14.6f} "
                f"{f['B_GPa']:10.2f} {rms_s}"
            )
        else:
            lines.append(
                f"{name:20s} {'N':3s} {'—':>12s} {'—':>14s} {'—':>10s}  {f.get('error')}"
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from-result",
        type=Path,
        required=True,
        help="Existing EOS result.json with volumes_A3 + energies_eV",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="Overwrite result.json (and eos.json if present) with multi-scheme fits",
    )
    args = ap.parse_args()
    path = args.from_result.resolve()
    row = json.loads(path.read_text(encoding="utf-8"))
    if "volumes_A3" not in row or "energies_eV" not in row:
        raise SystemExit(f"{path}: need volumes_A3 and energies_eV")
    updated = attach_fits(row)
    print(format_fits_table(updated["fits"]))
    print(f"n_fits_ok={updated['n_fits_ok']}/{updated['n_fits']}")
    if args.write:
        path.write_text(json.dumps(updated, indent=2), encoding="utf-8")
        eos_path = path.parent / "eos.json"
        eos_path.write_text(json.dumps([updated], indent=2), encoding="utf-8")
        print(f"wrote {path}")
        print(f"wrote {eos_path}")
    if not fits_all_ok(updated["fits"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
