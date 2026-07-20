#!/usr/bin/env python3
"""Local lattice distortion from relaxed SQS .extxyz.

Target properties: bond-length / NN distributions; displacements; RMSD; local strain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list


def bond_length_stats(atoms, pairs, cutoff: float):
    i, j, d = neighbor_list("ijd", atoms, cutoff)
    sym = np.array(atoms.get_chemical_symbols())
    out = {}
    for a, b in pairs:
        mask = ((sym[i] == a) & (sym[j] == b)) | ((sym[i] == b) & (sym[j] == a))
        dists = d[mask]
        if dists.size == 0:
            raise RuntimeError(
                f"No bonds for pair {a}-{b} within cutoff={cutoff}; "
                "adjust --cutoff or --bond-pairs"
            )
        out[f"{a}-{b}"] = {
            "count": int(dists.size),
            "mean": float(dists.mean()),
            "std": float(dists.std()),
            "min": float(dists.min()),
            "max": float(dists.max()),
            "histogram_counts": np.histogram(dists, bins=50)[0].astype(int).tolist(),
            "histogram_edges": np.histogram(dists, bins=50)[1].tolist(),
        }
    return out


def displacements(relaxed, ideal):
    if len(relaxed) != len(ideal):
        raise ValueError(
            f"Atom count mismatch: relaxed={len(relaxed)} ideal={len(ideal)}"
        )
    if relaxed.get_chemical_symbols() != ideal.get_chemical_symbols():
        raise ValueError(
            "Chemical symbol order differs between relaxed and ideal; "
            "ideal must use the same occupancy ordering"
        )
    dr = relaxed.get_positions() - ideal.get_positions()
    # minimum-image correction
    cell = np.asarray(relaxed.cell.array, dtype=float)
    frac = np.linalg.solve(cell.T, dr.T).T
    frac -= np.rint(frac)
    dr = frac @ cell
    norms = np.linalg.norm(dr, axis=1)
    rmsd = float(np.sqrt(np.mean(norms**2)))
    return dr, norms, rmsd


def local_strain(norms: np.ndarray, a_ref: float) -> dict:
    if a_ref <= 0:
        raise ValueError(f"--a-ref must be positive, got {a_ref}")
    # crude scalar strain proxy: |u| / (a/2) for rocksalt NN scale
    scale = 0.5 * a_ref
    strain = norms / scale
    return {
        "mean": float(strain.mean()),
        "std": float(strain.std()),
        "max": float(strain.max()),
        "histogram_counts": np.histogram(strain, bins=50)[0].astype(int).tolist(),
        "histogram_edges": np.histogram(strain, bins=50)[1].tolist(),
    }


def parse_pairs(values: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for item in values:
        if "-" not in item:
            raise ValueError(f"Bond pair must look like A-B, got {item!r}")
        a, b = item.split("-", 1)
        if not a or not b:
            raise ValueError(f"Invalid bond pair {item!r}")
        pairs.append((a, b))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="LLD analysis for relaxed SQS")
    parser.add_argument(
        "--relaxed",
        type=Path,
        default=Path("../03_relax_ase_mlip/relaxed/all_relaxed.extxyz"),
    )
    parser.add_argument(
        "--ideal",
        type=Path,
        required=True,
        help="Ideal undistorted prototype .extxyz (same order/occupancy)",
    )
    parser.add_argument("--cutoff", type=float, default=3.2)
    parser.add_argument(
        "--bond-pairs",
        nargs="+",
        default=["Ti-N", "Ti-Ti"],
        help="Pairs like Ti-N Zr-N",
    )
    parser.add_argument("--a-ref", type=float, required=True, help="Reference a (Å)")
    parser.add_argument("--out-dir", type=Path, default=Path("lld_out"))
    args = parser.parse_args()

    if not args.relaxed.is_file():
        raise FileNotFoundError(args.relaxed.resolve())
    if not args.ideal.is_file():
        raise FileNotFoundError(args.ideal.resolve())

    pairs = parse_pairs(args.bond_pairs)
    frames = read(args.relaxed, index=":")
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) == 0:
        raise RuntimeError(f"No frames in {args.relaxed}")
    ideal = read(args.ideal, index=0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n, atoms in enumerate(frames):
        bonds = bond_length_stats(atoms, pairs, args.cutoff)
        dr, norms, rmsd = displacements(atoms, ideal)
        strain = local_strain(norms, args.a_ref)
        payload = {
            "frame": n,
            "formula": atoms.get_chemical_formula(),
            "bonds": bonds,
            "rmsd_A": rmsd,
            "mean_abs_disp_A": float(norms.mean()),
            "local_strain": strain,
        }
        out = args.out_dir / f"lld_frame_{n:03d}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {out}  RMSD={rmsd:.5f} Å")


if __name__ == "__main__":
    main()
