#!/usr/bin/env python3
"""Local lattice distortion from relaxed SQS .extxyz."""
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
        counts, edges = np.histogram(dists, bins=50)
        out[f"{a}-{b}"] = {
            "count": int(dists.size),
            "mean": float(dists.mean()) if dists.size else None,
            "std": float(dists.std()) if dists.size else None,
            "min": float(dists.min()) if dists.size else None,
            "max": float(dists.max()) if dists.size else None,
            "histogram_counts": counts.astype(int).tolist(),
            "histogram_edges": edges.tolist(),
        }
    return out


def displacements(relaxed, ideal):
    dr = relaxed.get_positions() - ideal.get_positions()
    cell = np.asarray(relaxed.cell.array, dtype=float)
    frac = np.linalg.solve(cell.T, dr.T).T
    frac -= np.rint(frac)
    dr = frac @ cell
    norms = np.linalg.norm(dr, axis=1)
    return norms, float(np.sqrt(np.mean(norms**2)))


def local_strain(norms: np.ndarray, a_ref: float) -> dict:
    strain = norms / (0.5 * a_ref)
    counts, edges = np.histogram(strain, bins=50)
    return {
        "mean": float(strain.mean()),
        "std": float(strain.std()),
        "max": float(strain.max()),
        "histogram_counts": counts.astype(int).tolist(),
        "histogram_edges": edges.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLD analysis for relaxed SQS")
    parser.add_argument(
        "--relaxed",
        type=Path,
        default=Path("../03_relax_ase_mlip/relaxed/all_relaxed.extxyz"),
    )
    parser.add_argument("--ideal", type=Path, required=True)
    parser.add_argument("--cutoff", type=float, default=3.2)
    parser.add_argument("--bond-pairs", nargs="+", default=["Ti-N", "Ti-Ti"])
    parser.add_argument("--a-ref", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("lld_out"))
    args = parser.parse_args()

    pairs = [tuple(p.split("-", 1)) for p in args.bond_pairs]
    frames = read(args.relaxed, index=":")
    if not isinstance(frames, list):
        frames = [frames]
    ideal = read(args.ideal, index=0)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n, atoms in enumerate(frames):
        norms, rmsd = displacements(atoms, ideal)
        payload = {
            "frame": n,
            "formula": atoms.get_chemical_formula(),
            "bonds": bond_length_stats(atoms, pairs, args.cutoff),
            "rmsd_A": rmsd,
            "mean_abs_disp_A": float(norms.mean()),
            "local_strain": local_strain(norms, args.a_ref),
        }
        out = args.out_dir / f"lld_frame_{n:03d}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out}  RMSD={rmsd:.5f} Å")


if __name__ == "__main__":
    main()
