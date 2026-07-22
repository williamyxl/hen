#!/usr/bin/env python3
"""SQS SRO analysis from .extxyz (Warren–Cowley; pair correlations)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list


def warren_cowley(atoms, cutoff: float, shell_edges: list[float]):
    i, j, d = neighbor_list("ijd", atoms, cutoff)
    symbols = np.array(atoms.get_chemical_symbols())
    conc = {s: float((symbols == s).mean()) for s in sorted(set(symbols))}
    shells = np.digitize(d, shell_edges) - 1
    results = {}
    species = sorted(conc)
    for shell in range(len(shell_edges) - 1):
        mask = shells == shell
        if not np.any(mask):
            continue
        for a in species:
            for b in species:
                ia = symbols[i[mask]] == a
                n_bonds = int(ia.sum())
                if n_bonds == 0:
                    continue
                p_ab = float((symbols[j[mask]][ia] == b).sum() / n_bonds)
                results[f"shell{shell + 1}_{a}-{b}"] = {
                    "P_ab": p_ab,
                    "c_b": conc[b],
                    "alpha": 1.0 - p_ab / conc[b],
                    "n_bonds": n_bonds,
                }
    return results


def pair_correlation(atoms, cutoff: float, dr: float = 0.05):
    i, j, d = neighbor_list("ijd", atoms, cutoff)
    symbols = np.array(atoms.get_chemical_symbols())
    bins = np.arange(0.0, cutoff + dr, dr)
    out = {}
    for a in sorted(set(symbols)):
        for b in sorted(set(symbols)):
            hist, edges = np.histogram(d[(symbols[i] == a) & (symbols[j] == b)], bins=bins)
            out[f"{a}-{b}"] = {"edges": edges.tolist(), "counts": hist.astype(int).tolist()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="SRO analysis for SQS .extxyz")
    parser.add_argument(
        "--structures",
        type=Path,
        default=Path("../../sqs_sampling/final_sqs/sqs.extxyz"),
    )
    parser.add_argument("--cutoff", type=float, default=3.5)
    parser.add_argument("--shell-edges", type=float, nargs="+", default=[0.0, 2.6, 3.5])
    parser.add_argument("--out-dir", type=Path, default=Path("sro_out"))
    args = parser.parse_args()

    frames = read(args.structures, index=":")
    if not isinstance(frames, list):
        frames = [frames]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n, atoms in enumerate(frames):
        wc = warren_cowley(atoms, args.cutoff, list(args.shell_edges))
        payload = {
            "frame": n,
            "formula": atoms.get_chemical_formula(),
            "warren_cowley": wc,
            "pair_correlation": pair_correlation(atoms, args.cutoff),
        }
        out = args.out_dir / f"sro_frame_{n:03d}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out}")
        for key, val in sorted(wc.items()):
            print(f"  {key}: alpha={val['alpha']:.4f} P={val['P_ab']:.4f}")


if __name__ == "__main__":
    main()
