#!/usr/bin/env python3
"""SQS SRO analysis from .extxyz (no energy evaluation).

Target properties: Warren–Cowley parameters; pair correlation functions.
"""
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
                c_b = conc[b]
                if c_b <= 0:
                    raise RuntimeError(f"Non-positive concentration for {b}")
                results[f"shell{shell + 1}_{a}-{b}"] = {
                    "P_ab": p_ab,
                    "c_b": c_b,
                    "alpha": 1.0 - p_ab / c_b,
                    "n_bonds": n_bonds,
                }
    if not results:
        raise RuntimeError(
            f"No neighbor pairs within cutoff={cutoff}; increase --cutoff"
        )
    return results


def pair_correlation(atoms, cutoff: float, dr: float = 0.05):
    if dr <= 0:
        raise ValueError(f"dr must be positive, got {dr}")
    i, j, d = neighbor_list("ijd", atoms, cutoff)
    symbols = np.array(atoms.get_chemical_symbols())
    bins = np.arange(0.0, cutoff + dr, dr)
    out = {}
    for a in sorted(set(symbols)):
        for b in sorted(set(symbols)):
            mask = (symbols[i] == a) & (symbols[j] == b)
            hist, edges = np.histogram(d[mask], bins=bins)
            out[f"{a}-{b}"] = {
                "edges": edges.tolist(),
                "counts": hist.astype(int).tolist(),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="SRO analysis for SQS .extxyz")
    parser.add_argument(
        "--structures",
        type=Path,
        default=Path("../../sqs_sampling/final_sqs/sqs.extxyz"),
    )
    parser.add_argument("--cutoff", type=float, default=3.5)
    parser.add_argument(
        "--shell-edges",
        type=float,
        nargs="+",
        default=[0.0, 2.6, 3.5],
    )
    parser.add_argument("--out-dir", type=Path, default=Path("sro_out"))
    args = parser.parse_args()

    if not args.structures.is_file():
        raise FileNotFoundError(f"Structures not found: {args.structures.resolve()}")
    if len(args.shell_edges) < 2:
        raise ValueError("--shell-edges needs at least two values")

    frames = read(args.structures, index=":")
    if not isinstance(frames, list):
        frames = [frames]
    if len(frames) == 0:
        raise RuntimeError(f"No frames in {args.structures}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n, atoms in enumerate(frames):
        wc = warren_cowley(atoms, args.cutoff, list(args.shell_edges))
        pc = pair_correlation(atoms, args.cutoff)
        payload = {
            "frame": n,
            "formula": atoms.get_chemical_formula(),
            "warren_cowley": wc,
            "pair_correlation": pc,
        }
        out = args.out_dir / f"sro_frame_{n:03d}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {out}")
        for key, val in sorted(wc.items()):
            print(f"  {key}: alpha={val['alpha']:.4f} P={val['P_ab']:.4f}")


if __name__ == "__main__":
    main()
