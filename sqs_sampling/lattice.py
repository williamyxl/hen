"""Prototype supercells and cation-occupation helpers for HEN SQS MC."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list


def build_rocksalt_supercell(anion: str, a: float, supercell: tuple[int, int, int]) -> Atoms:
    """Conventional rocksalt cell repeated to supercell; tags: 1=cation, 2=anion."""
    cell = np.eye(3) * a
    cation_frac = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    )
    anion_frac = (cation_frac + 0.5) % 1.0
    symbols = ["Ti"] * 4 + [anion] * 4
    frac = np.vstack([cation_frac, anion_frac])
    base = Atoms(symbols=symbols, scaled_positions=frac, cell=cell, pbc=True)
    base.set_tags([1] * 4 + [2] * 4)
    return base.repeat(tuple(supercell))


def load_template(path: str | Path) -> Atoms:
    """Load template; tags 1=swappable, 2=fixed."""
    atoms = read(path, index=0)
    tags = np.asarray(atoms.get_tags(), dtype=int)
    if not np.any(tags == 1) or not np.any(tags == 2):
        raise ValueError(f"{path}: need tags 1 (cation) and 2 (anion)")
    return atoms


def assign_cation_composition(
    atoms: Atoms,
    cation_composition: dict[str, float],
    rng: np.random.Generator,
) -> Atoms:
    """Fill tag==1 sites with species at exact composition counts."""
    atoms = atoms.copy()
    cation_idx = cation_indices(atoms)
    n = len(cation_idx)
    species = list(cation_composition)
    fracs = np.array([float(cation_composition[s]) for s in species], dtype=float)
    if abs(fracs.sum() - 1.0) > 1e-6:
        raise ValueError(f"cation_composition must sum to 1; got {fracs.sum()}")

    counts = np.floor(fracs * n + 1e-12).astype(int)
    while counts.sum() < n:
        counts[int(np.argmax(fracs * n - counts))] += 1
    while counts.sum() > n:
        counts[int(np.argmax(counts))] -= 1

    labels = np.array([s for s, c in zip(species, counts) for _ in range(int(c))], dtype=object)
    rng.shuffle(labels)
    symbols = np.array(atoms.get_chemical_symbols(), dtype=object)
    symbols[cation_idx] = labels
    atoms.set_chemical_symbols([str(s) for s in symbols])
    return atoms


def cation_indices(atoms: Atoms) -> np.ndarray:
    idx = np.where(np.asarray(atoms.get_tags(), dtype=int) == 1)[0]
    if idx.size == 0:
        raise ValueError("No cation sites (tag==1) found")
    return idx


def propose_swap(atoms: Atoms, rng: np.random.Generator) -> tuple[Atoms, int, int]:
    """Swap two unlike cations."""
    idx = cation_indices(atoms)
    symbols = np.array(atoms.get_chemical_symbols())
    for _ in range(10_000):
        i, j = (int(x) for x in idx[rng.integers(0, len(idx), size=2)])
        if symbols[i] != symbols[j]:
            new = atoms.copy()
            sym = list(new.get_chemical_symbols())
            sym[i], sym[j] = sym[j], sym[i]
            new.set_chemical_symbols(sym)
            return new, i, j
    raise ValueError("Cannot propose unlike cation swap")


def warren_cowley_alpha(
    atoms: Atoms,
    cutoff: float,
    shell_edges: list[float],
) -> dict[tuple[int, str, str], float]:
    """Warren–Cowley alpha_ij for each shell and cation–cation pair."""
    cation_idx = cation_indices(atoms)
    symbols = np.array(atoms.get_chemical_symbols())
    cation_symbols = symbols[cation_idx]
    conc = {s: float((cation_symbols == s).mean()) for s in sorted(set(cation_symbols))}

    i_all, j_all, d_all = neighbor_list("ijd", atoms, cutoff)
    keep = np.isin(i_all, cation_idx) & np.isin(j_all, cation_idx)
    i_all, j_all, d_all = i_all[keep], j_all[keep], d_all[keep]
    shells = np.digitize(d_all, shell_edges) - 1

    alphas: dict[tuple[int, str, str], float] = {}
    species = sorted(conc)
    for shell in range(len(shell_edges) - 1):
        mask = shells == shell
        if not np.any(mask):
            continue
        for a in species:
            for b in species:
                ia = symbols[i_all[mask]] == a
                n_bonds = int(ia.sum())
                if n_bonds == 0:
                    continue
                p_ab = float((symbols[j_all[mask]][ia] == b).sum() / n_bonds)
                alphas[(shell + 1, a, b)] = 1.0 - p_ab / conc[b]
    return alphas


def sqs_correlation_score(atoms: Atoms, cutoff: float, shell_edges: list[float]) -> float:
    """Mean absolute Warren–Cowley alpha (0 = ideal random / SQS-like)."""
    alphas = warren_cowley_alpha(atoms, cutoff, shell_edges)
    if not alphas:
        raise RuntimeError(f"No cation–cation pairs within cutoff={cutoff}")
    return float(np.mean(np.abs(np.fromiter(alphas.values(), dtype=float))))


def composition_string(atoms: Atoms) -> str:
    return "".join(f"{s}{n}" for s, n in sorted(Counter(atoms.get_chemical_symbols()).items()))
