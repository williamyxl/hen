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
    if a <= 0:
        raise ValueError(f"Lattice constant a must be positive, got {a}")
    if any(n < 1 for n in supercell):
        raise ValueError(f"supercell repeats must be >= 1, got {supercell}")
    if not anion:
        raise ValueError("anion symbol must be non-empty")

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
    # Placeholder cation symbol replaced in assign_cation_composition
    symbols = ["Ti"] * 4 + [anion] * 4
    frac = np.vstack([cation_frac, anion_frac])
    base = Atoms(symbols=symbols, scaled_positions=frac, cell=cell, pbc=True)
    base.set_tags([1] * 4 + [2] * 4)
    nx, ny, nz = supercell
    return base.repeat((nx, ny, nz))


def load_template(path: str | Path) -> Atoms:
    """Load .extxyz template; require tags 1=swappable, 2=fixed."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Template structure not found: {path.resolve()}")
    atoms = read(path, index=0)
    tags = np.asarray(atoms.get_tags(), dtype=int)
    if tags.size != len(atoms):
        raise ValueError(f"{path}: tags length {tags.size} != natoms {len(atoms)}")
    if not np.any(tags == 1):
        raise ValueError(f"{path}: no cation/swappable sites with tag==1")
    if not np.any(tags == 2):
        raise ValueError(f"{path}: no anion/fixed sites with tag==2")
    return atoms


def assign_cation_composition(
    atoms: Atoms,
    cation_composition: dict[str, float],
    rng: np.random.Generator,
) -> Atoms:
    """Fill tag==1 sites with species at exact composition counts."""
    if not cation_composition:
        raise ValueError("cation_composition is empty")

    atoms = atoms.copy()
    tags = np.asarray(atoms.get_tags(), dtype=int)
    cation_idx = np.where(tags == 1)[0]
    n = len(cation_idx)
    if n == 0:
        raise ValueError("No cation sites (tag==1) found")

    species = list(cation_composition.keys())
    fracs = np.array([float(cation_composition[s]) for s in species], dtype=float)
    if np.any(fracs < 0):
        raise ValueError(f"cation_composition fractions must be >= 0: {cation_composition}")
    if abs(fracs.sum() - 1.0) > 1e-6:
        raise ValueError(f"cation_composition must sum to 1; got {fracs.sum()}")

    counts = np.floor(fracs * n + 1e-12).astype(int)
    while counts.sum() < n:
        resid = fracs * n - counts
        counts[int(np.argmax(resid))] += 1
    while counts.sum() > n:
        counts[int(np.argmax(counts))] -= 1
    if int(counts.sum()) != n:
        raise RuntimeError(f"Failed to assign site counts: counts={counts.tolist()} n={n}")
    if np.any(counts < 0):
        raise RuntimeError(f"Negative site counts: {counts.tolist()}")

    labels: list[str] = []
    for s, c in zip(species, counts):
        labels.extend([s] * int(c))
    if len(labels) != n:
        raise RuntimeError(f"Label length {len(labels)} != cation sites {n}")
    labels_arr = np.array(labels, dtype=object)
    rng.shuffle(labels_arr)

    symbols = np.array(atoms.get_chemical_symbols(), dtype=object)
    symbols[cation_idx] = labels_arr
    atoms.set_chemical_symbols([str(s) for s in symbols])
    return atoms


def cation_indices(atoms: Atoms) -> np.ndarray:
    idx = np.where(np.asarray(atoms.get_tags(), dtype=int) == 1)[0]
    if idx.size == 0:
        raise ValueError("No cation sites (tag==1) found")
    return idx


def propose_swap(atoms: Atoms, rng: np.random.Generator) -> tuple[Atoms, int, int]:
    """Swap two unlike cations. Raises if no unlike cation pair exists."""
    idx = cation_indices(atoms)
    symbols = np.array(atoms.get_chemical_symbols())
    cats = symbols[idx]
    if len(np.unique(cats)) < 2:
        raise ValueError(
            "Cannot propose cation swap: fewer than two distinct cation species "
            f"(found {sorted(set(cats.tolist()))})"
        )

    pairs: list[tuple[int, int]] = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = int(idx[a]), int(idx[b])
            if symbols[i] != symbols[j]:
                pairs.append((i, j))
    if not pairs:
        raise RuntimeError("Internal error: unlike cation species present but no pairs found")

    i, j = pairs[int(rng.integers(0, len(pairs)))]
    new = atoms.copy()
    sym = list(new.get_chemical_symbols())
    sym[i], sym[j] = sym[j], sym[i]
    new.set_chemical_symbols(sym)
    return new, i, j


def warren_cowley_alpha(
    atoms: Atoms,
    cutoff: float,
    shell_edges: list[float],
) -> dict[tuple[int, str, str], float]:
    """Warren–Cowley alpha_ij for each shell and cation–cation pair."""
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if len(shell_edges) < 2:
        raise ValueError("shell_edges needs at least two edges")
    if any(shell_edges[k] >= shell_edges[k + 1] for k in range(len(shell_edges) - 1)):
        raise ValueError(f"shell_edges must be strictly increasing: {shell_edges}")

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
                c_b = conc[b]
                if c_b <= 0:
                    raise RuntimeError(f"Non-positive concentration for {b}")
                alphas[(shell + 1, a, b)] = 1.0 - p_ab / c_b
    return alphas


def sqs_correlation_score(atoms: Atoms, cutoff: float, shell_edges: list[float]) -> float:
    """Mean absolute Warren–Cowley alpha (0 = ideal random / SQS-like)."""
    alphas = warren_cowley_alpha(atoms, cutoff, shell_edges)
    if not alphas:
        raise RuntimeError(
            "No cation–cation pairs found within cutoff for Warren–Cowley score; "
            f"increase sro_cutoff (current={cutoff})"
        )
    return float(np.mean(np.abs(np.fromiter(alphas.values(), dtype=float))))


def composition_string(atoms: Atoms) -> str:
    return "".join(f"{s}{n}" for s, n in sorted(Counter(atoms.get_chemical_symbols()).items()))
