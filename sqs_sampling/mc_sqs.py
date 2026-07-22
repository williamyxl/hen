#!/usr/bin/env python3
"""Configurationally biased MC SQS sampling with GFN2-xTB / UMA / nvalchemi-UMA.

Each step proposes cbmc_trials concurrent cation swaps, evaluates them
(GFN2: multiprocess Pool; UMA / nvalchemi-uma: shared calculator; nvalchemi
batching off by default), selects a trial with Rosenbluth weights, and
accepts with min(1, W_new / W_old).
"""

from __future__ import annotations

import argparse
import logging
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase import Atoms
from ase.io import write

from energy import (
    CBMC_SUPPORTED,
    build_calculator,
    evaluate_energies,
    formal_charge_and_multiplicity,
)
from calibrate_lattice import resolve_lattice_constant
from lattice import (
    assign_cation_composition,
    build_rocksalt_supercell,
    composition_string,
    load_template,
    propose_swap,
    sqs_correlation_score,
)

KB_EV = 8.617333262145e-5  # eV/K
DEFAULT_CBMC_TRIALS = 10

# Written onto the extxyz comment line (ASE atoms.info → key=value after Properties=).
_EXTXYZ_INFO_KEYS = (
    "sqs_abs_alpha",
    "energy",
    "mc_step",
    "energy_method",
    "rosenbluth_W_new",
    "rosenbluth_W_old",
)


def stamp_extxyz_info(
    frame: Atoms,
    *,
    energy: float,
    mc_step: int,
    method: str,
    alpha: float,
    w_new: float,
    w_old: float,
) -> Atoms:
    """Put SQS metadata on the extxyz comment line; alpha first for visibility."""
    for key in _EXTXYZ_INFO_KEYS:
        frame.info.pop(key, None)
    # Insertion order becomes comment-line order in ASE extxyz.
    frame.info["sqs_abs_alpha"] = float(alpha)
    frame.info["energy"] = float(energy)
    frame.info["mc_step"] = int(mc_step)
    frame.info["energy_method"] = str(method)
    frame.info["rosenbluth_W_new"] = float(w_new)
    frame.info["rosenbluth_W_old"] = float(w_old)
    return frame


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg["prototype"] == "from_file" and not cfg.get("structure_file"):
        raise KeyError("prototype=from_file requires structure_file")
    method = str(cfg.get("energy", "uma")).strip().lower()
    if method not in CBMC_SUPPORTED:
        raise ValueError(
            f"CBMC energy method {method!r} not supported; choose from {CBMC_SUPPORTED}"
        )
    cfg["energy"] = method
    return cfg


def build_initial(cfg: dict, rng: np.random.Generator) -> Atoms:
    if cfg["prototype"] == "from_file":
        atoms = load_template(cfg["structure_file"])
    elif cfg["prototype"] == "rocksalt":
        a = resolve_lattice_constant(cfg)
        cfg["a"] = a  # record resolved (possibly Vegard) lattice constant
        sc = tuple(int(x) for x in cfg["supercell"])
        atoms = build_rocksalt_supercell(cfg["anion"], a, sc)
    else:
        raise ValueError(f"Unknown prototype {cfg['prototype']!r}")
    return assign_cation_composition(atoms, cfg["cation_composition"], rng)


def rosenbluth_weights(energies: np.ndarray, beta: float) -> tuple[np.ndarray, float]:
    """Boltzmann weights with shift for stability; returns (w, W=sum w)."""
    shifted = energies - float(np.min(energies))
    weights = np.exp(-beta * shifted)
    wsum = float(weights.sum())
    if not np.isfinite(wsum) or wsum <= 0:
        raise RuntimeError(f"Invalid Rosenbluth weight sum: {wsum}")
    return weights, wsum


def _eval(
    atoms_list: list[Atoms],
    *,
    method: str,
    pool: Pool | None,
    calc: Any | None,
) -> list[float]:
    return evaluate_energies(atoms_list, method, pool=pool, calc=calc)


def cbmc_step(
    atoms: Atoms,
    energy: float,
    *,
    method: str,
    pool: Pool | None,
    calc: Any | None,
    rng: np.random.Generator,
    n_trials: int,
    beta: float,
) -> tuple[Atoms, float, bool, float, float]:
    """One Rosenbluth CBMC step with n_trials energy evaluations.

    Returns (atoms, energy, accepted, W_new, W_old).
    """
    # Forward: k random swaps from current occupation
    trials = [propose_swap(atoms, rng)[0] for _ in range(n_trials)]
    e_fwd = np.asarray(
        _eval(trials, method=method, pool=pool, calc=calc), dtype=float
    )
    w_fwd, w_new = rosenbluth_weights(e_fwd, beta)
    choice = int(rng.choice(n_trials, p=w_fwd / w_new))
    new_atoms = trials[choice]
    e_new = float(e_fwd[choice])

    # Reverse: old config + (k-1) random swaps from the selected trial
    rev_trials = [propose_swap(new_atoms, rng)[0] for _ in range(n_trials - 1)]
    e_rev_rest = (
        _eval(rev_trials, method=method, pool=pool, calc=calc) if rev_trials else []
    )
    e_rev = np.asarray([energy, *e_rev_rest], dtype=float)
    _, w_old = rosenbluth_weights(e_rev, beta)

    accept = bool(rng.random() < min(1.0, w_new / w_old))
    if accept:
        return new_atoms, e_new, True, w_new, w_old
    return atoms, energy, False, w_new, w_old


def run_mc(cfg: dict) -> None:
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(cfg["log_file"]), logging.StreamHandler()],
        force=True,
    )
    log = logging.getLogger("mc_sqs")

    n_trials = int(cfg.get("cbmc_trials", DEFAULT_CBMC_TRIALS))
    if n_trials < 2:
        raise ValueError(f"cbmc_trials must be >= 2, got {n_trials}")

    temperature_k = float(cfg["temperature_K"])
    if temperature_k <= 0:
        raise ValueError("CBMC requires temperature_K > 0")
    beta = 1.0 / (KB_EV * temperature_k)

    method = cfg["energy"]
    rng = np.random.default_rng(int(cfg["seed"]))
    atoms = build_initial(cfg, rng)
    sro_cutoff = float(cfg["sro_cutoff"])
    sro_edges = [float(x) for x in cfg["sro_shell_edges"]]
    n_steps = int(cfg["n_steps"])
    equil = int(cfg["equilibrate_steps"])
    sample_every = int(cfg["sample_every"])

    traj_path = Path(cfg["trajectory_file"])
    if traj_path.exists():
        traj_path.unlink()

    charge, _multiplicity = formal_charge_and_multiplicity(atoms)
    calc = None
    if method != "gfn2-xtb":
        calc = build_calculator(
            method,
            atoms=atoms,
            device=str(cfg.get("device", "cuda")),
            uma_model=str(
                cfg.get("uma_model", "/mnt/d/workdir/uma-cache/uma-s-1p2.pt")
            ),
            uma_task=str(cfg.get("uma_task", "omat")),
            inference_settings=str(
                cfg.get("inference_settings", "default")
            ),
            nvalchemi_batch=bool(cfg.get("nvalchemi_batch", False)),
            mace_model=cfg.get("mace_model"),
        )

    def _run(pool: Pool | None) -> None:
        nonlocal atoms
        e = _eval([atoms], method=method, pool=pool, calc=calc)[0]
        score = sqs_correlation_score(atoms, sro_cutoff, sro_edges)
        if method in ("uma", "nvalchemi-uma"):
            spin_note = "uma_spin=0 (off)"
        else:
            spin_note = "tblite_spin=ignored(mult=1)"
        log.info(
            "start %s energy=%s supercell=%s a=%.6f Å CBMC trials=%d charge=%d %s "
            "E=%.6f eV |alpha|=%.4f T=%s K steps=%s",
            composition_string(atoms),
            method,
            list(cfg.get("supercell", [])),
            float(cfg["a"]) if cfg.get("a") is not None else float("nan"),
            n_trials,
            charge,
            spin_note,
            e,
            score,
            temperature_k,
            n_steps,
        )

        accepted = 0
        samples: list[tuple[float, float, Atoms]] = []
        best_e, best_atoms, best_score = e, atoms.copy(), score

        for step in range(1, n_steps + 1):
            atoms, e, ok, w_new, w_old = cbmc_step(
                atoms,
                e,
                method=method,
                pool=pool,
                calc=calc,
                rng=rng,
                n_trials=n_trials,
                beta=beta,
            )
            if ok:
                accepted += 1
                if e < best_e:
                    best_e, best_atoms = e, atoms.copy()
                    best_score = sqs_correlation_score(atoms, sro_cutoff, sro_edges)

            if step > equil and step % sample_every == 0:
                frame = atoms.copy()
                frame_score = sqs_correlation_score(frame, sro_cutoff, sro_edges)
                stamp_extxyz_info(
                    frame,
                    energy=e,
                    mc_step=step,
                    method=method,
                    alpha=frame_score,
                    w_new=w_new,
                    w_old=w_old,
                )
                write(traj_path, frame, format="extxyz", append=True)
                samples.append((e, frame_score, frame))

            if step % max(n_steps // 10, 1) == 0:
                log.info(
                    "step %d/%d  E=%.6f  accept=%.3f  W_new/W_old=%.4f  "
                    "best_E=%.6f  best_|alpha|=%.4f",
                    step,
                    n_steps,
                    e,
                    accepted / step,
                    w_new / w_old,
                    best_e,
                    best_score,
                )

        # Write in CBMC step order (same as mc_trajectory.extxyz).
        written: list[Atoms] = []
        for e_i, score_i, frame in samples:
            stamp_extxyz_info(
                frame,
                energy=e_i,
                mc_step=int(frame.info["mc_step"]),
                method=method,
                alpha=score_i,
                w_new=float(frame.info["rosenbluth_W_new"]),
                w_old=float(frame.info["rosenbluth_W_old"]),
            )
            out = out_dir / f"sqs_{len(written):03d}.extxyz"
            write(out, frame, format="extxyz")
            written.append(frame)
            log.info(
                "wrote %s  mc_step=%s  E=%.6f eV  |alpha|=%.4f",
                out,
                frame.info.get("mc_step"),
                e_i,
                score_i,
            )
        write(out_dir / "sqs.extxyz", written, format="extxyz")
        log.info("wrote %s (%d frames, CBMC order)", out_dir / "sqs.extxyz", len(written))

    if method == "gfn2-xtb":
        with Pool(processes=n_trials) as pool:
            _run(pool)
    else:
        _run(None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rosenbluth CBMC SQS sampling (GFN2-xTB or UMA)"
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    run_mc(load_config(args.config))


if __name__ == "__main__":
    main()
