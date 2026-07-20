#!/usr/bin/env python3
"""Monte Carlo SQS sampling with selectable energy method.

Energy methods: gfn2-xtb | uma | mace
Outputs selected structures to final_sqs/ as .extxyz.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
from ase.io import read, write

from energy import SUPPORTED, build_calculator, evaluate_energy
from lattice import (
    assign_cation_composition,
    build_rocksalt_supercell,
    composition_string,
    load_template,
    propose_swap,
    sqs_correlation_score,
)

KB_EV = 8.617333262145e-5  # eV/K

REQUIRED_KEYS = (
    "prototype",
    "anion",
    "cation_composition",
    "a",
    "supercell",
    "seed",
    "n_steps",
    "temperature_K",
    "equilibrate_steps",
    "sample_every",
    "n_final",
    "optional_relax",
    "relax_fmax",
    "relax_steps",
    "energy",
    "device",
    "uma_model",
    "uma_task",
    "output_dir",
    "trajectory_file",
    "log_file",
    "sro_cutoff",
    "sro_shell_edges",
)


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing config {path.resolve()}; copy config.example.yaml -> config.yaml"
        )
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config root must be a mapping, got {type(cfg).__name__}")
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"Config missing required keys: {missing}")
    if cfg["prototype"] == "from_file" and not cfg.get("structure_file"):
        raise KeyError("prototype=from_file requires structure_file")
    if int(cfg["n_final"]) < 1:
        raise ValueError(f"n_final must be >= 1, got {cfg['n_final']}")
    if int(cfg["sample_every"]) < 1:
        raise ValueError(f"sample_every must be >= 1, got {cfg['sample_every']}")
    if int(cfg["n_steps"]) < 1:
        raise ValueError(f"n_steps must be >= 1, got {cfg['n_steps']}")
    if float(cfg["temperature_K"]) < 0:
        raise ValueError(f"temperature_K must be >= 0, got {cfg['temperature_K']}")
    return cfg


def build_initial(cfg: dict, rng: np.random.Generator):
    if cfg["prototype"] == "from_file":
        atoms = load_template(cfg["structure_file"])
    elif cfg["prototype"] == "rocksalt":
        sc = tuple(int(x) for x in cfg["supercell"])
        if len(sc) != 3:
            raise ValueError(f"supercell must have 3 integers, got {cfg['supercell']}")
        atoms = build_rocksalt_supercell(cfg["anion"], float(cfg["a"]), sc)
    else:
        raise ValueError(
            f"Unknown prototype {cfg['prototype']!r}; use 'rocksalt' or 'from_file'"
        )
    return assign_cation_composition(atoms, cfg["cation_composition"], rng)


def metropolis(de: float, temperature_k: float, rng: np.random.Generator) -> bool:
    if de <= 0:
        return True
    if temperature_k == 0:
        return False
    return bool(rng.random() < np.exp(-de / (KB_EV * temperature_k)))


def run_mc(cfg: dict, energy_method: str, device: str, mace_model: str | None) -> None:
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg["log_file"])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        force=True,
    )
    log = logging.getLogger("mc_sqs")

    rng = np.random.default_rng(int(cfg["seed"]))
    mace_path = mace_model if mace_model is not None else cfg.get("mace_model")
    calc = build_calculator(
        energy_method,
        device=device,
        uma_model=str(cfg["uma_model"]),
        uma_task=str(cfg["uma_task"]),
        mace_model=mace_path,
    )

    atoms = build_initial(cfg, rng)
    relax = bool(cfg["optional_relax"])
    fmax = float(cfg["relax_fmax"])
    relax_steps = int(cfg["relax_steps"])
    sro_cutoff = float(cfg["sro_cutoff"])
    sro_edges = [float(x) for x in cfg["sro_shell_edges"]]

    e = evaluate_energy(atoms, calc, relax=relax, fmax=fmax, steps=relax_steps)
    score = sqs_correlation_score(atoms, sro_cutoff, sro_edges)
    log.info(
        "start %s energy=%s E=%.6f eV |alpha|=%.4f T=%s K steps=%s",
        composition_string(atoms),
        energy_method,
        e,
        score,
        cfg["temperature_K"],
        cfg["n_steps"],
    )

    temperature_k = float(cfg["temperature_K"])
    n_steps = int(cfg["n_steps"])
    equil = int(cfg["equilibrate_steps"])
    sample_every = int(cfg["sample_every"])
    if equil >= n_steps:
        raise ValueError(
            f"equilibrate_steps ({equil}) must be < n_steps ({n_steps}) "
            "so at least one sample can be recorded"
        )

    traj_path = Path(cfg["trajectory_file"])
    if traj_path.exists():
        traj_path.unlink()

    accepted = 0
    samples: list[tuple[float, float, object]] = []
    best_e, best_atoms = e, atoms.copy()
    best_score = score

    for step in range(1, n_steps + 1):
        new_atoms, _, _ = propose_swap(atoms, rng)
        e_new = evaluate_energy(
            new_atoms, calc, relax=relax, fmax=fmax, steps=relax_steps
        )
        if metropolis(e_new - e, temperature_k, rng):
            atoms, e = new_atoms, e_new
            accepted += 1
            if e < best_e:
                best_e, best_atoms = e, atoms.copy()
                best_score = sqs_correlation_score(atoms, sro_cutoff, sro_edges)

        if step > equil and step % sample_every == 0:
            frame = atoms.copy()
            frame_score = sqs_correlation_score(frame, sro_cutoff, sro_edges)
            frame.info["energy"] = e
            frame.info["mc_step"] = step
            frame.info["energy_method"] = energy_method
            frame.info["sqs_abs_alpha"] = frame_score
            write(traj_path, frame, append=True)
            samples.append((e, frame_score, frame))

        if step % max(n_steps // 10, 1) == 0:
            log.info(
                "step %d/%d  E=%.6f  accept=%.3f  best_E=%.6f  best_|alpha|=%.4f",
                step,
                n_steps,
                e,
                accepted / step,
                best_e,
                best_score,
            )

    if not samples:
        raise RuntimeError(
            "No MC samples recorded; check equilibrate_steps / sample_every / n_steps"
        )

    samples.append((best_e, best_score, best_atoms))
    # Prefer low |alpha| (SQS-like), then low energy
    samples.sort(key=lambda x: (x[1], x[0]))

    n_final = int(cfg["n_final"])
    seen: set[tuple[str, ...]] = set()
    written_frames = []
    for e_i, score_i, frame in samples:
        key = tuple(frame.get_chemical_symbols())
        if key in seen:
            continue
        seen.add(key)
        out = out_dir / f"sqs_{len(written_frames):03d}.extxyz"
        frame.info["energy"] = e_i
        frame.info["energy_method"] = energy_method
        frame.info["sqs_abs_alpha"] = score_i
        write(out, frame)
        written_frames.append(frame)
        log.info("wrote %s  E=%.6f eV  |alpha|=%.4f", out, e_i, score_i)
        if len(written_frames) >= n_final:
            break

    if len(written_frames) < n_final:
        raise RuntimeError(
            f"Only collected {len(written_frames)} distinct occupations, "
            f"but n_final={n_final}; increase n_steps or lower sample_every"
        )

    write(out_dir / "sqs.extxyz", written_frames)
    # Verify readable
    check = read(out_dir / "sqs.extxyz", index=":")
    if not isinstance(check, list):
        check = [check]
    if len(check) != n_final:
        raise RuntimeError(
            f"Expected {n_final} frames in sqs.extxyz, found {len(check)}"
        )
    log.info("wrote %s (%d frames)", out_dir / "sqs.extxyz", len(check))


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo SQS sampling")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--energy",
        choices=SUPPORTED,
        default=None,
        help="Energy method (overrides config)",
    )
    parser.add_argument("--device", default=None, help="cpu|cuda (UMA/MACE)")
    parser.add_argument("--mace-model", default=None, help="Path to MACE model file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    energy = args.energy if args.energy is not None else str(cfg["energy"])
    if energy not in SUPPORTED:
        raise ValueError(f"Unsupported energy method {energy!r}; choose from {SUPPORTED}")
    device = args.device if args.device is not None else str(cfg["device"])
    run_mc(cfg, energy, device, args.mace_model)


if __name__ == "__main__":
    main()
