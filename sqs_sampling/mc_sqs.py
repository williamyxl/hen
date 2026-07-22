#!/usr/bin/env python3
"""Monte Carlo SQS sampling.

Default energy: GFN2-xTB (TBLite, 2000 SCC cycles).
Also supported: uma | mace. Outputs to final_sqs/ as .extxyz.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
from ase.io import write

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


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg["prototype"] == "from_file" and not cfg.get("structure_file"):
        raise KeyError("prototype=from_file requires structure_file")
    return cfg


def build_initial(cfg: dict, rng: np.random.Generator):
    if cfg["prototype"] == "from_file":
        atoms = load_template(cfg["structure_file"])
    elif cfg["prototype"] == "rocksalt":
        sc = tuple(int(x) for x in cfg["supercell"])
        atoms = build_rocksalt_supercell(cfg["anion"], float(cfg["a"]), sc)
    else:
        raise ValueError(f"Unknown prototype {cfg['prototype']!r}")
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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(cfg["log_file"]), logging.StreamHandler()],
        force=True,
    )
    log = logging.getLogger("mc_sqs")

    rng = np.random.default_rng(int(cfg["seed"]))
    calc = build_calculator(
        energy_method,
        device=device,
        uma_model=str(cfg["uma_model"]),
        uma_task=str(cfg["uma_task"]),
        mace_model=mace_model if mace_model is not None else cfg.get("mace_model"),
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
    traj_path = Path(cfg["trajectory_file"])
    if traj_path.exists():
        traj_path.unlink()

    accepted = 0
    samples: list[tuple[float, float, object]] = []
    best_e, best_atoms, best_score = e, atoms.copy(), score

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
            frame.info.update(
                energy=e,
                mc_step=step,
                energy_method=energy_method,
                sqs_abs_alpha=frame_score,
            )
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

    samples.append((best_e, best_score, best_atoms))
    samples.sort(key=lambda x: (x[1], x[0]))  # low |alpha|, then low E

    n_final = int(cfg["n_final"])
    seen: set[tuple[str, ...]] = set()
    written = []
    for e_i, score_i, frame in samples:
        key = tuple(frame.get_chemical_symbols())
        if key in seen:
            continue
        seen.add(key)
        frame.info.update(energy=e_i, energy_method=energy_method, sqs_abs_alpha=score_i)
        out = out_dir / f"sqs_{len(written):03d}.extxyz"
        write(out, frame)
        written.append(frame)
        log.info("wrote %s  E=%.6f eV  |alpha|=%.4f", out, e_i, score_i)
        if len(written) >= n_final:
            break

    if len(written) < n_final:
        raise RuntimeError(
            f"Only {len(written)} distinct occupations (n_final={n_final}); "
            "increase n_steps or lower sample_every"
        )
    write(out_dir / "sqs.extxyz", written)
    log.info("wrote %s (%d frames)", out_dir / "sqs.extxyz", len(written))


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo SQS sampling (default: GFN2-xTB)")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--energy",
        choices=SUPPORTED,
        default=None,
        help="Energy method (default: config or gfn2-xtb)",
    )
    parser.add_argument("--device", default=None, help="cpu|cuda (UMA/MACE)")
    parser.add_argument("--mace-model", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    energy = args.energy or str(cfg.get("energy", "gfn2-xtb"))
    device = args.device or str(cfg.get("device", "cpu"))
    run_mc(cfg, energy, device, args.mace_model)


if __name__ == "__main__":
    main()
