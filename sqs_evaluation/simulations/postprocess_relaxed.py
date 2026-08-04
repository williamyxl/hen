#!/usr/bin/env python3
"""Post-relax properties on login/CPU: formation enthalpy, SRO, LLD.

Reuses total cell energies from cell_opt ``result.json`` (no UMA re-eval).

Example::

  python postprocess_relaxed.py \\
    --cell-opt-dir workflow_out/cell_opt_mc_sqs_20260730_032035 \\
    --eref refs/uma/elemental_refs.json \\
    --out-dir workflow_out/post_mc_sqs_20260730_032035
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io import read, write

ROOT = Path(__file__).resolve().parents[1]  # sqs_evaluation/
sys.path.insert(0, str(ROOT / "simulations" / "sro"))
sys.path.insert(0, str(ROOT / "simulations" / "lld"))
from analyze_lld import bond_length_stats, displacements, local_strain  # noqa: E402
from analyze_sro import pair_correlation, warren_cowley  # noqa: E402


def _eref_per_atom(eref: dict) -> dict[str, float]:
    if "per_atom_eV" in eref and isinstance(eref["per_atom_eV"], dict):
        return {str(k): float(v) for k, v in eref["per_atom_eV"].items()}
    out = {
        str(k): float(v)
        for k, v in eref.items()
        if isinstance(v, (int, float)) and 1 <= len(str(k)) <= 2
    }
    if not out:
        raise ValueError("eref must contain per_atom_eV or element→eV map")
    return out


def formation_enthalpy(atoms: Atoms, energy: float, refs: dict[str, float]) -> float:
    symbols = atoms.get_chemical_symbols()
    missing = sorted(set(symbols) - set(refs))
    if missing:
        raise KeyError(f"elemental eref missing elements {missing}")
    n = len(symbols)
    return energy / n - sum(refs[s] for s in symbols) / n


def load_cell_opt_tasks(cell_opt_dir: Path) -> list[dict[str, Any]]:
    tasks_dir = cell_opt_dir / "tasks"
    rows: list[dict[str, Any]] = []
    for result_path in sorted(tasks_dir.glob("*/result.json")):
        meta = json.loads(result_path.read_text(encoding="utf-8"))
        task_dir = result_path.parent
        relaxed_path = task_dir / "relaxed.extxyz"
        if not relaxed_path.is_file():
            raise FileNotFoundError(relaxed_path)
        if "energy_eV" not in meta:
            raise KeyError(f"missing energy_eV in {result_path}")
        rows.append(
            {
                "task_id": meta.get("task_id", task_dir.name),
                "structure": meta["structure"],
                "energy_eV": float(meta["energy_eV"]),
                "result_json": str(result_path),
                "relaxed_path": str(relaxed_path),
                "meta": meta,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No tasks/*/result.json under {cell_opt_dir}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cell-opt-dir",
        type=Path,
        required=True,
        help="Directory from xpu_tile_pool (contains tasks/*/result.json)",
    )
    p.add_argument(
        "--eref",
        type=Path,
        default=ROOT / "refs" / "uma" / "elemental_refs.json",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--sro-cutoff", type=float, default=3.5)
    p.add_argument("--sro-shell-edges", type=float, nargs="+", default=[0.0, 2.6, 3.5])
    p.add_argument("--lld-cutoff", type=float, default=3.2)
    p.add_argument("--lld-a-ref", type=float, default=4.484538475263075)
    p.add_argument(
        "--lld-bond-pairs",
        nargs="+",
        default=["Ti-N", "Zr-N", "Hf-N", "Nb-N", "Ta-N"],
    )
    args = p.parse_args()

    cell_opt_dir = args.cell_opt_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sro").mkdir(exist_ok=True)
    (out_dir / "lld").mkdir(exist_ok=True)

    eref_raw = json.loads(args.eref.read_text(encoding="utf-8"))
    refs = _eref_per_atom(eref_raw)
    tasks = load_cell_opt_tasks(cell_opt_dir)
    pairs = [tuple(x.split("-", 1)) for x in args.lld_bond_pairs]

    print(f"loaded {len(tasks)} relaxed cells from {cell_opt_dir}")
    print(f"eref={args.eref}  species={sorted(refs)}")

    form_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    relaxed_frames: list[Atoms] = []

    for n, task in enumerate(tasks):
        relaxed = read(task["relaxed_path"], index=0)
        ideal = read(task["structure"], index=0)
        e = float(task["energy_eV"])
        dhf = formation_enthalpy(relaxed, e, refs)

        # Ensure total cell energy is stored on the structure + result.json
        relaxed.info["energy_eV"] = e
        relaxed.info["energy_model"] = task["meta"].get("energy_model", "uma")
        relaxed.info["delta_h_f_eV_per_atom"] = dhf
        relaxed.info["task_id"] = task["task_id"]
        relaxed.info["source_structure"] = task["structure"]
        write(task["relaxed_path"], relaxed)  # update in place with energy info
        relaxed_frames.append(relaxed)

        meta = dict(task["meta"])
        meta["energy_eV"] = e
        meta["delta_h_f_eV_per_atom"] = dhf
        Path(task["result_json"]).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        form_rows.append(
            {
                "frame": n,
                "task_id": task["task_id"],
                "formula": relaxed.get_chemical_formula(),
                "n_atoms": len(relaxed),
                "energy_eV": e,
                "delta_h_f_eV_per_atom": dhf,
                "energy_source": "cell_opt_result.json",
                "structure_ideal": task["structure"],
                "structure_relaxed": task["relaxed_path"],
            }
        )

        # SRO on relaxed geometry (chemical order + local distances)
        sro_payload = {
            "frame": n,
            "task_id": task["task_id"],
            "formula": relaxed.get_chemical_formula(),
            "warren_cowley": warren_cowley(
                relaxed, args.sro_cutoff, list(args.sro_shell_edges)
            ),
            "pair_correlation": pair_correlation(relaxed, args.sro_cutoff),
        }
        sro_path = out_dir / "sro" / f"{task['task_id']}.json"
        sro_path.write_text(json.dumps(sro_payload, indent=2), encoding="utf-8")

        # LLD: relaxed vs ideal (pre-relax SQS)
        norms, rmsd = displacements(relaxed, ideal)
        lld_payload = {
            "frame": n,
            "task_id": task["task_id"],
            "formula": relaxed.get_chemical_formula(),
            "bonds": bond_length_stats(relaxed, pairs, args.lld_cutoff),
            "rmsd_A": rmsd,
            "mean_abs_disp_A": float(norms.mean()),
            "local_strain": local_strain(norms, args.lld_a_ref),
            "ideal": task["structure"],
            "relaxed": task["relaxed_path"],
        }
        lld_path = out_dir / "lld" / f"{task['task_id']}.json"
        lld_path.write_text(json.dumps(lld_payload, indent=2), encoding="utf-8")

        summary_rows.append(
            {
                "frame": n,
                "task_id": task["task_id"],
                "formula": relaxed.get_chemical_formula(),
                "energy_eV": e,
                "delta_h_f_eV_per_atom": dhf,
                "lld_rmsd_A": rmsd,
                "lld_mean_abs_disp_A": float(norms.mean()),
                "lld_local_strain_mean": lld_payload["local_strain"]["mean"],
                "sro_file": str(sro_path),
                "lld_file": str(lld_path),
            }
        )
        if (n + 1) % 20 == 0 or n == 0:
            print(
                f"  [{n+1}/{len(tasks)}] {task['task_id']}  "
                f"E={e:.6f}  dHf={dhf:.6f} eV/atom  RMSD={rmsd:.5f} Å"
            )

    form_path = out_dir / "formation_enthalpy.json"
    form_path.write_text(json.dumps(form_rows, indent=2), encoding="utf-8")
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    write(out_dir / "all_relaxed.extxyz", relaxed_frames)

    dhf = np.array([r["delta_h_f_eV_per_atom"] for r in form_rows], dtype=float)
    rmsd = np.array([r["lld_rmsd_A"] for r in summary_rows], dtype=float)
    stats = {
        "n_frames": len(tasks),
        "cell_opt_dir": str(cell_opt_dir),
        "eref": str(args.eref.resolve()),
        "energy_reused": True,
        "delta_h_f_eV_per_atom": {
            "min": float(dhf.min()),
            "mean": float(dhf.mean()),
            "max": float(dhf.max()),
            "std": float(dhf.std()),
        },
        "lld_rmsd_A": {
            "min": float(rmsd.min()),
            "mean": float(rmsd.mean()),
            "max": float(rmsd.max()),
            "std": float(rmsd.std()),
        },
        "outputs": {
            "formation_enthalpy": str(form_path),
            "summary": str(summary_path),
            "sro_dir": str(out_dir / "sro"),
            "lld_dir": str(out_dir / "lld"),
            "all_relaxed": str(out_dir / "all_relaxed.extxyz"),
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
