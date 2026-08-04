#!/usr/bin/env python3
"""Compile preliminary results for one SQS evaluation campaign.

Joins cell_opt, postprocess (Hf / LLD / SRO), elastic, and multi-scheme EOS
on ``task_id`` and writes a combined table + short markdown report.

Example::

  python compile_preliminary_report.py \\
    --run-tag mc_sqs_20260730_032035 \\
    --workflow-out .../sqs_evaluation/workflow_out
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _chen_hv(B: float, G: float) -> tuple[float, float]:
    """Return (G/B, Hv_Chen_GPa). Chen et al., Intermetallics 19, 1275 (2011)."""
    k = G / B
    return k, float(2.0 * (k * k * G) ** 0.585 - 3.0)


def _mean_std(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": len(vals),
        "mean": float(st.mean(vals)),
        "std": float(st.pstdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def _fmt(ms: dict[str, float], fmt: str = ".4f") -> str:
    if not ms or ms.get("n", 0) == 0 or not math.isfinite(ms["mean"]):
        return "—"
    return f"{ms['mean']:{fmt}} ± {ms['std']:{fmt}}  [{ms['min']:{fmt}}, {ms['max']:{fmt}}]"


def compile_rows(workflow_out: Path, run_tag: str) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    cell_dir = workflow_out / f"cell_opt_{run_tag}"
    post_dir = workflow_out / f"post_{run_tag}"
    el_dir = workflow_out / f"elastic_{run_tag}"
    eos_dir = workflow_out / f"eos_{run_tag}"
    paths = {
        "cell_opt": cell_dir,
        "post": post_dir,
        "elastic": el_dir,
        "eos": eos_dir,
        "post_summary": post_dir / "summary.json",
        "elastic_summary": el_dir / "elastic_summary.json",
        "eos_summary": eos_dir / "eos_summary.json",
        "post_stats": post_dir / "stats.json",
    }
    for key in ("post_summary", "elastic_summary", "eos_summary"):
        if not paths[key].is_file():
            raise FileNotFoundError(paths[key])

    post = {r["task_id"]: r for r in _load(paths["post_summary"])}
    elastic = {r["task_id"]: r for r in _load(paths["elastic_summary"])}
    eos = {r["task_id"]: r for r in _load(paths["eos_summary"])}
    ids = sorted(set(post) | set(elastic) | set(eos))

    rows: list[dict[str, Any]] = []
    for tid in ids:
        p = post.get(tid, {})
        e = elastic.get(tid, {})
        o = eos.get(tid, {})
        cell_path = cell_dir / "tasks" / tid / "result.json"
        c = _load(cell_path) if cell_path.is_file() else {}

        row: dict[str, Any] = {
            "task_id": tid,
            "formula": p.get("formula") or e.get("formula") or o.get("formula") or c.get("formula"),
            "n_atoms": c.get("n_atoms"),
            "converged": c.get("converged"),
            "volume_A3": c.get("volume_A3"),
            "energy_eV": p.get("energy_eV", c.get("energy_eV")),
            "delta_h_f_eV_per_atom": p.get("delta_h_f_eV_per_atom", c.get("delta_h_f_eV_per_atom")),
            "lld_rmsd_A": p.get("lld_rmsd_A"),
            "lld_mean_abs_disp_A": p.get("lld_mean_abs_disp_A"),
            "lld_local_strain_mean": p.get("lld_local_strain_mean"),
            "C11_GPa": e.get("C11_GPa"),
            "C12_GPa": e.get("C12_GPa"),
            "C44_GPa": e.get("C44_GPa"),
            "elastic_B_GPa": e.get("B_GPa"),
            "elastic_G_GPa": e.get("G_GPa"),
            "elastic_E_GPa": e.get("E_GPa"),
            "elastic_nu": e.get("nu"),
            "G_over_B": e.get("G_over_B"),
            "Hv_Chen_GPa": e.get("Hv_Chen_GPa"),
            "eos_n_fits_ok": o.get("n_fits_ok"),
            "eos_n_fits": o.get("n_fits"),
            "has_post": tid in post,
            "has_elastic": tid in elastic,
            "has_eos": tid in eos,
            "has_cell_opt": bool(c),
        }
        b, g = row.get("elastic_B_GPa"), row.get("elastic_G_GPa")
        if isinstance(b, (int, float)) and isinstance(g, (int, float)) and b > 0 and g > 0:
            k, hv = _chen_hv(float(b), float(g))
            row["G_over_B"] = k
            row["Hv_Chen_GPa"] = hv
        for scheme, b_eos in (o.get("B_GPa_by_scheme") or {}).items():
            row[f"eos_B_GPa__{scheme}"] = b_eos
        for scheme, v0 in (o.get("V0_A3_by_scheme") or {}).items():
            row[f"eos_V0_A3__{scheme}"] = v0
        rows.append(row)
    return rows, paths


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    # Stable column order: core first, then sorted eos_* extras
    core = [
        "task_id",
        "formula",
        "n_atoms",
        "converged",
        "volume_A3",
        "energy_eV",
        "delta_h_f_eV_per_atom",
        "lld_rmsd_A",
        "lld_mean_abs_disp_A",
        "lld_local_strain_mean",
        "C11_GPa",
        "C12_GPa",
        "C44_GPa",
        "elastic_B_GPa",
        "elastic_G_GPa",
        "elastic_E_GPa",
        "elastic_nu",
        "G_over_B",
        "Hv_Chen_GPa",
        "eos_n_fits_ok",
        "eos_n_fits",
    ]
    extras = sorted(k for k in rows[0] if k not in core and not k.startswith("has_"))
    flags = sorted(k for k in rows[0] if k.startswith("has_"))
    fields = core + extras + flags
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def col(name: str) -> list[float]:
        return [float(r[name]) for r in rows if isinstance(r.get(name), (int, float))]

    stats: dict[str, Any] = {
        "n_rows": len(rows),
        "n_complete": sum(
            1
            for r in rows
            if r.get("has_post") and r.get("has_elastic") and r.get("has_eos") and r.get("has_cell_opt")
        ),
        "delta_h_f_eV_per_atom": _mean_std(col("delta_h_f_eV_per_atom")),
        "lld_rmsd_A": _mean_std(col("lld_rmsd_A")),
        "volume_A3": _mean_std(col("volume_A3")),
        "elastic": {
            "C11_GPa": _mean_std(col("C11_GPa")),
            "C12_GPa": _mean_std(col("C12_GPa")),
            "C44_GPa": _mean_std(col("C44_GPa")),
            "B_GPa": _mean_std(col("elastic_B_GPa")),
            "G_GPa": _mean_std(col("elastic_G_GPa")),
            "E_GPa": _mean_std(col("elastic_E_GPa")),
            "nu": _mean_std(col("elastic_nu")),
            "G_over_B": _mean_std(col("G_over_B")),
            "Hv_Chen_GPa": _mean_std(col("Hv_Chen_GPa")),
        },
        "eos_B_GPa": {},
        "eos_V0_A3": {},
    }
    schemes = sorted(
        {k.split("__", 1)[1] for r in rows for k in r if k.startswith("eos_B_GPa__")}
    )
    for s in schemes:
        stats["eos_B_GPa"][s] = _mean_std(col(f"eos_B_GPa__{s}"))
        stats["eos_V0_A3"][s] = _mean_std(col(f"eos_V0_A3__{s}"))
    return stats


def write_report(
    path: Path,
    *,
    run_tag: str,
    rows: list[dict[str, Any]],
    stats: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    el = stats["elastic"]
    lines = [
        f"# Preliminary SQS evaluation — `{run_tag}`",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        f"- Structures: **{stats['n_rows']}** (join key `task_id` = `tile_TT__sqs_SSS`)",
        f"- Complete (cell_opt + post + elastic + EOS): **{stats['n_complete']}/{stats['n_rows']}**",
        "- Energy model: UMA `uma-s-1p2` task `omat` via `hen-xpu` on Aurora XPU (12-tile pools)",
        "- Source SQS run: `sqs_sampling/runs/mc_sqs_20260730_032035`",
        "",
        "## Input directories",
        "",
        f"- cell_opt: `{paths['cell_opt']}`",
        f"- post: `{paths['post']}`",
        f"- elastic: `{paths['elastic']}`",
        f"- eos: `{paths['eos']}`",
        "",
        "## Thermodynamics / structure metrics",
        "",
        "| Quantity | mean ± std [min, max] |",
        "|---|---|",
        f"| ΔH_f (eV/atom) | {_fmt(stats['delta_h_f_eV_per_atom'], '.4f')} |",
        f"| LLD RMSD (Å) | {_fmt(stats['lld_rmsd_A'], '.4f')} |",
        f"| Relaxed volume (Å³) | {_fmt(stats['volume_A3'], '.2f')} |",
        "",
        "## Elastic constants (Voigt–Reuss–Hill)",
        "",
        "| Quantity | mean ± std [min, max] |",
        "|---|---|",
        f"| C11 (GPa) | {_fmt(el['C11_GPa'], '.2f')} |",
        f"| C12 (GPa) | {_fmt(el['C12_GPa'], '.2f')} |",
        f"| C44 (GPa) | {_fmt(el['C44_GPa'], '.2f')} |",
        f"| B (GPa) | {_fmt(el['B_GPa'], '.2f')} |",
        f"| G (GPa) | {_fmt(el['G_GPa'], '.2f')} |",
        f"| E (GPa) | {_fmt(el['E_GPa'], '.2f')} |",
        f"| ν | {_fmt(el['nu'], '.4f')} |",
        f"| G/B | {_fmt(el['G_over_B'], '.4f')} |",
        f"| H_V Chen (GPa) | {_fmt(el['Hv_Chen_GPa'], '.2f')} |",
        "",
        "Chen hardness: H_V = 2(k^2 G)^0.585 - 3, k = G/B "
        "(Chen et al., Intermetallics 19, 1275, 2011); derived from VRH B, G.",
        "",
        "## Equation of state (equal multi-scheme ASE fits)",
        "",
        "Isotropic E–V scan (scales 0.94–1.06), single-point UMA; all ASE schemes fit equally.",
        "",
        "| Scheme | ⟨B⟩ ± std (GPa) | ⟨V0⟩ ± std (Å³) |",
        "|---|---|---|",
    ]
    for scheme in stats["eos_B_GPa"]:
        b = stats["eos_B_GPa"][scheme]
        v = stats["eos_V0_A3"][scheme]
        lines.append(
            f"| `{scheme}` | {b['mean']:.2f} ± {b['std']:.2f} | {v['mean']:.2f} ± {v['std']:.2f} |"
        )
    lines += [
        "",
        "## Artifacts",
        "",
        "- `REPORT_preliminary_interactive.html` — Plotly interactive plots (open in a browser; needs CDN)",
        "- `combined_table.json` — one record per `task_id`",
        "- `combined_table.csv` — flat table (includes `Hv_Chen_GPa`, EOS `eos_B_GPa__*`)",
        "- `stats.json` — aggregate mean/std/min/max",
        "",
        "## Notes",
        "",
        "- Formation enthalpies reuse cell_opt `energy_eV` with elemental refs in `refs/uma/elemental_refs.json` (μ_N on `omat`).",
        "- Chen H_V from elastic VRH B, G (no extra XPU work).",
        "- SRO Warren–Cowley and full LLD bond histograms live under `post_*/sro/` and `post_*/lld/` (not flattened here).",
        "- This is a **preliminary** compile; DOS / DFT cross-checks not included.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-tag", default="mc_sqs_20260730_032035")
    ap.add_argument(
        "--workflow-out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "workflow_out",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: workflow_out/report_<run-tag>/",
    )
    args = ap.parse_args()
    workflow_out = args.workflow_out.resolve()
    out_dir = (args.out_dir or (workflow_out / f"report_{args.run_tag}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, paths = compile_rows(workflow_out, args.run_tag)
    stats = build_stats(rows)

    (out_dir / "combined_table.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    write_csv(out_dir / "combined_table.csv", rows)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_report(
        out_dir / "REPORT_preliminary.md",
        run_tag=args.run_tag,
        rows=rows,
        stats=stats,
        paths=paths,
    )
    print(f"wrote {out_dir}")
    print(f"  n_rows={stats['n_rows']} complete={stats['n_complete']}")
    print(f"  REPORT_preliminary.md  combined_table.{{json,csv}}  stats.json")


if __name__ == "__main__":
    main()
