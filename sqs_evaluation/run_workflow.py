#!/usr/bin/env python3
"""HEN (High Entropy Nitride) structure evaluation workflow CLI.

Energy model (default: FairChem UMA) + property-oriented simulations on .extxyz.

Examples::

  python run_workflow.py --structures a.extxyz b.extxyz
  python run_workflow.py --config configs/default.yaml \\
      --structures runs/.../sqs.extxyz --properties sro,cell_opt,elastic
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from energy_models import build_energy_model, list_energy_models  # noqa: E402
from energy_models.base import ENERGY_DRIVEN, GEOMETRY_ONLY  # noqa: E402
from simulations.registry import (  # noqa: E402
    _load_frames,
    list_properties,
    run_property,
)


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        path = ROOT / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Structure evaluation: energy model + property simulations"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "default.yaml",
        help="YAML config (default: configs/default.yaml → UMA)",
    )
    parser.add_argument(
        "--structures",
        type=Path,
        nargs="+",
        default=None,
        help="One or more .extxyz files (multi-frame OK)",
    )
    parser.add_argument(
        "--properties",
        type=str,
        default=None,
        help="Comma list (default from config). "
        f"Available: {', '.join(list_properties())}",
    )
    parser.add_argument(
        "--energy-model",
        type=str,
        default=None,
        help=f"Override config energy.model. Available: {', '.join(list_energy_models())}",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List energy models and properties, then exit",
    )
    args = parser.parse_args()

    if args.list:
        print("energy models:", ", ".join(list_energy_models()))
        print("properties:", ", ".join(list_properties()))
        print("energy-driven:", ", ".join(sorted(ENERGY_DRIVEN)))
        print("geometry-only:", ", ".join(sorted(GEOMETRY_ONLY)))
        return

    if not args.structures:
        parser.error("--structures is required unless --list")

    cfg = _load_config(args.config)
    energy_cfg = dict(cfg.get("energy") or {})
    if args.energy_model:
        energy_cfg["model"] = args.energy_model

    prop_names = args.properties
    if prop_names is None:
        prop_names = cfg.get("properties") or [
            "sro",
            "cell_opt",
            "formation_enthalpy",
            "lld",
            "elastic",
        ]
    if isinstance(prop_names, str):
        prop_names = [p.strip() for p in prop_names.split(",") if p.strip()]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_root = args.out_dir or Path(
        cfg.get("out_dir") or f"workflow_out/eval_{stamp}"
    )
    out_root = out_root if out_root.is_absolute() else (Path.cwd() / out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    model = build_energy_model(energy_cfg)
    frames = _load_frames(args.structures)

    # Snapshot
    meta = {
        "created_utc": stamp,
        "structures": [str(p) for p in args.structures],
        "n_frames": len(frames),
        "energy": model.describe(),
        "properties": prop_names,
        "config": str(args.config),
    }
    (out_root / "workflow_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    (out_root / "config_used.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )

    print(f"energy_model={model.name}  frames={len(frames)}  out={out_root}")
    print(f"properties={prop_names}")

    prop_cfg = dict(cfg.get("property_options") or {})
    summary = []
    relaxed_frames = None

    for name in prop_names:
        key = name.strip().lower()
        pdir = out_root / key
        kwargs = dict(prop_cfg.get(key) or {})

        # Convenience: formation after cell_opt can reuse relaxed energies
        if key == "formation_enthalpy":
            if "elemental_eref" in kwargs and kwargs["elemental_eref"]:
                kwargs["elemental_eref"] = Path(kwargs["elemental_eref"])
            eref = kwargs.get("elemental_eref")
            if eref is None:
                print(f"skip {key}: set property_options.formation_enthalpy.elemental_eref")
                summary.append({"property": key, "status": "skipped_no_eref"})
                continue
            use_frames = relaxed_frames if relaxed_frames is not None else frames
            if relaxed_frames is not None:
                kwargs["energies"] = [
                    float(a.info.get("energy_eV", a.get_potential_energy()))
                    for a in relaxed_frames
                ]
            result = run_property(key, use_frames, model, out_dir=pdir, **kwargs)
        elif key == "lld":
            ideal = kwargs.get("ideal")
            if ideal is None:
                # default: use input (unrelaxed) as ideal vs relaxed if available
                use_relaxed = relaxed_frames if relaxed_frames is not None else frames
                kwargs["ideal"] = frames
                result = run_property(
                    key, use_relaxed, None, out_dir=pdir, **kwargs
                )
            else:
                kwargs["ideal"] = Path(ideal)
                use_relaxed = relaxed_frames if relaxed_frames is not None else frames
                result = run_property(key, use_relaxed, None, out_dir=pdir, **kwargs)
        elif key == "elastic" or key == "eos":
            use_frames = relaxed_frames if relaxed_frames is not None else frames
            result = run_property(key, use_frames, model, out_dir=pdir, **kwargs)
        elif key in GEOMETRY_ONLY:
            result = run_property(key, frames, None, out_dir=pdir, **kwargs)
        else:
            result = run_property(key, frames, model, out_dir=pdir, **kwargs)

        if key == "cell_opt":
            from ase.io import read as ase_read

            relaxed_frames = ase_read(pdir / "all_relaxed.extxyz", index=":")
            if not isinstance(relaxed_frames, list):
                relaxed_frames = [relaxed_frames]

        summary.append({"property": key, "status": "ok", "result": {
            k: v for k, v in result.items() if k != "results"
        }})
        print(f"  done {key} → {pdir}")

    (out_root / "workflow_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"wrote {out_root / 'workflow_summary.json'}")


if __name__ == "__main__":
    main()
