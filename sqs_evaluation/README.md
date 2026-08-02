# SQS / structure evaluation

Property-oriented evaluation of SQS `.extxyz` with a **pluggable energy model**.

**Default energy model: FairChem UMA** (`uma-s-1p2.pt`, Intel XPU).

```text
.extxyz ──► energy_models/ (uma | mace | gfn2_* | cp2k_dft | siesta)
                 │
                 ▼
         simulations/  (sro, cell_opt, formation_enthalpy, lld, elastic, eos, dos)
```

CP2K is an **energy model**, not a property. Its input writers live under
`energy_models/cp2k/`; elastic strains that also emit CP2K inputs are still the
`elastic` simulation (`simulations/elastic/make_cp2k_strains.py`).

## Layout

```text
sqs_evaluation/
  run_workflow.py
  configs/default.yaml
  energy_models/
    uma.py, mace.py, gfn2_*.py, cp2k_dft.py, siesta.py
    cp2k/                  # CP2K input helpers for cp2k_dft / gfn2_cp2k
  simulations/
    registry.py
    sro/, cell_opt/, formation_enthalpy/, lld/, elastic/, eos/, dos/
  docs/architecture.md
```

## Quick start (UMA default)

```bash
cd sqs_evaluation
python simulations/cell_opt/build_elemental_refs.py --energy uma --device xpu \
  --out elemental_refs.json
python run_workflow.py --structures path/to/sqs.extxyz --config configs/default.yaml
```

## Energy models vs simulations

| Energy model | Role |
|---|---|
| `uma` (default) | FairChem UMA ASE calculator |
| `mace` | MACE ASE calculator |
| `gfn2_tblite` / `gfn2_cp2k` | GFN2-xTB (TBLite or CP2K) |
| `cp2k_dft` | CP2K DFT (XC + basis + dispersion) |
| `siesta` | Siesta DFT |

| Simulation | Needs energy model? |
|---|---|
| `sro`, `lld` | No (geometry) |
| `cell_opt`, `formation_enthalpy`, `elastic`, `eos` | Yes |
| `dos` | DFT backends only |

See [`docs/architecture.md`](docs/architecture.md).
