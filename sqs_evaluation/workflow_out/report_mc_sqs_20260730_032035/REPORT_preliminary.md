# Preliminary SQS evaluation — `mc_sqs_20260730_032035`

Generated: 2026-08-04T02:35:07.896185+00:00

## Scope

- Structures: **108** (join key `task_id` = `tile_TT__sqs_SSS`)
- Complete (cell_opt + post + elastic + EOS): **108/108**
- Energy model: UMA `uma-s-1p2` task `omat` via `hen-xpu` on Aurora XPU (12-tile pools)
- Source SQS run: `sqs_sampling/runs/mc_sqs_20260730_032035`

## Input directories

- cell_opt: `/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/sqs_evaluation/workflow_out/cell_opt_mc_sqs_20260730_032035`
- post: `/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/sqs_evaluation/workflow_out/post_mc_sqs_20260730_032035`
- elastic: `/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/sqs_evaluation/workflow_out/elastic_mc_sqs_20260730_032035`
- eos: `/lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/sqs_evaluation/workflow_out/eos_mc_sqs_20260730_032035`

## Thermodynamics / structure metrics

| Quantity | mean ± std [min, max] |
|---|---|
| ΔH_f (eV/atom) | -1.4236 ± 0.0013  [-1.4271, -1.4203] |
| LLD RMSD (Å) | 0.1057 ± 0.0038  [0.0947, 0.1146] |
| Relaxed volume (Å³) | 2399.17 ± 0.76  [2397.21, 2400.79] |

## Elastic constants (Voigt–Reuss–Hill)

| Quantity | mean ± std [min, max] |
|---|---|
| C11 (GPa) | 538.51 ± 7.49  [520.29, 555.72] |
| C12 (GPa) | 139.15 ± 4.18  [128.70, 151.89] |
| C44 (GPa) | 111.52 ± 3.02  [97.99, 116.33] |
| B (GPa) | 272.27 ± 2.79  [264.54, 279.12] |
| G (GPa) | 141.09 ± 2.82  [128.62, 146.70] |
| E (GPa) | 360.91 ± 6.27  [333.79, 373.00] |
| ν | 0.2791 ± 0.0041  [0.2704, 0.2975] |

## Equation of state (equal multi-scheme ASE fits)

Isotropic E–V scan (scales 0.94–1.06), single-point UMA; all ASE schemes fit equally.

| Scheme | ⟨B⟩ ± std (GPa) | ⟨V0⟩ ± std (Å³) |
|---|---|---|
| `anton-schmidt` | 285.54 ± 0.17 | 2399.26 ± 0.76 |
| `birch` | 285.50 ± 0.17 | 2399.26 ± 0.76 |
| `birchmurnaghan` | 285.50 ± 0.17 | 2399.26 ± 0.76 |
| `murnaghan` | 285.03 ± 0.17 | 2399.27 ± 0.76 |
| `p3` | 288.55 ± 0.17 | 2399.24 ± 0.76 |
| `pouriertarantola` | 286.04 ± 0.17 | 2399.26 ± 0.76 |
| `sj` | 285.66 ± 0.17 | 2399.26 ± 0.76 |
| `taylor` | 288.55 ± 0.17 | 2399.24 ± 0.76 |
| `vinet` | 285.70 ± 0.17 | 2399.26 ± 0.76 |

Elastic VRH B (272.27 GPa) vs EOS birchmurnaghan B (285.50 GPa): Δ ≈ +13.22 GPa.

## Artifacts

- `combined_table.json` — one record per `task_id`
- `combined_table.csv` — flat table (EOS columns `eos_B_GPa__<scheme>`, `eos_V0_A3__<scheme>`)
- `stats.json` — aggregate mean/std/min/max

## Notes

- Formation enthalpies reuse cell_opt `energy_eV` with elemental refs in `refs/uma/elemental_refs.json` (μ_N on `omat`).
- SRO Warren–Cowley and full LLD bond histograms live under `post_*/sro/` and `post_*/lld/` (not flattened here).
- This is a **preliminary** compile; DOS / DFT cross-checks not included.
