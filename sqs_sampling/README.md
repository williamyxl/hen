# SQS sampling (Rosenbluth CBMC + GFN2-xTB / UMA)

Configurationally biased Monte Carlo on cation occupations. Each step proposes
`cbmc_trials` (default 10) concurrent swaps, evaluates them, selects a trial with
Rosenbluth weights, and accepts with \(\min(1, W_\mathrm{new}/W_\mathrm{old})\).

| `energy` | Evaluation |
|---|---|
| `gfn2-xtb` | Multiprocess Pool of single-thread TBLite workers (`map_async`) |
| `uma` | Shared FairChem UMA ASE calculator (sequential trials) |
| `nvalchemi-uma` | NVIDIA [`UMAWrapper`](https://github.com/NVIDIA/nvalchemi-toolkit) (uses [`nvalchemi-toolkit-ops`](https://github.com/NVIDIA/nvalchemi-toolkit-ops)); sequential by default, optional batching |

## GFN2-xTB settings

| Setting | Value |
|---|---|
| SCC / SCF cycles | see `GFN2_TBLITE` in `energy.py` |
| `OMP_NUM_THREADS` | 1 (per worker) |
| `MKL_NUM_THREADS` | 1 (per worker) |
| `OMP_STACKSIZE` | 4G |
| Concurrent workers | `cbmc_trials` (10) |
| Formal oxidation | M³⁺ / N³⁻ (`charge = 3 N_M - 3 N_N`) |
| Multiplicity | Ignored (closed-shell `multiplicity=1`, no spGFN) |

## UMA settings

Set in config (or CLI for the smoke test):

```yaml
energy: uma            # or nvalchemi-uma
device: cuda           # or cpu
uma_model: /mnt/d/workdir/uma-cache/uma-s-1p2.pt
uma_task: omat
inference_settings: default   # nvalchemi-uma only: default | turbo
nvalchemi_batch: false        # true = one forward for all CBMC trials (high VRAM)
```

Requires `fairchem-core`. For `nvalchemi-uma`, also install
`nvalchemi-toolkit-ops` and `nvalchemi-toolkit` (main branch has `UMAWrapper`).
`uma_model` may be a local `.pt` path or a pretrained name.

Before each UMA / nvalchemi-uma evaluation, `atoms.info["charge"]` is set from
the formal M³⁺ / N³⁻ estimate and `atoms.info["spin"]=0` (spin off). Calibration
uses the same `supercell` as SQS.

## Lattice calibration

Before SQS sampling, cubic rocksalt end-members **TiN, ZrN, HfN, NbN, TaN** are
volume-optimized (hydrostatic cell strain → cubic symmetry kept) with the same
`energy` backend. The SQS cell uses the Vegard average

\[
a = \sum_i x_i a_i
\]

```bash
python calibrate_lattice.py --config config.yaml
# or let mc_sqs.py run it when calibrate_lattice: true
```

Outputs: `calibration/{TiN,…}.extxyz`, `calibration/calibration.json`.

## Quick start

```bash
cp config.example.yaml config.yaml
python calibrate_lattice.py --config config.yaml
python mc_sqs.py --config config.yaml

# single-point smoke test (one structure)
python run_gfn2_sqs.py --config config.yaml
python run_gfn2_sqs.py --config config.yaml --energy uma
```

Outputs: `final_sqs/sqs_XXX.extxyz` and `final_sqs/sqs.extxyz` (CBMC step order), `mc_trajectory.extxyz`, `mc_sqs.log`.

## Algorithm

1. Calibrate cubic \(a_i\) for MN end-members; set SQS \(a\) by Vegard.  
2. Build rocksalt supercell; occupy cations at target composition.  
3. Each CBMC step: propose 10 unlike-cation swaps → 10 energies (parallel GFN2 or sequential UMA).  
4. Pick trial \(i\) with probability \(w_i / W_\mathrm{new}\), \(w_i=e^{-\beta E_i}\).  
5. Reverse Rosenbluth weight \(W_\mathrm{old}\) from new state (old config + 9 swaps).  
6. Accept with \(\min(1, W_\mathrm{new}/W_\mathrm{old})\).  
7. Rank samples by mean \|Warren–Cowley α\| then energy; write all sampled frames.
