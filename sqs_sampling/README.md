# SQS sampling (Rosenbluth CBMC + GFN2-xTB / FairChem UMA on Intel XPU)

Configurationally biased Monte Carlo on cation occupations. Each step proposes
`cbmc_trials` (default 10) concurrent swaps, evaluates them, selects a trial with
Rosenbluth weights, and accepts with \(\min(1, W_\mathrm{new}/W_\mathrm{old})\).

Intel XPU only for UMA (Aurora FLAT tiles). No NVIDIA / CUDA backends.

| `energy` | Evaluation |
|---|---|
| `gfn2-xtb` | Multiprocess Pool of single-thread TBLite workers (`map_async`) |
| `uma` | Shared FairChem UMA ASE calculator on one XPU tile (sequential trials) |

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

## UMA settings (FairChem / Intel XPU)

```yaml
energy: uma
device: xpu            # Intel GPU (FLAT tile); cpu allowed for debug only
dtype: float64         # required
uma_workers: 1         # CBMC: vanilla FAIRChemCalculator + MLIPPredictUnit (no Ray/GP)
uma_model: /lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen/uma-cache/uma-s-1p2.pt
uma_task: omat
```

Requires `fairchem-core` with **PyTorch XPU** wheels (see root `install.bash`). Only `uma-s-1p2.pt` is allowed.
Device `xpu` uses a small FXPU runtime patch because upstream FairChem asserts `cuda|cpu`
only — see [`docs/fairchem_xpu_fork.md`](../docs/fairchem_xpu_fork.md).

Before each UMA evaluation, `atoms.info["charge"]` is set from the formal M³⁺ / N³⁻
estimate and `atoms.info["spin"]=0` (spin off). Calibration uses the same `supercell` as SQS.

### Aurora 12-tile PBS (one node)

[`mc_sqs_uma_12tiles.pbs`](mc_sqs_uma_12tiles.pbs) launches **12 independent** single-tile
`mc_sqs.py` processes on one Aurora node (one FLAT tile each), with ALCF CPU/NUMA
binding (`numactl --physcpubind=… --membind=0|1`, `ZE_AFFINITY_MASK=0..11`).

```bash
qsub sqs_sampling/mc_sqs_uma_12tiles.pbs   # from hen/
# or: qsub mc_sqs_uma_12tiles.pbs          # from sqs_sampling/
# 100k-step capacity campaign: mc_sqs_uma_12tiles_100k.pbs
```

Config: [`config_uma_xpu_1tile.yaml`](config_uma_xpu_1tile.yaml) (`uma_workers: 1`, 3×3×3).

Each PBS submit creates a timestamped job tree; each rank gets its own tile folder:

```text
runs/mc_sqs_YYYYMMDD_HHMMSS/
  config_input.yaml
  tile_00/ … tile_11/     # --run-dir for each process
  tile_XX.log             # combined stdout+stderr
```

`mc_sqs.py` does **not** create the run folder; pass an existing path with `--run-dir`.

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

mkdir -p runs/my_run
python mc_sqs.py --config config.yaml --run-dir runs/my_run
# optional: --seed N

# single-point smoke test (one structure)
python run_gfn2_sqs.py --config config.yaml
python run_gfn2_sqs.py --config config.yaml --energy uma
```

Outputs written into `--run-dir`: `config.yaml`, `mc_sqs.log`, `mc_trajectory.extxyz`,
`sqs_XXX.extxyz`, and `sqs.extxyz`.

## Algorithm

1. Calibrate cubic \(a_i\) for MN end-members; set SQS \(a\) by Vegard.  
2. Build rocksalt supercell; occupy cations at target composition.  
3. Each CBMC step: propose 10 unlike-cation swaps → 10 energies (parallel GFN2 or sequential UMA).  
4. Pick trial \(i\) with probability \(w_i / W_\mathrm{new}\), \(w_i=e^{-\beta E_i}\).  
5. Reverse Rosenbluth weight \(W_\mathrm{old}\) from new state (old config + 9 swaps).  
6. Accept with \(\min(1, W_\mathrm{new}/W_\mathrm{old})\).  
7. Write sampled frames in CBMC step order.
