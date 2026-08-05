# HEN — High-entropy nitride SQS pipeline

Monte Carlo special quasirandom structure (SQS) sampling and property evaluation for rocksalt high-entropy nitrides. Includes FairChem/UMA **FXPU** multi-tile XPU scaling on Aurora.

## Contents

| Path | Role |
|---|---|
| [`plan/`](plan/) | Target properties and computation plan |
| [`sqs_sampling/`](sqs_sampling/) | Rosenbluth CBMC SQS sampler (GFN2-xTB or UMA) |
| [`sqs_evaluation/`](sqs_evaluation/) | SRO, relax, LLD, elastic, CP2K validation |
| [`pbs/10_1node_ef_atoms.pbs`](pbs/10_1node_ef_atoms.pbs) | **1-node** multi-tile FP64 E+F launcher (env inline) |
| [`scripts/fxpu_1node_ef_atoms.py`](scripts/fxpu_1node_ef_atoms.py) | E+F for any ASE `Atoms`; `__main__` demo NaCl 18³ ladder |
| [`docs/w12_nacl_xccl_recipe.md`](docs/w12_nacl_xccl_recipe.md) | Final 1-node XCCL recipe (broadcast+sockets, max N=32 @ W=12) |
| [`docs/nacl18_ladder_parity_timing.md`](docs/nacl18_ladder_parity_timing.md) | NaCl 18³ W=1…12 energy / AG / timing analysis |
| [`docs/canvases/`](docs/canvases/) | Interactive scaling / parity canvases |
| [`patches/`](patches/) | FXPU Phase1–4,6 + Wigner-prep (stock UMA, no arch change) |
| [`shim/fairchem_xpu_parallel.py`](shim/fairchem_xpu_parallel.py) | Patch loader for FairChem XPU / Ray workers |
| [`docs/NAMING.md`](docs/NAMING.md) | HEN vs FXPU naming |
| [`docs/multinode_xccl_n32_optimization_plan.md`](docs/multinode_xccl_n32_optimization_plan.md) | Multinode XCCL plan (N=32, W&gt;12) |
| [`docs/finding_xpu_ag_fd_cliff_n10.md`](docs/finding_xpu_ag_fd_cliff_n10.md) | XPU AG≠FD root cause (`prepare_wigner`) + fix |
| [`docs/fairchem_xpu_fork.md`](docs/fairchem_xpu_fork.md) | Multi-tile FairChem / FXPU design notes |

## Setup

```bash
# Aurora: FairChem/UMA XPU conda env (does NOT clone ALCF frameworks)
bash install.bash
source scripts/activate_fxpu.sh
# Env: .../conda/envs/fxpu → symlink to hen-xpu
```

Naming: **HEN** = this High Entropy Nitride workspace; FairChem/UMA infra uses **`FXPU_*`** — see [`docs/NAMING.md`](docs/NAMING.md).

UMA checkpoint: `uma-cache/uma-s-1p2.pt` only (not `uma-m-1p1.pt`). Queues: **`debug`** / **`debug-scaling`** only.

## 1-node multi-tile E+F (recommended)

Validated NaCl **18³** W=1…12: energy/AG parity PASS, warm `ef_mean` **~9.6×** W1→W12 (job 8735819). See [`docs/w12_nacl_xccl_recipe.md`](docs/w12_nacl_xccl_recipe.md) and [`docs/canvases/nacl18-1node-recipe-ladder.canvas.tsx`](docs/canvases/nacl18-1node-recipe-ladder.canvas.tsx).

```bash
qsub pbs/10_1node_ef_atoms.pbs
# optional: qsub -v FXPU_LADDER=1,2,4,6,12,FXPU_NACL_N=18 pbs/10_1node_ef_atoms.pbs
```

```python
from fxpu_1node_ef_atoms import predict_ef
energy_eV, forces = predict_ef(atoms, workers=12)
```

1-node knobs (also embedded in the PBS): `FXPU_PHASE6_MULTINODE=0`, `FXPU_XCCL_UNEVEN_GATHER=broadcast`, `CCL_ZE_IPC_EXCHANGE=sockets`. Do **not** use multinode hierarchical/pidfd defaults on a single node.

## SQS quick start

```bash
cd sqs_sampling
cp config.example.yaml config.yaml
# edit composition / MC settings; device: xpu, dtype: float64, uma_workers: 1

python calibrate_lattice.py --config config.yaml   # cubic TiN…TaN → Vegard a
python mc_sqs.py --config config.yaml              # Rosenbluth CBMC (gfn2-xtb|uma)
python run_gfn2_sqs.py --config config.yaml        # single-point smoke test
python run_gfn2_sqs.py --config config.yaml --energy uma --device xpu
```

Selected structures land under `sqs_sampling/runs/mc_sqs_YYYYMMDD_HHMMSS/` (see [`sqs_sampling/README.md`](sqs_sampling/README.md)). Evaluation order: [`sqs_evaluation/README.md`](sqs_evaluation/README.md). Single-tile UMA CBMC config: [`sqs_sampling/config_uma_xpu_1tile.yaml`](sqs_sampling/config_uma_xpu_1tile.yaml).

## Target properties

1. Total energy / mixing enthalpy / formation enthalpy  
2. Elastic constants and derived moduli  
3. Local lattice distortion  
4. Short-range order (Warren–Cowley)

See [`plan/properties.md`](plan/properties.md) and [`plan/computation.md`](plan/computation.md).
