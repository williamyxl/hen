# HEN — High-entropy nitride SQS pipeline

Monte Carlo special quasirandom structure (SQS) sampling and property evaluation for rocksalt high-entropy nitrides.

## Contents

| Path | Role |
|---|---|
| [`plan/`](plan/) | Target properties and computation plan |
| [`sqs_sampling/`](sqs_sampling/) | Rosenbluth CBMC SQS sampler (GFN2-xTB or UMA, 10 trials/step) |
| [`sqs_evaluation/`](sqs_evaluation/) | Numbered templates: SRO, relax, LLD, elastic, CP2K validation |
| [`docs/w12_nacl_xccl_recipe.md`](docs/w12_nacl_xccl_recipe.md) | **Final** NaCl W=12+ XCCL recipe (max N=32, entrypoints) |
| [`docs/multinode_xccl_n32_optimization_plan.md`](docs/multinode_xccl_n32_optimization_plan.md) | Multinode XCCL bottleneck / optimization plan (N=32) |
| [`docs/finding_xpu_ag_fd_cliff_n10.md`](docs/finding_xpu_ag_fd_cliff_n10.md) | XPU FP64 AG≠FD root cause (`prepare_wigner` einsum) + fix |

## Setup

```bash
# Aurora: FairChem/UMA XPU conda env (does NOT clone ALCF frameworks)
bash install.bash
# Prefer:
#   source scripts/activate_fxpu.sh
# Env path: .../conda/envs/fxpu → symlink to hen-xpu (physical env name kept)
```

Naming: **HEN** = this High Entropy Nitride workspace; FairChem/UMA infra uses **FXPU_*** — see [`docs/NAMING.md`](docs/NAMING.md).

UMA checkpoint for XPU tests: `uma-cache/uma-s-1p2.pt` only (not `uma-m-1p1.pt`).
PBS smoke jobs: `pbs/01_xpu_smoke.pbs`, `pbs/02_uma_fp64_spe.pbs` (`debug`, FLAT, `ZE_AFFINITY_MASK=0`).
Single-tile UMA CBMC: `pbs/06_mc_sqs_uma_1tile.pbs` → `sqs_sampling/mc_sqs.py` (`config_uma_xpu_1tile.yaml`).
Multi-tile FairChem design: [`docs/fairchem_xpu_fork.md`](docs/fairchem_xpu_fork.md).

## Quick start

```bash
cd sqs_sampling
cp config.example.yaml config.yaml
# edit composition / MC settings; device: xpu, dtype: float64, uma_workers: 1

python calibrate_lattice.py --config config.yaml   # cubic TiN…TaN → Vegard a
python mc_sqs.py --config config.yaml              # Rosenbluth CBMC (gfn2-xtb|uma)
python run_gfn2_sqs.py --config config.yaml        # single-point smoke test
python run_gfn2_sqs.py --config config.yaml --energy uma --device xpu
```



Selected structures are written under timestamped folders
`sqs_sampling/runs/mc_sqs_YYYYMMDD_HHMMSS/` (see [`sqs_sampling/README.md`](sqs_sampling/README.md)).
Evaluation steps and suggested order are documented in [`sqs_evaluation/README.md`](sqs_evaluation/README.md).

## Target properties

1. Total energy / mixing enthalpy / formation enthalpy  
2. Elastic constants and derived moduli  
3. Local lattice distortion  
4. Short-range order (Warren–Cowley)

See [`plan/properties.md`](plan/properties.md) and [`plan/computation.md`](plan/computation.md).
