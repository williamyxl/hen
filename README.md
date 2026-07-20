# HEN — High-entropy nitride SQS pipeline

Monte Carlo special quasirandom structure (SQS) sampling and property evaluation for rocksalt high-entropy nitrides.

## Contents

| Path | Role |
|---|---|
| [`plan/`](plan/) | Target properties and computation plan |
| [`sqs_sampling/`](sqs_sampling/) | Metropolis MC SQS generator (GFN2-xTB / UMA / MACE) |
| [`sqs_evaluation/`](sqs_evaluation/) | Numbered templates: SRO, relax, LLD, elastic, CP2K validation |

## Setup

```bash
bash install.bash
# then install energy backends as needed, e.g.:
#   pip install tblite          # gfn2-xtb
#   pip install fairchem-core   # uma
#   pip install mace-torch      # mace
```

## Quick start

```bash
cd sqs_sampling
cp config.example.yaml config.yaml
# edit composition / MC settings

python mc_sqs.py --config config.yaml --energy gfn2-xtb
```

Selected structures are written to `sqs_sampling/final_sqs/` as `.extxyz`. Evaluation steps and suggested order are documented in [`sqs_evaluation/README.md`](sqs_evaluation/README.md).

## Target properties

1. Total energy / mixing enthalpy  
2. Elastic constants and derived moduli  
3. Local lattice distortion  
4. Short-range order (Warren–Cowley)

See [`plan/properties.md`](plan/properties.md) and [`plan/computation.md`](plan/computation.md).
