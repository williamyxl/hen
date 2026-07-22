# SQS sampling (Monte Carlo)

Metropolis Monte Carlo on lattice occupations for HEN SQS candidates. Structures written as `.extxyz` to [`final_sqs/`](final_sqs/).

## Energy: GFN2-xTB (default)

Primary sampling energy is **TBLite GFN2-xTB**:

| Setting | Value |
|---|---|
| SCC / SCF cycles | 2000 (`GFN2_MAX_ITERATIONS`) |
| `OMP_NUM_THREADS` | 1 |
| `MKL_NUM_THREADS` | 1 |
| `OMP_STACKSIZE` | 8G |

Also available: `uma`, `mace`.

## Quick start

```bash
cp config.example.yaml config.yaml
# edit composition / MC settings

# GFN2-xTB (default)
python mc_sqs.py --config config.yaml

# single-point smoke test on the initial occupation
python run_gfn2_sqs.py --config config.yaml

# other backends
python mc_sqs.py --config config.yaml --energy uma --device cuda
python mc_sqs.py --config config.yaml --energy mace --mace-model /path/to/mace.model --device cuda
```

Outputs: `final_sqs/sqs_XXX.extxyz` and `final_sqs/sqs.extxyz`.

## Layout

| Path | Role |
|---|---|
| `config.example.yaml` | Composition, cell, MC, SRO ranking (`energy: gfn2-xtb`) |
| `energy.py` | Calculator factory; GFN2 SCC/thread policy |
| `lattice.py` | Rocksalt supercell, swaps, Warren–Cowley score |
| `mc_sqs.py` | Monte Carlo driver (default GFN2-xTB) |
| `run_gfn2_sqs.py` | Single-point GFN2-xTB on initial SQS |
| `final_sqs/` | Selected SQS `.extxyz` for [`sqs_evaluation/`](../sqs_evaluation/) |

## Algorithm

1. Build rocksalt supercell (or tagged template) with fixed anion sublattice.  
2. Occupy cation sites at exact target composition.  
3. Propose random swaps of two unlike cations.  
4. Evaluate energy with GFN2-xTB (or chosen method; optional ionic LBFGS).  
5. Metropolis accept/reject at temperature `T`.  
6. Rank samples by mean \|Warren–Cowley α\| then energy; write `n_final` distinct occupations to `final_sqs/`.
