# SQS sampling (Monte Carlo)

Metropolis Monte Carlo on lattice occupations for HEN SQS candidates. Structures written as `.extxyz` to [`final_sqs/`](final_sqs/).

## Energy methods

| `--energy` | Backend | Notes |
|---|---|---|
| `gfn2-xtb` | tblite `TBLite(method="GFN2-xTB")` | Install: `pip install tblite` |
| `uma` | FAIRChem `FAIRChemCalculator` | Install: `pip install fairchem-core`; task `omat` for bulk inorganics |
| `mace` | `MACECalculator` | Install: `pip install mace-torch`; pass `--mace-model` |

Missing backends raise `ImportError`. Missing MACE model path raises `FileNotFoundError` / `ValueError`.

## Quick start

```bash
cp config.example.yaml config.yaml
# edit composition / MC settings

python mc_sqs.py --config config.yaml --energy gfn2-xtb
python mc_sqs.py --config config.yaml --energy uma --device cuda
python mc_sqs.py --config config.yaml --energy mace --mace-model /path/to/mace.model --device cuda
```

Outputs: `final_sqs/sqs_XXX.extxyz` and `final_sqs/sqs.extxyz`.

## Layout

| Path | Role |
|---|---|
| `config.example.yaml` | Required keys for composition, cell, MC, SRO ranking |
| `energy.py` | Calculator factory (`gfn2-xtb` / `uma` / `mace`) |
| `lattice.py` | Rocksalt supercell, swaps, Warren–Cowley score |
| `mc_sqs.py` | Monte Carlo driver |
| `final_sqs/` | Selected SQS `.extxyz` for [`sqs_evaluation/`](../sqs_evaluation/) |

## Algorithm

1. Build rocksalt supercell (or tagged template) with fixed anion sublattice.  
2. Occupy cation sites at exact target composition.  
3. Propose random swaps of two unlike cations.  
4. Evaluate energy with the chosen method (optional ionic relax; fails if not converged).  
5. Metropolis accept/reject at temperature `T`.  
6. Rank samples by mean \|Warren–Cowley α\| then energy; write `n_final` distinct occupations to `final_sqs/`.
