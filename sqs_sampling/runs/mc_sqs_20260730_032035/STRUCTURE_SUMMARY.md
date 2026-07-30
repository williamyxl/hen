# Structure summary — `mc_sqs_20260730_032035`

12-tile production CBMC run (UMA / Intel XPU, 3×3×3 rocksalt HEN nitride).

## Run setup

| Item | Value |
|---|---|
| Supercell | 3×3×3 (216 atoms: 108 cations + 108 N) |
| Composition | equiatomic Ti/Zr/Hf/Nb/Ta → `Hf22N108Nb21Ta21Ti22Zr22` |
| Lattice | Vegard \(a = 4.484538\) Å (from calibration) |
| Sampler | Rosenbluth CBMC, `cbmc_trials=10`, \(T = 1200\) K |
| Length | `n_steps=500`, `equilibrate_steps=50`, `sample_every=50` |
| Samples / tile | 9 frames at MC steps 100, 150, …, 500 |
| Parallel ranks | 12 independent seeds 42–53 (one FLAT tile each) |
| Energy | FairChem UMA `uma-s-1p2.pt`, FP64, `uma_task=omat` |

## Structure counts

| Quantity | Count |
|---|---|
| **Total generated structures** (`tile_*/sqs_*.extxyz`) | **108** (12 × 9) |
| Exact unique occupations (no symmetry) | **108** |
| Exact duplicate groups | 0 |
| **Unique under cubic lattice symmetry** | **108** |

All 108 frames are distinct cation occupations. None are related by a cubic crystallographic symmetry of the rocksalt supercell.

### Symmetry criterion

Two structures are treated as equivalent if one cation coloring can be mapped onto the other by an element of the **cubic point group \(O_h\)** (48 proper + improper operations) combined with a **supercell translation that preserves the cation sublattice** (5184 automorphisms total). Anion (N) sites are held fixed. This is the natural discrete symmetry of fixed-lattice SQS on a cubic rocksalt cell.

Machine-readable details and per-cluster members: [`structure_uniqueness.json`](structure_uniqueness.json).

## Energy and SRO

| Metric | Min | Mean | Max | Std |
|---|---|---|---|---|
| UMA energy (eV) | −2239.0154 | −2238.3387 | −2237.7578 | 0.2099 |
| \(\lvert\alpha\rvert\) (SRO score) | 0.0624 | 0.1133 | 0.1798 | 0.0198 |

- Lowest energy: `tile_04/sqs_005.extxyz` — \(E = -2239.015417\) eV, \(\lvert\alpha\rvert = 0.1494\), MC step 350
- Lowest \(\lvert\alpha\rvert\): `tile_10/sqs_005.extxyz` — \(\lvert\alpha\rvert = 0.0624\), \(E = -2238.102261\) eV, MC step 350

## Per-tile overview

| Tile | Seed | \(E_\mathrm{min}\) (eV) | \(E_\mathrm{mean}\) (eV) | \(\lvert\alpha\rvert_\mathrm{min}\) | \(\lvert\alpha\rvert_\mathrm{mean}\) |
|---|---|---|---|---|---|
| 00 | 42 | −2238.5387 | −2238.3025 | 0.0743 | 0.1097 |
| 01 | 43 | −2238.5653 | −2238.4149 | 0.0959 | 0.1181 |
| 02 | 44 | −2238.4738 | −2238.3277 | 0.0855 | 0.1110 |
| 03 | 45 | −2238.5525 | −2238.2990 | 0.0740 | 0.1105 |
| 04 | 46 | −2239.0154 | −2238.4851 | 0.0956 | 0.1181 |
| 05 | 47 | −2238.7102 | −2238.2742 | 0.0762 | 0.1077 |
| 06 | 48 | −2238.6697 | −2238.3933 | 0.1071 | 0.1232 |
| 07 | 49 | −2238.6214 | −2238.2411 | 0.0722 | 0.1074 |
| 08 | 50 | −2238.5608 | −2238.2136 | 0.1083 | 0.1160 |
| 09 | 51 | −2238.5441 | −2238.3497 | 0.0981 | 0.1189 |
| 10 | 52 | −2238.7534 | −2238.4190 | 0.0624 | 0.1088 |
| 11 | 53 | −2238.5945 | −2238.3448 | 0.0846 | 0.1100 |

## Files

Each `tile_XX/` contains:

- `sqs_000.extxyz` … `sqs_008.extxyz` — individual sampled frames
- `sqs.extxyz` — concatenated samples (CBMC order)
- `mc_trajectory.extxyz` — same frames as trajectory
- `mc_sqs.log`, `config.yaml`

Combined rank stdout/stderr: `tile_XX.log` at the job root.
