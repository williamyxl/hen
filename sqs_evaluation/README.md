# SQS evaluation templates

Inputs for a small set of SQS structures from [`sqs_sampling/final_sqs/`](../sqs_sampling/final_sqs/) (`.extxyz`).

Python drivers take CLI args (no silent defaults for calculators). Missing files, failed relaxations, and incomplete samples raise exceptions.

| Dir | Simulation | Energy method | Target properties |
|---|---|---|---|
| `01_sro/` | SRO analysis | None | §4 Warren–Cowley, pair correlations |
| `02_endmember_cp2k/` | End-member cell+ion relax | CP2K DFT | §1 Mixing enthalpy baseline |
| `03_relax_ase_mlip/` | Alloy ion±cell relax | ASE + MLIP (`--energy`) | §1 \(E\), \(\Delta H_\mathrm{mix}\) |
| `04_relax_lammps_mace/` | Alloy ion±cell relax | LAMMPS + MACE | §1 \(E\), \(\Delta H_\mathrm{mix}\) |
| `05_validate_cp2k/` | Alloy relax / SP validation | CP2K DFT | §1 DFT \(E\), \(\Delta H_\mathrm{mix}\) |
| `06_lld/` | LLD on relaxed `.extxyz` | None (ASE) | §3 bond/NN/RMSD/strain |
| `07_elastic_ase_mlip/` | Finite-strain elastic | ASE + MLIP (`--energy`) | §2 \(C_{ij}\), \(B\), \(G\), \(E\), \(\nu\) |
| `08_elastic_lammps_mace/` | Finite-strain elastic | LAMMPS + MACE | §2 \(C_{ij}\), \(B\), \(G\), \(E\), \(\nu\) |
| `09_elastic_cp2k/` | Finite-strain elastic | CP2K DFT | §2 DFT \(C_{ij}\) + derived |

CP2K inputs are generated from `.extxyz` via `02_endmember_cp2k/make_cp2k_input.py` (no hand-pasted COORD).

Suggested order: `01` → `03`/`04` → `06` → `07`/`08` → validate with `02`+`05`+`09`.
