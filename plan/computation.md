# Computation Plan

Stack: **CP2K** (DFT), **ASE + MLIP**, **LAMMPS + MACE**. Target properties: [`properties.md`](properties.md).

SQS sampling: cubic end-member lattice calibration (TiN…TaN → Vegard \(a\)) then Rosenbluth CBMC in [`sqs_sampling/`](../sqs_sampling/) with **GFN2-xTB** or **UMA** → [`final_sqs/`](../sqs_sampling/final_sqs/).  
SQS evaluation: draft inputs in [`sqs_evaluation/`](../sqs_evaluation/).

| Simulation type | Energy method | Measurement in simulation | Related target properties | SQS template |
|---|---|---|---|---|
| Lattice-occupation / SRO analysis | None (config only) | Warren–Cowley \(\alpha_{ij}^{(n)}\); pair correlations \(P_{ij}(r)\) / \(g_{ij}(r)\) | §4 SRO | [`01_sro/`](../sqs_evaluation/01_sro/) |
| End-member ionic ± cell relaxation | CP2K DFT | Reference total energies \(E_i^{\mathrm{ref}}\) | §1 Mixing enthalpy (baseline) | [`02_endmember_cp2k/`](../sqs_evaluation/02_endmember_cp2k/) |
| Alloy ionic ± cell relaxation | ASE + MLIP | \(E(\sigma)\); \(\Delta H_\mathrm{mix}\) | §1 Total energy; mixing enthalpy | [`03_relax_ase_mlip/`](../sqs_evaluation/03_relax_ase_mlip/) |
| Alloy ionic ± cell relaxation | LAMMPS + MACE | \(E(\sigma)\); \(\Delta H_\mathrm{mix}\) | §1 Total energy; mixing enthalpy | [`04_relax_lammps_mace/`](../sqs_evaluation/04_relax_lammps_mace/) |
| Alloy relax / single-point validation | CP2K DFT | DFT \(E(\sigma)\), \(\Delta H_\mathrm{mix}\); MLIP vs DFT error | §1 Total energy; mixing enthalpy; uncertainty | [`05_validate_cp2k/`](../sqs_evaluation/05_validate_cp2k/) |
| Structure analysis on relaxed cells | None (ASE post-process) | Bond-length / NN histograms; \(\Delta\mathbf{r}_i\); RMSD; local strain | §3 LLD | [`06_lld/`](../sqs_evaluation/06_lld/) |
| Finite-strain stress–strain | ASE + MLIP | \(C_{11}\), \(C_{12}\), \(C_{44}\); VRH \(B\), \(G\); \(E\); \(\nu\) | §2 Elastic | [`07_elastic_ase_mlip/`](../sqs_evaluation/07_elastic_ase_mlip/) |
| Finite-strain stress–strain | LAMMPS + MACE | \(C_{11}\), \(C_{12}\), \(C_{44}\); VRH \(B\), \(G\); \(E\); \(\nu\) | §2 Elastic | [`08_elastic_lammps_mace/`](../sqs_evaluation/08_elastic_lammps_mace/) |
| Finite-strain stress–strain (subset) | CP2K DFT | DFT \(C_{ij}\), \(B\), \(G\), \(E\), \(\nu\); MLIP vs DFT \(\Delta C_{ij}\) | §2 Elastic; uncertainty | [`09_elastic_cp2k/`](../sqs_evaluation/09_elastic_cp2k/) |
