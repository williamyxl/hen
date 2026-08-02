# Computation Plan

Stack: pluggable **energy models** (default **FairChem UMA** on Intel XPU) plus
**property-oriented simulations** in [`sqs_evaluation/`](../sqs_evaluation/).

- Energy backends: [`sqs_evaluation/energy_models/`](../sqs_evaluation/energy_models/)
- Simulations: [`sqs_evaluation/simulations/`](../sqs_evaluation/simulations/)
- CLI: [`run_workflow.py`](../sqs_evaluation/run_workflow.py) + [`configs/default.yaml`](../sqs_evaluation/configs/default.yaml)

Optional DFT: CP2K (XC+basis+dispersion), Siesta; optional MLIP: MACE; optional GFN2 (TBLite or CP2K).
Target properties: [`properties.md`](properties.md).

SQS sampling: Vegard lattice calibration then Rosenbluth CBMC in [`sqs_sampling/`](../sqs_sampling/) (UMA default) → `runs/` or curated `final_sqs/`.

| Property simulation | Typical energy model | Measurements | Location |
|---|---|---|---|
| Short-range order | none (geometry) | Warren–Cowley \(\alpha\); pair histograms | `simulations/sro` |
| Cell optimization | UMA (default) | Relaxed geometry; \(E\) | `simulations/cell_opt` |
| Formation enthalpy | UMA + elemental refs | \(\Delta H_f\) / atom | `simulations/` + `cell_opt` helpers |
| Local lattice distortion | none (geometry) | RMSD; bonds; local strain | `simulations/lld` |
| Elastic tensor | UMA | \(C_{ij}\); VRH moduli | `simulations/elastic` |
| Equation of state | UMA | \(E(V)\); \(B\), \(V_0\) | `simulations/eos` |
| Density of states | CP2K DFT / Siesta | Electronic DOS | `simulations/dos` (scaffold) |
| DFT helpers | `cp2k_dft` energy model | Input writers | `energy_models/cp2k/` |
