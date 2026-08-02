# Target Properties

## 1. Total energy / Mixing enthalpy / Formation enthalpy

Primary metric.

### Measurements

- Total energy
- Mixing enthalpy (vs rocksalt MN end-members)
- Formation enthalpy (vs elemental metals + ½ N₂)
- Distribution of energies
- Mean ± standard deviation
- Convergence vs number of sampled configurations

### Purpose

- Measures thermodynamic stability
- Demonstrates ensemble convergence
- Shows sampling efficiency

Formation enthalpy uses elemental references from
[`sqs_evaluation/simulations/cell_opt/build_elemental_refs.py`](../sqs_evaluation/simulations/cell_opt/build_elemental_refs.py)
(μ_M from bulk metal; μ_N = E(N₂)/2). At fixed composition, ranking by
\(E\), \(\Delta H_\mathrm{mix}\), or \(\Delta H_f\) differs only by a constant.
## 2. Elastic properties

Compute the full elastic tensor.

### Report

- C₁₁
- C₁₂
- C₄₄

### Derived properties

- Bulk modulus
- Shear modulus
- Young's modulus
- Poisson ratio

### Ensemble statistics

- Property distribution
- Ensemble average
- Uncertainty
- Convergence

Strongest engineering validation.

## 3. Local lattice distortion

Structural descriptor.

### Examples

- Bond-length distribution
- Nearest-neighbor distance distribution
- Atomic displacement after relaxation
- RMS displacement
- Local strain distribution

### Purpose

- Explains why energies differ
- Connects configuration to mechanics
- Inexpensive to compute

## 4. Short-range order (SRO)

### Measure

- Warren–Cowley parameters
- Pair correlation functions

### Purpose

- Characterize sampled configurations
- Quantify how the sampler differs from SQS
- Demonstrate configurational diversity
