# Structure evaluation architecture

## Folders

| Path | Role |
|---|---|
| `energy_models/` | Backends that supply \(E\)/forces/stress/DOS (UMA default; MACE; GFN2; **CP2K**; Siesta) |
| `energy_models/cp2k/` | Input writers for the CP2K backends (not a simulation) |
| `simulations/` | Property-oriented simulations (`sro`, `cell_opt`, `elastic`, …) |
| `configs/` | Workflow YAML |
| `run_workflow.py` | CLI |

## Rule

**Code names the property or the energy model — never the DFT package as a simulation.**

Wrong (old): `simulations/cp2k_elastic/`  
Right: `simulations/elastic/` + `energy.model: cp2k_dft` (+ `energy_models/cp2k` for inputs)

## Energy models

| Name | Mode | DOS |
|---|---|---|
| `uma` | ASE | no |
| `mace` | ASE | no |
| `gfn2_tblite` | ASE | no |
| `gfn2_cp2k` | file-IO | no |
| `cp2k_dft` | file-IO | yes (target) |
| `siesta` | file-IO | yes (target) |

## Simulations

| Property | Notes |
|---|---|
| `sro`, `lld` | Geometry-only |
| `cell_opt`, `formation_enthalpy`, `elastic`, `eos` | Use configured energy model |
| `dos` | Requires DFT energy model |

## Extending

- New backend: `energy_models/<name>.py` + registry.
- New property: `simulations/<name>/` + `simulations/registry.py`.
- CP2K input details: only under `energy_models/cp2k/`.
