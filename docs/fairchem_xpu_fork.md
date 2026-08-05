# FairChem multi-XPU fork design (Aurora FLAT tiles)

Status: Phase-1 single-tile done; multi-tile XPU path prototyped in
[`sqs_sampling/fairchem_xpu_parallel.py`](../sqs_sampling/fairchem_xpu_parallel.py)
and exercised by [`scripts/scale_uma_nacl_tiles.py`](../scripts/scale_uma_nacl_tiles.py)
(`pbs/03_scale_nacl_tiles.pbs`).
Checkpoint for all current work: `uma-cache/uma-s-1p2.pt` only (not `uma-m-1p1.pt`).
Scope: **FP64** single-structure inference. Non-goal: batched multi-structure inference.

## Upstream blockers (must patch / fork)

FairChem 2.21 `MLIPPredictUnit._setup_device` hard-asserts `device in ["cpu", "cuda"]` and
`set_seed` calls `torch.cuda.manual_seed_all`. This stack currently monkeypatches these in
[`sqs_sampling/energy.py`](../sqs_sampling/energy.py) (`_patch_fairchem_for_xpu`) for
single-tile Phase-1. A proper fork should replace that assert with:

- `cpu` | `cuda` | `xpu` (and optional `xpu:N` under FLAT without affinity mask)
- seed via `torch.xpu.manual_seed_all` when XPU is active

FP64 is supported upstream via `InferenceSettings.base_precision_dtype=torch.float64`
(FXPU sets this; do not rely on post-hoc `.to(float64)` alone).

## Problem

Upstream FairChem large-system inference uses `get_predict_unit(..., device="cuda", workers=N)` with Ray to partition one large `Atoms` across N GPUs. That path assumes CUDA device enumeration and does not target Intel XPU / Level Zero affinity on Aurora.

Aurora node under **FLAT** ([docs](https://docs.alcf.anl.gov/aurora/)): 12 PVC tiles appear as `xpu:0`…`xpu:11`. Goal: one large structure forward across W ≤ 12 tiles on the **same node**.

## Proposed fork surface

Keep FairChem’s public shape; add an XPU backend:

```python
predictor = get_predict_unit(
    "/path/to/uma-s-1p2.pt",
    device="xpu",
    workers=W,                 # one Ray worker per tile
    dtype=torch.float64,       # required for FXPU tests
    inference_settings="default",  # no turbo / no batching for FP64 tests
)
calc = FAIRChemCalculator(predictor, task_name="omat")
```

### Worker binding (FLAT)

| Worker `i` | Env | Torch device |
|---|---|---|
| 0 … W-1 | `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, `ZE_AFFINITY_MASK=i` | `torch.device("xpu:0")` (logical 0 after mask) |

Driver process owns no tile (or tile 0 only for orchestration). Do not expose all 12 tiles to one process when `workers=W`.

### Internals to change in a fork

1. Replace `torch.cuda.*` device discovery with `torch.xpu.*`.
2. Ray worker init: set affinity env **before** importing torch in the worker.
3. Collectives: prefer FairChem’s existing host-side gather of energy/forces if present; otherwise use XCCL / oneCCL only where FairChem already requires GPU collectives.
4. Force `dtype=torch.float64` on model parameters and inference tensors; raise if any parameter remains non-FP64 after load.
5. Reject `batch=True` / multi-structure batch APIs in the FXPU test path.

### Validation ladder (PBS queues)

**Allowed queues only:** `debug` (1-node tests) and `debug-scaling` (≥2 nodes if ever needed).  
**Do not use** `prod`, `capacity`, `tiny`/`small`/`medium`/`large`, or any other queue.

1. W=1 vs W=2 on the same medium cell: energies agree within a tight FP64 tolerance.
2. W=6 and W=12 on one node (`debug`).
3. Multi-node Ray only if required → `debug-scaling` (≥2 nodes).

### Energy / force parity + detailed timing (20×20×20)

Script: [`scripts/parity_uma_nacl_tiles.py`](../scripts/parity_uma_nacl_tiles.py)  
PBS: [`pbs/04_parity_nacl_20.pbs`](../pbs/04_parity_nacl_20.pbs) → `pbs/out/parity_nacl_20/`

- **Ground truth:** ASE `FAIRChemCalculator` + `MLIPPredictUnit` on **1 FLAT tile** (`workers=1`; no Ray / no graph parallel).
- **Under test:** `workers` ∈ {1,2,3,4,6,8,12} on the same fixed NaCl 20×20×20 geometry (64000 atoms), FP64, `uma-s-1p2.pt`.
- **Parity:** total energy ΔE (atol/rtol) and per-atom forces (max|ΔF|, RMS, MAE, per-atom ‖ΔF‖₂); force arrays saved as `forces/forces_workers_XX.npy`.
- **Timing:** load, warmup E+F, timed energy-only, timed E+F (primary `uma_inference_total_s`), ms/atom, speedup vs GT; see `parity_timing_report.md`.

### Scaling timing documentation

`scripts/scale_uma_nacl_tiles.py` writes:

| Artifact | Content |
|---|---|
| `pbs/out/scale_nacl_10/scaling_summary.json` | Full results + `timing_table` |
| `pbs/out/scale_nacl_10/uma_inference_timing.tsv` | Compact table |
| `pbs/out/scale_nacl_10/uma_inference_timing.md` | Human-readable table |

**Primary metric:** `uma_inference_total_s` — sum of timed FairChem/UMA
`get_potential_energy` calls (with `torch.xpu.synchronize`) over `--repeats`.
Excludes model load and warmup. Also recorded: `uma_inference_mean_s`,
`load_s`, `warmup_inference_s`, `wall_total_s`.

Multi-tile note: with `FXPU_DIST_BACKEND=gloo`, FXPU stages graph-parallel
collectives via CPU (`fairchem_xpu_parallel._patch_gp_utils_gloo_xpu`) because
gloo cannot operate on XPU tensors.

### Scaling optimization roadmap

Phased plan (20×20×20 NaCl every phase; W=1 = vanilla FairChem, no multi-GPU):
[`uma_scaling_optimization_plan.md`](uma_scaling_optimization_plan.md).

### Project integration (later)

Optional YAML: `uma_workers: N`. CBMC remains sequential single-structure SPE (no trial batching on GPU).

## Out of scope

- Cloning ALCF `frameworks` env
- `uma-m-1p1.pt`
- COMPOSITE hierarchy
- Batched CBMC trial evaluation
