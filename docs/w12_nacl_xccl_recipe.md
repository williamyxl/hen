# Final recipe: NaCl 1-node XCCL (FP64 E+F)

**Status:** production 1-node defaults (updated 2026-08-05)  
**Queues:** `debug` / `debug-scaling` only  
**Canonical 1-node cell for multinode work:** **N=32** (262 144 atoms) at **W=12**  
**Evidence baseline:** `pbs/out/mn_n32_xccl/w12/` (job **8731377**, `ef_mean≈14.68 s`)

> Multinode (W&gt;12) uses a **different** gather/IPC default (hierarchical + pidfd + inter chunks).  
> This document is **1-node only** (`FXPU_PHASE6_MULTINODE=0`).

## Bounds (1 node)

| | |
|---|---|
| **Max N @ W=12** | **32** (262 144 atoms, ~21 845/tile) |
| **OOM** | N≥33 @ W=12; **vanilla W=1 OOM at N=32** |
| Checkpoint | `uma-cache/uma-s-1p2.pt` only |
| Property | FP64 energy + forces |
| Speedups at N=32 | **vs W=12 only** (no vanilla W=1) |

Parity: N=21/22 W=12 vs W=24 all-atom forces PASS  
(`pbs/out/parity_nacl_21_22_W12_24/`). N=32 cross-W parity vs this W=12 AG reference.

## Env (1-node)

All knobs live inline in [`pbs/10_1node_ef_atoms.pbs`](../pbs/10_1node_ef_atoms.pbs)
(broadcast gather + sockets IPC; `FXPU_PHASE6_MULTINODE=0`).
Do **not** source [`scripts/mn_n32_xccl_env.sh`](../scripts/mn_n32_xccl_env.sh) on 1-node — it defaults to multinode hierarchical/pidfd.

### 1-node vs multinode (do not mix)

| Knob | 1-node (this recipe) | Multinode (W&gt;12) |
|------|----------------------|---------------------|
| `FXPU_PHASE6_MULTINODE` | `0` | `1` |
| `FXPU_XCCL_UNEVEN_GATHER` | `broadcast` | `hierarchical` |
| `CCL_ZE_IPC_EXCHANGE` | `sockets` | `pidfd` |
| Inter chunks / OVERLAP | n/a | see `mn_n32_xccl` checklist |

## What the fix does

| Path | Behavior |
|---|---|
| GP gather forward | XCCL **broadcast** loop (avoids native `all_gather` `NotPresent`) |
| SumGrad backward | **`auto`**: native `reduce_scatter` if pad&lt;19000; else **`dist.reduce` loop** |
| Equal vs uneven | Same path; no gloo demote; no atom padding |

**Do not use on 1-node:** `FXPU_GP_PAD_ATOMS=1`, `FXPU_AUTO_GLOO_UNEVEN=1`,
`FXPU_XCCL_UNEVEN_GATHER=native`, hierarchical gather (no benefit; different code path).

## Entrypoints

| Purpose | Script |
|---|---|
| **Minimal 1-node E+F** | `pbs/10_1node_ef_atoms.pbs` + `scripts/fxpu_1node_ef_atoms.py` |
| N=32 W=12 instrumented baseline (+AG≡FD) | `pbs/10_mn_n32_w12_inst.pbs` → `scripts/opt_scale_nacl_20.py` |
| Memory / OOM sweep W=12 | `pbs/10_w12_n_memory_sweep.pbs` |
| Strong scaling W=12…96 (multi-node) | `pbs/10_scale_nmax_W12_96.pbs` |

### Minimal launch

```bash
qsub pbs/10_1node_ef_atoms.pbs
qsub -v FXPU_WORKERS=6,FXPU_NACL_N=18 pbs/10_1node_ef_atoms.pbs
```

```python
from fxpu_1node_ef_atoms import predict_ef
energy_eV, forces = predict_ef(atoms, workers=12)
```

`FXPU_LADDER=1,2,4,6,12` (PBS default) runs a W climb with warm `ef_mean` + vs-W1 AG parity.

### Validated — NaCl 18³ ladder (job **8735819**)

Artifacts: `pbs/out/1node_ef_ladder_n18/` · canvas `docs/canvases/nacl18-1node-recipe-ladder.canvas.tsx`

| W | ef_mean (s) | vs W1 | ΔE meV/atom | max\|ΔF\| | cos |
|--:|------------:|------:|------------:|----------:|----:|
| 1 | 23.26 | 1.00× | — | — | — |
| 2 | 13.07 | 1.78× | ~3e−12 | ~1e−15 | 1.0 |
| 4 | 6.62 | 3.51× | ~2e−12 | ~1e−15 | 1.0 |
| 6 | 4.49 | 5.18× | ~5e−12 | ~1e−15 | 1.0 |
| 12 | **2.43** | **9.56×** | ~5e−12 | ~1e−15 | 1.0 |

E = −157578.53111522 eV, Fmax = 0.719101 at all W. Warm E+F ~**80%** parallel efficiency at W=12.

**Git-minimal:** `pbs/10_1node_ef_atoms.pbs` + `scripts/fxpu_1node_ef_atoms.py` + this recipe + canvases. Runtime still needs local `patches/`, `shim/`, `sqs_sampling/energy.py`, `uma-cache/uma-s-1p2.pt`.

## Strong scaling W=1,2,4,6,8,12 (inventory)

**Do we have a complete warm `ef_mean` ladder for every N?** No — coverage depends on cell size.

| Cell | Atoms | W=1,2,4,6,8,12? | Metric | Strong scaling? | Where |
|------|------:|:----------------:|--------|-----------------|-------|
| **10³** | 8 000 | **Yes** (also W=3) | warm `ef_mean` | Partial (~**2.8×** W1→12; plateaus ~W=8) | `pbs/out/scale_nacl_10/` |
| **12³** | 13 824 | **Yes** (also W=3) | warm `ef_mean` | Partial (~**3.0×** W1→12) | `pbs/out/parity_nacl_12/` + `docs/uma_timing_analysis.md` |
| **18³** | 46 656 | **W=1,2,4,6,12** (no W=8) | wall `elapsed_s` (E+AG+FD, **not** ef-only) | Parity **PASS**; wall ~**12.2×** W1→12 (fat timer, not `ef_mean`) | `docs/nacl18_ladder_parity_timing.md` · `pbs/out/parity_nacl_18/ladder_wigner_fix/` |
| **20³** | 64 000 | Partial ladders in phase outs; often **no W=1** in same run | phase gates | See `opt_nacl_20_phase*` | STRICT policy cell |
| **32³** | 262 144 | **No** — W=1 **OOM**; no W=2/4/6/8 1-node matrix | — | 1-node production point is **W=12 only** (`mn_n32_xccl/w12/`) | |

### Representative warm `ef_mean` (1-node XCCL)

**NaCl 10³** (`scale_nacl_10`):

| W | ef_mean (s) | vs W=1 |
|--:|------------:|-------:|
| 1 | 3.30 | 1.00× |
| 2 | 2.30 | 1.44× |
| 4 | 1.42 | 2.33× |
| 6 | 1.19 | 2.77× |
| 8 | 1.15 | 2.87× |
| 12 | 1.16 | 2.84× |

**NaCl 12³** (`parity_nacl_12`):

| W | ef_mean (s) | vs W=1 |
|--:|------------:|-------:|
| 1 | 5.63 | 1.00× |
| 2 | 3.88 | 1.45× |
| 4 | 2.64 | 2.13× |
| 6 | 2.01 | 2.80× |
| 8 | 1.90 | 2.97× |
| 12 | 1.86 | 3.04× |

**Takeaway:** Same-node multi-tile **helps** on mid-size cells but is **far from ideal linear** (~3× at W=12 on 10³/12³, not 12×). The multinode plan’s “N=18 ~12×” refers to a **fatter wall timer** (includes FD), not warm `ef_mean`. At **N=32** there is **no W=1…12 strong-scaling curve** — only the W=12 1-node anchor.

## Evidence (other)

| Result | Location |
|---|---|
| N=32 W=12 baseline (current) | `pbs/out/mn_n32_xccl/w12/` |
| N=16…40 sweep + OOM cliff | `pbs/out/w12_n_memory_sweep_16_40/` |
| Cliff fix (24/31) | `pbs/out/w12_n_fix_cliff_v6b/` |
| N=21/22 W=12 vs W=24 parity | `pbs/out/parity_nacl_21_22_W12_24/` |
| N=26 / N=32 W=12…96 (multi-node) | `pbs/out/scale_nacl_26_W12_96/`, `scale_nacl_32_W12_96/` |

## Code

| File | Role |
|---|---|
| `patches/phase2_xccl.py` | XCCL env, broadcast gather, auto SumGrad bwd |
| `shim/fairchem_xpu_parallel.py` | Routes xccl vs gloo; Phase3/6 launch |
| `patches/phase1_force_correctness.py` | Force AC (keep on) |

## Related

- Multinode N=32 plan: [`multinode_xccl_n32_optimization_plan.md`](multinode_xccl_n32_optimization_plan.md)
- Multinode checklist: `pbs/out/mn_n32_xccl/CHECKLIST.md`
- Base XCCL Ray rules: [`phase2_xccl_runbook.md`](phase2_xccl_runbook.md)
