# Finding: XPU FP64 AG≠FD at large edge count — `prepare_wigner` einsum

**Date:** 2026-08-02  
**Status:** Root cause pinned; faithful fix validated and wired into stock XPU path  
**Policy:** Full UMA only (no `FXPU_EDGEWISE_MAX_LAYERS` / `FXPU_DETACH_AFTER_MP`)

## One-line summary

On Aurora XPU, stock FairChem UMA-s-1p2 FP64 **energies are correct** but **autograd forces disagree with −∇E** once the neighbor graph is large (~≥2×10⁵ edges). The bad op is the first full-edge Wigner-preparation `einsum`; edge-chunking that contraction restores AG≡FD without changing the model.

## Symptoms

| Observation | Detail |
|-------------|--------|
| Size cliff | NaCl N=1…9 **PASS** AG≡FD; **N=10 first FAIL** (8000 atoms, 256002 edges) — job `8727547` |
| Energy | E/atom ~−3.377 eV, stable and bit-reproducible across jobs |
| FD | δ²E ~ 1e−8 → central-difference forces are the true −∇E |
| AG | Wrong by O(0.01–0.05); **nondeterministic across jobs** at fixed E/FD |
| Same bug class | NaCl 18³ W=1 (`8727507`): AG≠FD ~0.12, E OK |

Not caused by: maxL1 / architecture truncation, Phase1 SeqChunkAC, multi-tile GP, stress head, activation-checkpoint chunk size, or the 1→2 Edgewise AC-chunk transition (N=8 already has 2 chunks and PASSes).

## Root cause

FairChem `ExecutionBackend.prepare_wigner` (stock path):

```python
# FIRST einsum — THIS is the failure on large-edge XPU FP64
wigner = torch.einsum("mk,nkj->nmj", mappingReduced.to_m, wigner)

# SECOND einsum — NOT the failure
wigner_inv = torch.einsum("njk,mk->njm", wigner_inv, mappingReduced.to_m)
```

On Intel XPU FP64, reverse mode through the **first** contraction is wrong when the edge dimension `n` is large (stock N=10: `n=256002`). Forward values stay correct, so energy matches; the VJP into positions does not.

### Split proof (`8728108`)

| Mode | max\|AG−FD\| | Result |
|------|-------------:|--------|
| Chunk **first/forward** einsum only | 9.25e−9 | **PASS** |
| Chunk **second/inverse** einsum only | 0.055 | **FAIL** |

Trigger follows **edge count**, not atom count (`8727926`):

| Case | edges | max\|AG−FD\| |
|------|------:|-------------:|
| N=9, high edges | 307946 | 0.040 **FAIL** |
| N=10, low edges | 212392 | 1.3e−7 **PASS** |

## Faithful fix

**File:** `patches/xpu_prepare_wigner.py`  
**API:** `apply_xpu_prepare_wigner_chunking(chunk_size=65536, mode="both"|"forward"|"inverse")`

Replaces `ExecutionBackend.prepare_wigner` with the same two einsums, looped over edge chunks (default 65536). Identical forward math and full UMA architecture.

**Stock wiring:** `shim/fairchem_xpu_parallel.py` wraps `patch_fairchem_xpu_device()` so the chunking patch is applied automatically whenever the XPU device allowlist is installed (`hen/shim` first on `PYTHONPATH`).

| Env | Effect |
|-----|--------|
| (default) | Chunking **on** |
| `FXPU_SKIP_WIGNER_PREP_CHUNK=1` | Disable workaround (stock broken path) |
| `FXPU_WIGNER_PREP_CHUNK=N` | Chunk size (default 65536) |
| `FXPU_WIGNER_PREP_CHUNK_MODE=forward\|inverse\|both` | Which einsum(s) to chunk (default `both`) |

## Validation

| Job | Case | max\|AG−FD\| | Result |
|-----|------|-------------:|--------|
| `8727596` | Stock N=10 (9 comps) | 0.055 | FAIL |
| `8728017` | Chunk both einsums | 1.18e−7 | PASS |
| `8728108` | Forward-only / inverse-only | 9e−9 / 0.055 | PASS / FAIL |
| `8728042` | Permanent patch, N=10, 9 comps | 2.37e−7 | PASS |
| `8728088` | Permanent patch, N=18 atom0 Fx (ε=1e−3) | 4.58e−7 | PASS |
| **`8728165`** | **Shim auto-fix + stock sweep N=10** | **1.76e−7** | **PASS** |

After fix, atom0 Fx ≈ −0.000871 (matches FD). Before: AG ≈ −0.055.

## Ruled out (batteries)

| Hypothesis | Evidence | Verdict |
|------------|----------|---------|
| int32 index overflow | Indices already int64; force-int64 unchanged | Falsified |
| 2D `index_add` (E,128) | CPU≡XPU VJP at N=9/10 sizes (`8727661`) | Falsified |
| 3D `index_add` (E,9,128) | CPU≡XPU VJP (`8727677`) | Falsified |
| Edge→node scatter / `segment_reduce` swap | Same wrong AG (`8727661`) | Falsified |
| Safe `IndexAdd` gather backward | Unchanged FAIL (`8727677`) | Falsified |
| Node→edge gather safe backward | Unchanged FAIL | Falsified |
| Isolated `bmm` (E,9,9)@(E,9,128) | CPU≡XPU VJP | Falsified |
| AC chunk size / nostress | Identical AG (`8727596`) | Falsified |
| `use_deterministic_algorithms` | Blocked: no det. `index_reduce_xpu` | Inconclusive / unused |

## How to reproduce

```bash
# Broken baseline (opt out of fix)
export FXPU_SKIP_WIGNER_PREP_CHUNK=1
# Fixed (default when shim is on PYTHONPATH)
unset FXPU_SKIP_WIGNER_PREP_CHUNK

export ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_AFFINITY_MASK=0
export PYTHONPATH=hen/shim:hen/sqs_sampling:hen/scripts:$PYTHONPATH
# PBS: pbs/08_parity_nacl_n10_fix_gate.pbs  or
FXPU_N_MIN=10 FXPU_N_MAX=10 FXPU_RESUME=0 python -u scripts/sweep_ag_fd_nacl_n.py
```

## Artifacts

| Path | Role |
|------|------|
| `docs/finding_xpu_ag_fd_cliff_n10.md` | This document |
| `patches/xpu_prepare_wigner.py` | Fix |
| `shim/fairchem_xpu_parallel.py` | Auto-apply on XPU device patch |
| `pbs/out/parity_nacl_18/size_sweep_ag_fd/sweep.json` | N=1…10 cliff |
| `…/diag_n10_cases.json` | Stress / AC-chunk ablation |
| `…/pin_n10_root.json` | Scatter / int64 battery |
| `…/n10_wigner_chunk_{forward,inverse}_only.json` | Split pin |
| `…/gate_n10_wigner_fix.json` | Shim gate PASS |
| `…/n18_wigner_fix_spot.json` | N=18 **Fx-only** spot (insufficient; Fz can still fail) |
| `pbs/out/parity_nacl_18/ladder_wigner_fix/w01/` | ASE ladder W=1: **AG≠FD** max~0.12 (`8728189`) |
| `pbs/08_diag_n18_wigner_chunks.pbs` | N=18 full 9-spot × chunk-size probe |
| `pbs/08_parity_nacl_n10_fix_gate.pbs` | Regression gate |
| `scripts/sweep_ag_fd_nacl_n.py` | Size sweep / gate runner |

## Caveat / resolution (2026-08-02)

- N=10 AG≡FD with default chunk 65536: **confirmed**.
- N=18 **Fx-only** spot is insufficient as validation (can pass while Fz fails on another builder).
- Full 9-spot 18³ on **sweep `build_nacl`** + Wigner chunk 65536: **PASS** max|AG−FD|=1.9e−6 (`8728225`, `ladder_wigner_fix/diag_chunk/`).
- Ladder `8728189` AG≠FD and `8728249` `wigner_patch_live=False` were caused by
  `sys.path.insert(0, …)` in forward order leaving **`sqs_sampling` ahead of `shim`**, so the
  auto Wigner wrap never loaded. Fixed via `sys.path[:0] = [shim, …]`.

## Upstream

Report to PyTorch / Intel XPU: FP64 reverse mode of  
`einsum("mk,nkj->nmj", A[m,k], B[n,k,j])` for large `n` (~2×10⁵+) yields incorrect gradients while forward is correct. Workaround: chunk over `n`. Until fixed upstream, keep `apply_xpu_prepare_wigner_chunking()` on Aurora.
