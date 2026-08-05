# NaCl 18³ ladder — energy, AG/FD forces, timing, launch

**Status:** 2026-08-05 (W=1…12 clean analysis). Same 18×18×18 cell for all W. **W=1…36 PASS** (AG≡FD + energy/force vs W=1).

Primary artifacts: `pbs/out/parity_nacl_18/ladder_wigner_fix/` · mirror report: `pbs/out/parity_nacl_18/ladder_wigner_fix/REPORT_parity_timing.md`.

Interactive canvas (same W=1…12 tables/charts): [`docs/canvases/nacl18-ladder-parity.canvas.tsx`](canvases/nacl18-ladder-parity.canvas.tsx) — Cursor opens live canvases from its project `canvases/` folder; this path is the **git-tracked** copy.

---

## Clean analysis — W=1,2,4,6,12 (1-node)

**Verdict:** Energy and full AG forces match W=1 at every rung; spot AG≡FD **PASS** (tol `1e-5` eV/Å). Wall `elapsed_s` drops **~12.2×** W1→W12, but that timer is **not** warm `ef_mean` and does **not** split energy-only vs AG-only.

### Setup

| Item | Value |
|---|---|
| Cell | NaCl rocksalt **18×18×18** (46 656 atoms) |
| Geometry | `a=5.64` Å; `rattle=0.05`; `seed=0`; `sweep_ag_fd_nacl_n.build_nacl` |
| Model | `uma-cache/uma-s-1p2.pt`, **FP64**, task `omat` |
| API | W=1: `FAIRChemCalculator` + `make_calc`; W≥2: `uma_predict_unit(..., workers=W)` |
| Backend | W=1 none (single-tile); W=2…12 **xccl** (`ofi`/`FI_PROVIDER=tcp`), 1 node |
| Spot FD | atoms `0`, `23328` (`N/2`), `46655` (`N−1`) × `x,y,z`; `eps=1e-4` |
| Wigner | shim `apply_xpu_prepare_wigner_chunking()` required for AG≡FD |
| Jobs | W=1 `8728551`; W=2…12 `8728585` |
| PBS | `pbs/08_ladder_nacl18_W1_baseline.pbs`, `pbs/08_ladder_nacl18_W2_12.pbs` |

### Energy + AG force parity

| W | E (eV) | E/atom | Fmax AG | AG≡FD | max\|AG−FD\| | ΔE meV/atom vs W1 | max\|ΔF\| vs W1 | cos F vs W1 |
|--:|-------:|-------:|--------:|:-----:|-------------:|------------------:|---------------:|------------:|
| 1 | −157578.53111522 | −3.37745480 | 0.719101 | PASS | 2.204e−06 | — | — | — |
| 2 | −157578.53111522 | −3.37745480 | 0.719101 | PASS | 1.185e−06 | −1.248e−12 | 1.055e−15 | 1.00000000 |
| 4 | −157578.53111522 | −3.37745480 | 0.719101 | PASS | 5.609e−07 | −3.119e−12 | 9.714e−16 | 1.00000000 |
| 6 | −157578.53111522 | −3.37745480 | 0.719101 | PASS | 6.865e−07 | −3.119e−12 | 1.055e−15 | 1.00000000 |
| 12 | −157578.53111522 | −3.37745480 | 0.719101 | PASS | 3.669e−07 | 0 | 1.100e−15 | 1.00000000 |

- \|ΔE\|/N ≪ **1e−6 meV/atom** (policy energy bar).
- Force diffs vs W=1 are FP noise (~**1e−15** eV/Å); force cosine **1.0**.
- Frms AG = **0.135512** at every W.

### Wall timing (setup + E₀ + AG + 9×FD)

`elapsed_s` in `summary.json` = calc/Ray setup + E₀ + full AG forces + **9 spot FD energies**. Not inference-only; not split E vs AG.

| W | elapsed_s | vs W=1 | elapsed/W % |
|--:|----------:|-------:|------------:|
| 1 | 876.6 | 1.00× | 100% |
| 2 | 363.3 | 2.41× | 121% |
| 4 | 140.8 | 6.23× | 156% |
| 6 | 102.8 | 8.53× | 142% |
| 12 | 71.8 | **12.20×** | 102% |

`elapsed/W %` = `100 × (elapsed_W1 / elapsed_W) / W`. Values >100% are expected: FD energy evals also speed up with W — this is **not** warm-E+F parallel efficiency. Compare to true `ef_mean` ladders on 10³/12³ (~**2.8–3.0×** W1→12); see `docs/w12_nacl_xccl_recipe.md`.

### What is missing

- No per-rung **energy-only** or **AG-only** wall times on this ladder (`scripts/ladder_nacl18_ase_ag_fd.py` records a single `elapsed_s`).
- No dedicated warm `ef_mean` W=1…12 matrix for N=18.

Ancillary (different jobs — **do not mix into parity**):

| Source | Metric | Value | Note |
|---|---|---:|---|
| `pbs/out/sweep_nacl_w1_memory/n18` | single E+F | ≈19.5 s | W=1; not ladder `elapsed_s` |
| `pbs/out/w12_n_memory_sweep_16_40/n18` | warm `ef_mean` | ≈2.47 s | W=12; Phase1 on — E differs ~0.017 meV/atom from this ladder |

---

## Full ladder summary (includes multi-node)

| W | GPU tiles | nodes | backend | E (eV) | E/atom (eV) | Fmax AG | AG≡FD | max\|AG−FD\| | ΔE meV/atom vs W1 | max\|ΔF\| vs W1 | cos F vs W1 | elapsed_s | job |
|--:|--------:|------:|:--------|-------:|------------:|--------:|:-----:|-------------:|------------------:|---------------:|------------:|----------:|---:|
| 1 | 1 | 1 | none (single-tile) | -157578.53111522 | -3.37745480 | 0.719101 | PASS | 2.204e-06 | — | — | — | 876.6 | `8728551` |
| 2 | 2 | 1 | xccl | -157578.53111522 | -3.37745480 | 0.719101 | PASS | 1.185e-06 | -1.248e-12 | 1.055e-15 | 1.00000000 | 363.3 | `8728585` |
| 4 | 4 | 1 | xccl | -157578.53111522 | -3.37745480 | 0.719101 | PASS | 5.609e-07 | -3.119e-12 | 9.714e-16 | 1.00000000 | 140.8 | `8728585` |
| 6 | 6 | 1 | xccl | -157578.53111522 | -3.37745480 | 0.719101 | PASS | 6.865e-07 | -3.119e-12 | 1.055e-15 | 1.00000000 | 102.8 | `8728585` |
| 12 | 12 | 1 | xccl | -157578.53111522 | -3.37745480 | 0.719101 | PASS | 3.669e-07 | 0 | 1.100e-15 | 1.00000000 | 71.8 | `8728585` |
| 24 | 24 | 2 | gloo | -157578.53111522 | -3.37745480 | 0.719101 | PASS | 4.577e-07 | -1.871e-12 | 1.034e-15 | 1.00000000 | 2256.4 | `8728814` |
| 24 | 24 | 2 | xccl (ofi+tcp) | -157578.53111522 | -3.37745480 | 0.719101 | PASS | 4.142e-07 | -1.248e-12 | 1.258e-15 | 1.00000000 | 595.8 | `8729300` |
| 36 | 36 | 3 | xccl (ofi+tcp) | -157578.53111522 | -3.37745480 | 0.719101 | PASS | 4.449e-07 | -1.871e-12 | 9.888e-16 | 1.00000000 | 772.1 | `8729350` |

Notes:
- W=24 **gloo** multi-node PASS `8728814` (slow). W=24 **xccl** multi-node PASS `8729300` after `CCL_PROCESS_LAUNCHER=none` + PMI scrub.
- W=36 **xccl** multi-node PASS `8729350` (3 nodes × 12 tiles).
- Prefer **xccl** for W≥24 production; gloo W=24 kept for A/B.

## Per-atom AG vs FD forces (eV/Å)

### W=1 · 1 tiles · backend `none (single-tile)` · `w01/`

- **E** = `-157578.531115217105` eV
- **elapsed_s** = `876.62`
- **max\|AG−FD\|** = `2.204112e-06` → **PASS**
- **job** `8728551` · **PBS** `pbs/08_ladder_nacl18_W1_baseline.pbs`

| atom | comp | AG | FD | \|AG−FD\| |
|-----:|:----:|---:|---:|----------:|
| 0 | x | -0.0956220688 | -0.0956242729 | 2.2041e-06 |
| 0 | y | 0.1049932776 | 0.1049935236 | 2.4600e-07 |
| 0 | z | -0.0518941894 | -0.0518941670 | 2.2391e-08 |
| 23328 | x | -0.1006051885 | -0.1006042294 | 9.5912e-07 |
| 23328 | y | -0.0651122615 | -0.0651144364 | 2.1749e-06 |
| 23328 | z | -0.0223824547 | -0.0223843381 | 1.8834e-06 |
| 46655 | x | 0.0454811167 | 0.0454817200 | 6.0327e-07 |
| 46655 | y | 0.0443316125 | 0.0443305180 | 1.0945e-06 |
| 46655 | z | 0.1702213605 | 0.1702201553 | 1.2053e-06 |

### W=2 · 2 tiles · backend `xccl` · `w02/`

- **E** = `-157578.531115217163` eV
- **elapsed_s** = `363.29`
- **max\|AG−FD\|** = `1.185342e-06` → **PASS**
- **job** `8728585` · **PBS** `pbs/08_ladder_nacl18_W2_12.pbs`

| atom | comp | AG | FD | \|AG−FD\| |
|-----:|:----:|---:|---:|----------:|
| 0 | x | -0.0956220688 | -0.0956222357 | 1.6684e-07 |
| 0 | y | 0.1049932776 | 0.1049925049 | 7.7263e-07 |
| 0 | z | -0.0518941894 | -0.0518950401 | 8.5072e-07 |
| 23328 | x | -0.1006051885 | -0.1006049570 | 2.3153e-07 |
| 23328 | y | -0.0651122615 | -0.0651128357 | 5.7421e-07 |
| 23328 | z | -0.0223824547 | -0.0223823008 | 1.5384e-07 |
| 46655 | x | 0.0454811167 | 0.0454823021 | 1.1853e-06 |
| 46655 | y | 0.0443316125 | 0.0443319732 | 3.6065e-07 |
| 46655 | z | 0.1702213605 | 0.1702210284 | 3.3215e-07 |

### W=4 · 4 tiles · backend `xccl` · `w04/`

- **E** = `-157578.531115217251` eV
- **elapsed_s** = `140.80`
- **max\|AG−FD\|** = `5.608879e-07` → **PASS**
- **job** `8728585` · **PBS** `pbs/08_ladder_nacl18_W2_12.pbs`

| atom | comp | AG | FD | \|AG−FD\| |
|-----:|:----:|---:|---:|----------:|
| 0 | x | -0.0956220688 | -0.0956219446 | 1.2419e-07 |
| 0 | y | 0.1049932776 | 0.1049935236 | 2.4600e-07 |
| 0 | z | -0.0518941894 | -0.0518940215 | 1.6791e-07 |
| 23328 | x | -0.1006051885 | -0.1006055390 | 3.5055e-07 |
| 23328 | y | -0.0651122615 | -0.0651125447 | 2.8317e-07 |
| 23328 | z | -0.0223824547 | -0.0223827374 | 2.8272e-07 |
| 46655 | x | 0.0454811167 | 0.0454805559 | 5.6089e-07 |
| 46655 | y | 0.0443316125 | 0.0443318277 | 2.1513e-07 |
| 46655 | z | 0.1702213605 | 0.1702210284 | 3.3215e-07 |

### W=6 · 6 tiles · backend `xccl` · `w06/`

- **E** = `-157578.531115217251` eV
- **elapsed_s** = `102.82`
- **max\|AG−FD\|** = `6.864858e-07` → **PASS**
- **job** `8728585` · **PBS** `pbs/08_ladder_nacl18_W2_12.pbs`

| atom | comp | AG | FD | \|AG−FD\| |
|-----:|:----:|---:|---:|----------:|
| 0 | x | -0.0956220688 | -0.0956220902 | 2.1325e-08 |
| 0 | y | 0.1049932776 | 0.1049930870 | 1.9055e-07 |
| 0 | z | -0.0518941894 | -0.0518935849 | 6.0447e-07 |
| 23328 | x | -0.1006051885 | -0.1006053935 | 2.0503e-07 |
| 23328 | y | -0.0651122615 | -0.0651121081 | 1.5339e-07 |
| 23328 | z | -0.0223824547 | -0.0223827374 | 2.8272e-07 |
| 46655 | x | 0.0454811167 | 0.0454814290 | 3.1223e-07 |
| 46655 | y | 0.0443316125 | 0.0443315366 | 7.5905e-08 |
| 46655 | z | 0.1702213605 | 0.1702220470 | 6.8649e-07 |

### W=12 · 12 tiles · backend `xccl` · `w12/`

- **E** = `-157578.531115217105` eV
- **elapsed_s** = `71.85`
- **max\|AG−FD\|** = `3.669429e-07` → **PASS**
- **job** `8728585` · **PBS** `pbs/08_ladder_nacl18_W2_12.pbs`

| atom | comp | AG | FD | \|AG−FD\| |
|-----:|:----:|---:|---:|----------:|
| 0 | x | -0.0956220688 | -0.0956222357 | 1.6684e-07 |
| 0 | y | 0.1049932776 | 0.1049930870 | 1.9055e-07 |
| 0 | z | -0.0518941894 | -0.0518940215 | 1.6791e-07 |
| 23328 | x | -0.1006051885 | -0.1006052480 | 5.9511e-08 |
| 23328 | y | -0.0651122615 | -0.0651123992 | 1.3765e-07 |
| 23328 | z | -0.0223824547 | -0.0223827374 | 2.8272e-07 |
| 46655 | x | 0.0454811167 | 0.0454809924 | 1.2433e-07 |
| 46655 | y | 0.0443316125 | 0.0443312456 | 3.6694e-07 |
| 46655 | z | 0.1702213605 | 0.1702214649 | 1.0441e-07 |

### W=24 · 24 tiles · backend `gloo` · `w24/`

- **E** = `-157578.531115217193` eV
- **elapsed_s** = `2256.36`
- **max\|AG−FD\|** = `4.577461e-07` → **PASS**
- **job** `8728814` · **PBS** `FXPU_DIST_BACKEND=gloo + W24.pbs`

| atom | comp | AG | FD | \|AG−FD\| |
|-----:|:----:|---:|---:|----------:|
| 0 | x | -0.0956220688 | -0.0956220902 | 2.1325e-08 |
| 0 | y | 0.1049932776 | 0.1049933780 | 1.0048e-07 |
| 0 | z | -0.0518941894 | -0.0518943125 | 1.2313e-07 |
| 23328 | x | -0.1006051885 | -0.1006049570 | 2.3153e-07 |
| 23328 | y | -0.0651122615 | -0.0651118171 | 4.4443e-07 |
| 23328 | z | -0.0223824547 | -0.0223820098 | 4.4488e-07 |
| 46655 | x | 0.0454811167 | 0.0454815745 | 4.5775e-07 |
| 46655 | y | 0.0443316125 | 0.0443316821 | 6.9615e-08 |
| 46655 | z | 0.1702213605 | 0.1702213194 | 4.1110e-08 |

### W=24 · 24 tiles · backend `xccl (ofi+tcp)` · `w24_xccl_tcp_launcher_none/`

- **E** = `-157578.531115217163` eV
- **elapsed_s** = `595.81`
- **max\|AG−FD\|** = `4.141669e-07` → **PASS**
- **job** `8729300` · **PBS** `pbs/08_ladder_nacl18_W24.pbs (default xccl)`

| atom | comp | AG | FD | \|AG−FD\| |
|-----:|:----:|---:|---:|----------:|
| 0 | x | -0.0956220688 | -0.0956220902 | 2.1325e-08 |
| 0 | y | 0.1049932776 | 0.1049933780 | 1.0048e-07 |
| 0 | z | -0.0518941894 | -0.0518946035 | 4.1417e-07 |
| 23328 | x | -0.1006051885 | -0.1006052480 | 5.9511e-08 |
| 23328 | y | -0.0651122615 | -0.0651121081 | 1.5339e-07 |
| 23328 | z | -0.0223824547 | -0.0223825919 | 1.3720e-07 |
| 46655 | x | 0.0454811167 | 0.0454811379 | 2.1189e-08 |
| 46655 | y | 0.0443316125 | 0.0443315366 | 7.5905e-08 |
| 46655 | z | 0.1702213605 | 0.1702216105 | 2.4993e-07 |

### W=36 · 36 tiles · backend `xccl (ofi+tcp)` · `w36/`

- **E** = `-157578.531115217193` eV
- **elapsed_s** = `772.09`
- **max\|AG−FD\|** = `4.448806e-07` → **PASS**
- **job** `8729350` · **PBS** `pbs/08_ladder_nacl18_W36.pbs`

| atom | comp | AG | FD | \|AG−FD\| |
|-----:|:----:|---:|---:|----------:|
| 0 | x | -0.0956220688 | -0.0956219446 | 1.2419e-07 |
| 0 | y | 0.1049932776 | 0.1049935236 | 2.4600e-07 |
| 0 | z | -0.0518941894 | -0.0518938759 | 3.1343e-07 |
| 23328 | x | -0.1006051885 | -0.1006051025 | 8.6008e-08 |
| 23328 | y | -0.0651122615 | -0.0651123992 | 1.3765e-07 |
| 23328 | z | -0.0223824547 | -0.0223820098 | 4.4488e-07 |
| 46655 | x | 0.0454811167 | 0.0454809924 | 1.2433e-07 |
| 46655 | y | 0.0443316125 | 0.0443319732 | 3.6065e-07 |
| 46655 | z | 0.1702213605 | 0.1702216105 | 2.4993e-07 |

## How to launch (correct recipes)

Queues: **`debug`** (≤1 node) or **`debug-scaling`** (≥2 nodes) only.

```bash
cd /lus/flare/projects/MatSciAI/xiaoliyan/workdir/hen
```

### W=1 baseline (`debug`, 1 tile)

```bash
qsub pbs/08_ladder_nacl18_W1_baseline.pbs
# → pbs/out/parity_nacl_18/ladder_wigner_fix/w01/
```

```bash
export FXPU_NACL_N=18
export FXPU_FD_WORKERS=1
export FXPU_SKIP_EDGEWISE_PATCH=1
export FXPU_SKIP_EDEG_PATCH=1
# Do not set FXPU_SKIP_WIGNER_PREP_CHUNK — shim must apply Wigner chunking
export PYTHONPATH="$PWD/shim:$PWD:$PWD/sqs_sampling:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
```

### W=2,4,6,12 (`debug`, 1 node, xccl)

```bash
qsub pbs/08_ladder_nacl18_W2_12.pbs
# sequential W=2,4,6,12 → w02/ w04/ w06/ w12/
```

```bash
export FXPU_DIST_BACKEND=xccl
export FXPU_FI_PROVIDER=tcp FI_PROVIDER=tcp
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_WORKER_COUNT=1
export CCL_PROCESS_LAUNCHER=none   # required under Ray
unset CCL_KVS_MODE FI_TCP_IFACE
unset ZE_AFFINITY_MASK
export FXPU_PHASE3_LAUNCH=1
```

### W=24 (`debug-scaling`, 2×12 tiles) — recommended xccl

```bash
qsub pbs/08_ladder_nacl18_W24.pbs              # default: xccl
qsub pbs/08_ladder_nacl18_W24_xccl.pbs         # explicit A/B out dir
qsub -v FXPU_DIST_BACKEND=gloo pbs/08_ladder_nacl18_W24.pbs   # rollback
```

Multi-node XCCL **must** include:

```bash
export FXPU_DIST_BACKEND=xccl
export FXPU_PHASE6_MULTINODE=1
export FXPU_TILES_PER_NODE=12
export FXPU_FD_WORKERS=24
export FXPU_FI_PROVIDER=tcp FI_PROVIDER=tcp
export CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
export CCL_PROCESS_LAUNCHER=none
unset CCL_KVS_MODE FI_TCP_IFACE
unset PMI_RANK PMI_SIZE PMI_FD PMI_JOBID PALS_LOCAL_RANKID MPI_LOCALRANKID
# Do NOT export FI_TCP_IFACE from the head — each rank pins local hsn*
```

### W=36 (`debug-scaling`, 3×12 tiles) — PASS `8729350`

```bash
qsub pbs/08_ladder_nacl18_W36.pbs
# → w36/
```

Same multi-node XCCL env as W=24 with `FXPU_FD_WORKERS=36` and PBS `select=3`.

### Manual one-shot (inside an allocation)

```bash
source scripts/activate_fxpu.sh
export PYTHONPATH="$PWD/shim:$PWD:$PWD/sqs_sampling:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
unset ZE_AFFINITY_MASK
export FXPU_NACL_N=18
export FXPU_FD_WORKERS=12   # 1|2|4|6|12|24|36
export FXPU_BASELINE_DIR=pbs/out/parity_nacl_18/ladder_wigner_fix/w01
export FXPU_AG_OUT=pbs/out/parity_nacl_18/ladder_wigner_fix/w12
export FXPU_FD_EPS=1e-4 FXPU_FD_QUICK=1 FXPU_AG_FD_TOL=1e-5
export FXPU_SKIP_EDGEWISE_PATCH=1 FXPU_SKIP_EDEG_PATCH=1
export FXPU_PHASE3_LAUNCH=1
export FXPU_DIST_BACKEND=xccl CCL_PROCESS_LAUNCHER=none
export FXPU_FI_PROVIDER=tcp FI_PROVIDER=tcp CCL_ATL_TRANSPORT=ofi
export CCL_ZE_IPC_EXCHANGE=sockets
# if W>12:
export FXPU_PHASE6_MULTINODE=1 FXPU_TILES_PER_NODE=12
python -u scripts/ladder_nacl18_ase_ag_fd.py
```

## Artifacts & code

| Path | Role |
|---|---|
| `wXX/summary.json` | energy, spots, `elapsed_s`, `vs_w01` |
| `wXX/report.md` | AG/FD table |
| `wXX/forces_wXX.npy` | AG forces `(46656, 3)` |
| `LADDER.md` | compact climb |
| `scripts/ladder_nacl18_ase_ag_fd.py` | driver |
| `shim/fairchem_xpu_parallel.py` | affinity + phases |
| `patches/phase2_xccl.py` | XCCL / launcher=none / PMI scrub |
| `patches/phase6_multinode.py` | multi-node Ray |
| `patches/xpu_prepare_wigner.py` | AG≡FD Wigner fix |

