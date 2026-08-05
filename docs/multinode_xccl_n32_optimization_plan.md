# Multinode XCCL optimization plan — NaCl 32³

**Status:** Phase C6 plateau — best warm multinode **C6e W=48 hier+pidfd+CHUNKS=4 @ ef≈16.29 s** (vs W12≈14.68 s; gap ~1.6 s). Inter ≈7.5 s/SPE for ~9.7 GiB (4 layer gathers) @ ~1.3 GB/s tcp;ofi_rxm.  
**Date:** 2026-08-04  
**Scope:** stock UMA (`uma-s-1p2.pt`), FP64 E+F, **NaCl 32×32×32 only** (262 144 atoms), **XCCL only**  
**Queues:** `debug` / `debug-scaling` only  
**Forbidden:** gloo, FP32/mixed/turbo/compile, architecture truncation, atom pad as “fix”, edits under `hen/sqs_sampling/`

Related:

- Production XCCL recipe: [`w12_nacl_xccl_recipe.md`](w12_nacl_xccl_recipe.md)
- Phase 2 runbook: [`phase2_xccl_runbook.md`](phase2_xccl_runbook.md)
- Baseline out: `pbs/out/scale_nacl_32_W12_96/`
- Planner subagent: [multinode XCCL plan](e18afd68-ff68-4023-9e83-192ebd96a079)

---

## 1. Problem statement

Same-node multi-tile acceleration works (N=18 W=1→12 ~12×). Fixed-N strong scaling across nodes does **not**: warm E+F **increases** with W because graph-parallel collectives (especially the XCCL-safe **broadcast-loop gather**) dominate.

This is a **multinode communication optimization** problem, not a UMA correctness or architecture-change problem.

---

## 2. Frozen baseline (do not reinterpret)

### Warm E+F — `pbs/out/scale_nacl_32_W12_96/`

| W | Nodes | ef_mean_s | coll_frac | AG_s (reported) | RS_s | AR_s (bucket) | load_s | status |
|--:|------:|----------:|----------:|----------------:|-----:|--------------:|-------:|:-------|
| 12 | 1 | 14.83 | 0.717 | 3.09 | 0 | **7.55** | 148.5 | ok |
| 24 | 2 | 36.58 | 0.699 | 25.54 | 0.017 | 0.022 | 322.1 | ok |
| 48 | 4 | 46.44 | 0.814 | 37.74 | 0.019 | 0.025 | 551.4 | ok |
| 96 | 8 | 88.84 | 0.897 | **79.66** | 0.029 | 0.038 | 1169.8 | ok (continuation) |

Parity vs W=12: |ΔE|/N ~ 1e−12 meV/atom; max|ΔF| ~ 1e−15 eV/Å; cos(F)≈1.

Artifacts: `workers_{12,24,48,96}.json`, `report.md`, `TIMING_ANALYSIS.md`, `forces/forces_workers_*.npy`.

**Caveats (measurement flaws to fix before claiming speedups):**

1. W=96 completed via `pbs/10_scale_nacl_32_W96_cont.pbs` after walltime kill — label as continuation.
2. No vanilla W=1 at N=32 (OOM) — speedups are **vs W=12 only**, never “vs vanilla W1”.
3. W=12 `all_reduce_s≈7.55 s` is almost certainly **`RS_MODE=auto` reduce-loop** (padded shard > 19000), mis-bucketed under `all_reduce` in `patches/phase2_xccl.py` `_reduce_scatter_safe` (`acc_fn("all_reduce", …)`).
4. Reported “all_gather” at multinode is the **broadcast loop**, not native `all_gather`.
5. `compute≈ = ef − coll_timers` is **not** pure model compute (includes graph build, sync, H2D, uninstrumented work).
6. Baseline used `repeats=1` — need ≥3 warm repeats for variance.
7. N=18 ladder `elapsed_s` includes FD spots + setup — **not** comparable to N=32 `ef_mean_s`.

Safety recipe in force (do not turn off without a gated A/B):

```text
FXPU_DIST_BACKEND=xccl
FXPU_XCCL_UNEVEN_GATHER=broadcast
FXPU_XCCL_RS_MODE=auto
FXPU_XCCL_RS_REDUCE_MIN=19000
FXPU_XCCL_PAD_ALIGN=32
FXPU_GP_PAD_ATOMS=0
FXPU_AUTO_GLOO_UNEVEN=0
FI_PROVIDER=tcp  CCL_ATL_TRANSPORT=ofi  CCL_ZE_IPC_EXCHANGE=sockets
CCL_WORKER_COUNT=1  CCL_PROCESS_LAUNCHER=none
FXPU_PHASE3_LAUNCH=1  FXPU_PHASE6_MULTINODE=1 (W>12)
```

---

## 3. Hypothesis tree (priority)

| ID | Hypothesis | Confidence | Primary evidence |
|----|------------|------------|------------------|
| **H1** | Broadcast-loop gather dominates multinode wall time | Very high | W=96: AG 79.7 s / 88.8 s; ~9 GiB payload; `phase2_xccl._bcast_all_gather` = W sequential `dist.broadcast` |
| **H2** | Timer taxonomy hides reduce-loop vs native RS | High | W=12 AR=7.55 s vs W≥24 RS~0.02 s; reduce-loop records as `all_reduce` |
| **H3** | Sync / no overlap inflates critical path | Med-high | Explicit waits in timed wrappers; harness `torch.xpu.synchronize()` after every E+F |
| **H4** | Ray + replicated model load dominates **cold** path | Very high | load 148→1170 s W=12→96; Phase3/6 launch + per-actor load |
| **H5** | Affinity / placement stragglers | Medium | Phase6 node order, `ZE_AFFINITY_MASK`, CPU pin |
| **H6** | Conservative oneCCL/TCP/sockets recipe costs BW | Med-high | Needed for correctness historically; cost unquantified |

---

## 4. Execution phases (sequential gates)

Do **not** start Phase N+1 until Phase N exit criteria pass. No gloo. No `sqs_sampling` edits.

### Phase A — Instrumentation (code, no science claim yet)

**Status:** timer taxonomy landed 2026-08-03 (broadcast / reduce_loop / reduce_scatter labels).

**Goal:** separate cold load / graph / model / sync / each collective algorithm.

Touch only (outside `sqs_sampling`):

| File | Change |
|------|--------|
| `patches/phase2_xccl.py` | Split timers: `broadcast_s`, `native_all_gather_s`, `native_reduce_scatter_s`, `reduce_loop_s`, `all_reduce_s`; per-call metadata (bytes, rank, node class local/remote) |
| `scripts/opt_scale_nacl_20.py` | Phase timers around load, warmup, timed E+F; ≥3 repeats default for MN study; keep `ef_mean_s` primary |
| `scripts/analyze_scale_timing.py` | Consume new buckets; stop calling residual “compute≈” model compute |
| `scripts/mn_n32_validate_env.py` | **New:** dump effective XCCL env + rank/node map sanity (login-safe; no FairChem compute) |

Exit criteria:

- JSON contains distinct broadcast vs native AG vs reduce-loop vs native RS fields.
- One dry parse of existing `workers_*.json` still works (missing new keys → “n/a”).
- `python3.10 -m py_compile` + `bash -n` on touched PBS clean.

### Phase B — Instrumented baseline matrix (N=32, XCCL)

| Step | W | Nodes | Queue | PBS (prep) | Out dir |
|------|--:|------:|-------|------------|---------|
| B0 | 12 | 1 | `debug` | `pbs/10_mn_n32_w12_inst.pbs` | `pbs/out/mn_n32_xccl/w12/` |
| B1 | 24 | 2 | `debug-scaling` | `pbs/10_mn_n32_w24_inst.pbs` | `pbs/out/mn_n32_xccl/w24/` |
| B2 | 48 | 4 | `debug-scaling` | `pbs/10_mn_n32_w48_inst.pbs` | `pbs/out/mn_n32_xccl/w48/` |
| B3 | 96 | 8 | `debug-scaling` | `pbs/10_mn_n32_w96_inst.pbs` | `pbs/out/mn_n32_xccl/w96/` |

Common args:

- `--nx 32 --ny 32 --nz 32 --rattle 0.05 --seed 0 --repeats 3 --phase1-patches --skip-vanilla-w1`
- W>12: `--ref-forces` from B0 `forces_workers_12.npy` + energy/ef from B0
- Walltime: W=12 `00:40:00`; W=24 `00:45:00`; W=48 `01:00:00`; W=96 `01:00:00` (load-heavy)

**Correctness gate (split by step):**

1. **Primary (every B step) — ALL-ATOM AG(W) vs W=12 AG** (262 144 atoms × 3):
   - Runtime AG forces from PyTorch (`∂E/∂pos`) at that W
   - `|ΔE|/N ≤ 1e-6` meV/atom
   - **every** component `|ΔF| ≤ 1e-10` eV/Å and every atom `‖ΔF‖₂ ≤ 1e-10`
   - `n_atom_fail == 0`, `n_comp_fail == 0`, cos(F)≈1
   - Artifacts: `FORCE_PARITY.md`, `force_parity.json`, `forces/forces_workers_XX.npy`,
     optional `per_atom_dF_XX.npy` / `delta_forces_XX.npy`, worst-atom table
2. **Secondary (B0 / W=12 only) — sparse AG≡FD** as a **stationary** energy-consistency
   check of the 1-node baseline:
   - Atoms with `index % 15016 == 0` (≈18 sites → 108 energy evals)
   - `max|AG−FD| ≤ 1e-5` eV/Å
   - **Do not re-run FD on multinode (B1–B3).** If AG(W)≡AG(W=12) all-atom and
     AG(W=12)≡FD(W=12) sparse, then AG(W) is consistent with that FD reference by
     transitivity. Recomputing FD at W≥24 is redundant and does not fit in the 1 h
     `debug-scaling` walltime cap.
3. status=ok, FP64 E+F; no `NotPresent`; no gloo demotion

Harness: `--spot-ag-fd` on **B0 only**; B1–B3 use
`force_parity_vs_ref.py --ref-dir … --dump-per-atom --dump-delta-forces` only.

**Per-step instrumentation gate:**

- ≥3 timed E+F samples (W=96 may use repeats=1)
- Gather path labeled `broadcast_s` (not only legacy all_gather)
- RS path labeled `reduce_scatter_s` or `reduce_loop_s`
- `FORCE_PARITY.md` present with vs-W=12 table; B0 also records sparse AG≡FD

### Phase C — Communication variants (one knob at a time)

Only after B0–B3 pass. Each variant: W=12 + W=24 minimum (then W=48/96 if W=24 improves).

| Order | Variant | Risk | Note |
|------:|---------|------|------|
| C1 | Current recipe (control redo) | Low | Confirm instrumented control |
| C2 | `FXPU_XCCL_RS_MODE=reduce_scatter` at W=12 | Med | Only if N=32 shard stable; parity hard gate |
| C3 | `FXPU_XCCL_UNEVEN_GATHER=native` | High | Known NotPresent history — abort on first segfault |
| C4 | Hierarchical gather (intra-node then inter-node) | Med | New code in `phase2_xccl.py`; must match broadcast math |
| C5 | OneCCL IPC/transport A/B (`sockets`→`drmfd`/`pidfd`, or `CCL_WORKER_COUNT`) | Med | One env change per job |
| C6 | Sync boundary cleanup (measure-only sync) | Med | Must not change force AD |

Exit: no correctness regression; warm `ef_mean_s` must drop on the targeted collective component.

### Phase D — Cold-path / orchestration (optional, separate KPI)

Persistent Ray worker pool / model reuse — does **not** fix H1 warm gather. Track separately as “cold SPE” KPI.

- **D0** (`FXPU_RAY_SKIP_PRESTOP=1`): warm OK (~16.30 s); **cold FAIL** — load 698 s vs C6e 332 s. Keep sequential prestop.
- **D1** (`FXPU_RAY_PARALLEL_BRINGUP=1`): **PASS / adopt** — load **44.4 s**, cold_1spe **82 s** vs C6e 332 / 376; warm 16.46≈C6e. Default on in `mn_n32_xccl_env.sh`.

---

## 5. Ranked optimizations (after evidence)

### Tier 1 — low/med risk, high impact

1. Fix timer taxonomy (required for honest claims).
2. Replace or hierarchicalize broadcast gather **if** native/hier path passes parity+stability.
3. Correct W=12 RS path accounting; try native RS when shard allows.
4. Measurement-boundary sync only.

### Tier 2 — medium risk

5. oneCCL IPC / worker-count / fabric A/B under parity gates.
6. Affinity / placement straggler audit (Phase6).
7. Ray reuse for cold latency.

### Tier 3 — out of scope unless project expands

Spatial halo DD, changing FairChem GP semantics, layer/cutoff changes — **forbidden** under STRICT UMA-immutable policy for this study.

---

## 6. Code map (inspect / later edit)

| Path | Role |
|------|------|
| `patches/phase2_xccl.py` | `_bcast_all_gather`, `_reduce_scatter_safe`, env setdefaults |
| `patches/phase3_launch.py` | Ray / placement / CPU pin |
| `patches/phase6_multinode.py` | Multi-node Ray + affinity |
| `shim/fairchem_xpu_parallel.py` | Routes xccl; env guards |
| `scripts/opt_scale_nacl_20.py` | Timing harness |
| `scripts/analyze_scale_timing.py` | Post-process report |
| `scripts/force_parity_vs_ref.py` | Per-atom F parity |
| `pbs/10_scale_nmax_W12_96.pbs` | Historical combined scale (1 h tight) |
| `pbs/10_mn_n32_w{12,24,48,96}_inst.pbs` | **Execution matrix** (this prep) |

Read-only: `hen/sqs_sampling/**`.

---

## 7. Execution checklist

```text
[x] Phase A: split collective timers + harness repeats=3 + analyze update
[x] Syntax-check subagent (py_compile + bash -n)
[x] B0–B3 instrumented matrix PASS
[x] Write mn_n32_xccl/REPORT_instrumented_baseline.md
[x] C2 native RS W=12 PASS (parity OK; ef≈flat)
[x] C4 hierarchical gather W=24/48/96 (W=48 best: 17.22s; need pidfd for W≥48)
[x] C5 IPC: drmfd FAIL hang; pidfd PASS (~1.06× on W=24; enables W=48)
[x] Write mn_n32_xccl/REPORT_phase_c.md
```

**Phase C closed.** Best warm multinode before C6 chunks: **W=48 hier+pidfd @ 17.22 s**.

**Phase C6 (2026-08-04):** locked recipe **hier + pidfd + `FXPU_XCCL_HIER_INTER_CHUNKS=4`** → **ef≈16.29 s** at W=48 (`c6_w48_chunks4`). Still ~11% slower than W=12 (14.68 s). Tried and rejected for warm win: async chunks, overlap sources, ring P2P (hang), recursive doubling (±chunks), W=36, RXM SAR tweak, skip `FI_TCP_IFACE` pin (hang), `hsn1` pin (≈flat), **C7 HSN stripe ± OVERLAP**, **C8 RXM dyn/direct**. Remaining gap is tcp inter bandwidth for 4× full activation gathers/SPE (~7.5 s); `cxi` under Ray remains unsafe. **Phase D1 adopted** for cold (`FXPU_RAY_PARALLEL_BRINGUP=1`, load 44 s vs 332). Warm stretch still open pending fabric or same-math gather-volume cut.

**Phase C9 (2026-08-04/05):** AC chunk sweep — **2097152 best** (`c9_w48_chunk2m`, ef≈**15.36 s**, job 8734229); 1M→15.40; 4M→15.74 (**slower**, job 8734245); 512k→15.59; 256k→15.84; C6e 16.29. ONE_CHUNK hung. Gap to W12 (14.68) ≈0.68 s. Adopted `FXPU_EDGEWISE_CHUNK_SIZE=2097152`. **AC chunk ladder closed** (peak at 2M).

**Phase C10 (2026-08-05):** same-math probes on C9f — scatter2M / BISECT=3 / GATHER_IN_AC / INTER_ASYNC / OVERLAP_SRC all no warm win (GATHER_IN_AC 21 s). Best remains C9f **15.36 s**. Stretch still open on hierarchical inter tcp BW.

**W=24-first (2026-08-05):** paused W=48 probing. Priority: 2-node W=24 warm ef &lt; W12 (14.68 s), then W=48, then W=96. Prior best W=24 C6z ≈19.36 s (no C9f AC).

---

## 8. Success definition

- **Science:** FP64 E+F, stock UMA, XCCL-only, cross-W parity gates green at N=32 for W=12/24/48/96.
- **Perf (minimum):** identify ≥70% of multinode `ef_mean` growth in a **named** collective algorithm with correct labels.
- **Perf (target):** W=24 warm `ef_mean` **below** current 36.6 s without correctness loss; stretch: W=24 ≤ 1.5× W=12 (≤22 s) while preserving recipe stability.
- **Perf (stretch, C6):** multinode warm `ef_mean` **below** W=12 (14.68 s) — **not yet met** (best 16.29 s).
