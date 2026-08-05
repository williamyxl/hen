# HEN 16³ science campaign plan

**Status:** draft plan (not started)  
**Cell:** rocksalt \(N=16\) → **32 768 atoms** (8 atoms/conventional cell)  
**Why 16³:** single-tile UMA limit is ~18³; full-node (12-tile) limit ~32³. 16³ leaves VRAM margin on **one tile**, so vacancy/MD campaigns can run many independent tasks without requiring multi-tile collectives.  
**Energy model:** FairChem UMA `uma-s-1p2`, task `omat`, FP64, `hen-xpu`.  
**Scope:** materials science only — N-vacancy thermodynamics, extended SRO, finite-size bulk thermo, thermal MD, grain / vapor-deposition proxies at 16³ scale.

This plan is **independent** of `docs/uma_scaling_optimization_plan.md` (XCCL / multi-tile infra). Do not block science phases on XCCL phase gates.

---

## Goals (science questions)

| ID | Question |
|----|----------|
| Q1 | How does \(E_v^f(\mathrm{N})\) vary with local cation neighborhood in this HEN? |
| Q2 | Does SRO persist / decay beyond the shells accessible in 3³ SQS cells? |
| Q3 | Are small-SQS ensemble means for ΔH_f, V, LLD finite-size converged vs 16³? |
| Q4 | What do thermal MD give for MSD/diffusion, thermal expansion, and fluctuation bulk modulus? |
| Q5 | At ~7 nm (16³) scale, what can we learn about **nano-grain / GB** proxies and **vapor-deposition growth** of HEN films? |

---

## Shared setup (before Phase A)

### Host construction
- **Composition:** same target as production SQS  
  (`Ti/Zr/Hf/Nb/Ta` equiatomic cations, full N) → formula scale to 16 384 cation + 16 384 N sites.
- **Disorder:** build one (or few) **large SQS-like** cation occupations:
  - Preferred: tile/expand a well-equilibrated **3³ CBMC frame** (from `mc_sqs_100k_*` or `mc_sqs_20260730_032035`) by replication **only if** followed by a short large-cell cation MC / random reshuffle + SRO check — pure \(4\times\) replication of 3³ creates artificial long-period order.  
  - Better: generate **native 16³** occupation with CBMC or random + short MC at fixed lattice (even if only \(10^3\)–\(10^4\) steps — goal is host, not full SQS production).
- **Lattice:** Vegard \(a\) from existing calibration (`calibration/calibration.json`).
- **Relax:** Frechet cell + ions (UMA), same protocol as small-cell `cell_opt` (tighten `fmax` if needed for MD).
- **Deliverable:** `workflow_out/hen16/hosts/host_XX/{unrelaxed,relaxed}.extxyz` + energy/volume JSON.

### Compute pattern
- Default: **1 structure ↔ 1 XPU tile** (fits 16³).  
- Pool 12 independent hosts or 12 vacancy tasks per node (`debug` for smoke; long MD/vacancy campaigns may need `capacity` by explicit user exception, as with 100k CBMC).

### Exit criteria for setup
- [ ] ≥1 relaxed stoichiometric 16³ HEN host with documented composition and \(E\), \(V\).
- [ ] Smoke: single-point + 20-step MD on host completes without OOM on one tile.

---

## Phase A — Finite-size convergence of bulk thermo

**Depends on:** Shared setup  
**Science:** Q3

### Work
1. Take \(K\) relaxed 16³ hosts (\(K=1\) minimum; \(K=3\) preferred for host variance).
2. Compute on each (reuse small-cell postprocess where possible):
   - \(\Delta H_f\) / atom (same elemental μ as `refs/uma/elemental_refs.json`, μ_N on `omat`)
   - Relaxed \(V\), \(a_\mathrm{eff}\)
   - LLD RMSD vs ideal rocksalt
3. Compare to **3³ ensemble** stats (`report_mc_sqs_20260730_032035` / 100k subset when ready):
   - mean ΔH_f, V, LLD and std
   - report bias: \(\Delta = \langle X\rangle_{16^3} - \langle X\rangle_{3^3}\)

### Optional (same hosts)
- One Birch–Murnaghan EOS and/or cubic elastic set on **one** 16³ host as a finite-size reference (not a full 108-frame campaign).

### Exit criteria
- [ ] Table: 3³ ensemble vs 16³ host(s) for ΔH_f, V, LLD (± optional B).
- [ ] Written verdict: converged within tolerance (propose **|Δ(ΔH_f)| < 5 meV/atom**, **|ΔV/V| < 0.2%**) or “not converged — need larger K / better hosts.”
- [ ] Short note in plan status + path to results JSON/MD.

**Status:** not started

---

## Phase B — SRO beyond near shells

**Depends on:** Shared setup (same hosts as A; can run after or in parallel with A analysis)  
**Science:** Q2

### Work
1. Extend SRO analysis beyond current `sro_cutoff=3.5` Å / two shells:
   - Shell edges out to **≥ 2–3 conventional cells** (e.g. cutoffs ~6–9 Å; exact edges from RDF peaks of the relaxed host).
2. Report Warren–Cowley \(\alpha_{AB}(r)\) for metal–metal and metal–N as function of shell index.
3. Compare shell-1/2 \(\alpha\) to 3³ ensemble; quantify decay of |\(\alpha\)| with \(r\).
4. Optional: partial RDFs \(g_{AB}(r)\) for publication plots.

### Implementation notes
- Generalize `sqs_evaluation/simulations/sro/analyze_sro.py` (configurable shell edges / cutoff).
- For 33k atoms, neighbor lists at 9 Å are fine on CPU postprocess (login node).

### Exit criteria
- [ ] \(\alpha(r)\) / \(g(r)\) up to agreed cutoff for each host.
- [ ] Statement: SRO is short-ranged (decays by shell \(n\)) vs lingering order (finite-size / sampling artifact).
- [ ] Comparison panel vs 3³ near-shell α.

**Status:** not started

---

## Phase C — N-vacancy thermodynamics

**Depends on:** Phase A host(s) relaxed; B optional but useful for correlating \(E_v^f\) with local SRO  
**Science:** Q1

### Definition
\[
E_v^f = E(N_\mathrm{vac}) - E(\mathrm{perf}) + \mu_{\mathrm{N}}
\]
with \(\mu_{\mathrm{N}}=\tfrac12 E(\mathrm{N}_2)\) on **omat** (same as formation-enthalpy convention). Neutral vacancy; no charged-defect formalism in v1.

### Work
1. On each stoichiometric host, enumerate N sites (or a stratified subset):
   - **Stratified sampling (recommended):** bin N sites by first-neighbor cation fingerprint (counts of Ti/Zr/Hf/Nb/Ta); sample \(n\) sites per bin (e.g. high/median/low energy-density proxy, or random within bin). Target **≥ 50–100** vacancies per host initially.
2. For each site: remove N → ionic relax (cell fixed to host, or full Frechet — pick one and document; **fixed cell** is closer to dilute bulk vacancy).
3. Correlate \(E_v^f\) with:
   - NN cation composition
   - optional local energy density / LLD
4. Report distribution: mean, std, min/max, and linear model vs NN composition (DeSilva-style interpretability).

### Throughput (order of magnitude)
- 100 vacancies × (relax ~50–200 UMA steps) on 16³ is heavy but tile-parallelizable (12 vacancies/node).
- Smoke: 3 vacancies on one host before full campaign.

### Exit criteria
- [ ] \(E_v^f\) dataset JSON (site id, fingerprint, \(E_v^f\), relax metrics).
- [ ] Correlation / regression vs local chemistry.
- [ ] Compare qualitative ranking of metals (Hf-rich vs Mo-like neighbors) to DeSilva trends where species overlap.
- [ ] Note limitations: neutral only; MLIP vs DFT; fixed vs relaxed cell.

**Status:** not started

---

## Phase D — Thermal MD (NVT / \(NPE\)-like)

**Depends on:** Shared setup; ideally after A confirms host is physically sensible  
**Science:** Q4

### Ensemble
- **NVT** (Langevin or Nosé–Hoover): primary for MSD / diffusion.
- **\(N\!P\!T\) or barostatted \(NPE\)-like** (ASE `NPT` / Berendsen / MTK if available): thermal expansion + fluctuation bulk modulus.  
  (ASE naming: use hydrostatic barostat + thermostat; document exact class.)

### Protocol (per host)
1. Equilibrate 5–20 ps at target \(T\) (start **300 K** and **1200 K** — latter matches CBMC T).
2. Production ≥ 50–100 ps (extend if diffusivity is low).
3. Timestep: 1 fs (nitrides; verify energy drift in NVE smoke).
4. Outputs every 0.1–1 ps: positions, \(E\), \(P\)/\(\sigma\), \(V\).

### Analyses
| Observable | Method |
|------------|--------|
| MSD / diffusivity | Einstein relation for N and each metal (expect metals ≪ N mobility) |
| Thermal expansion | \(\langle V\rangle(T)\) from NPT; \(\alpha_V = (1/V)(\partial V/\partial T)\) between 300–1200 K |
| Fluctuation \(B_T\) | \(B_T = k_B T \langle V\rangle / \langle(\delta V)^2\rangle\) in NPT |
| Sanity | RDF stability; no melting/amorphization at 1200 K |

### Exit criteria
- [ ] Trajectories + summary JSON for each \(T\).
- [ ] MSD plots; diffusivities (or upper bounds if no diffusion).
- [ ] \(\alpha_V\) and fluctuation \(B_T\) with block-error estimates.
- [ ] Cross-check fluctuation \(B_T\) vs 0 K EOS/elastic \(B\) from Phase A (order-of-magnitude agreement expected, not identity).

**Status:** not started

---

## Phase E — Grain structure / vapor deposition (nano proxies)

**Depends on:** Setup + Phase A smoke (stable bulk/slab energetics); Phases C–D optional but useful for comparing bulk vs GB/surface vacancies and growth \(T\)  
**Science:** Q5

### Size reality check
- 16³ edge ≈ \(16 \times a \approx 7\,\mathrm{nm}\) (\(a\sim 4.5\,\text{Å}\)).
- Enough for **nanocrystalline / bicrystal GB patches** and **thin film growth on a slab**, not for experimental polycrystal grain-size distributions or reactor-scale PVD.
- Real HEN films grow mainly from **vapor → solid** (sputtering/CVD), not liquid quench — Phase E follows that physics.

### E1 — Grain / grain-boundary proxies
1. Build **bicrystal** HEN cells (same composition as bulk host):
   - Choose boundary plane, e.g. \(\Sigma5(210)\) or simpler **twist/tilt** on \(\{100\}\)/\(\{111\}\) (document Miller indices via plane normal in crystal frame).
   - Two misoriented grains + a GB slab; periodic GB pair or free surfaces far from the boundary.
2. Relax (ions ± cell as appropriate); report:
   - Grain-boundary energy \(\gamma_{\mathrm{GB}}\) (vs bulk reference from Phase A)
   - Excess volume, LLD, and **SRO / segregation** of cations toward the GB
3. Optional: N vacancy **at the GB** vs bulk (link to Phase C) — is \(E_v^f\) lower at the boundary?

### E2 — Vapor-deposition MD (film growth proxy)
1. Substrate: oriented HEN or binary nitride slab, e.g. \((100)\) or \((111)\), thermostatted bottom layers.
2. Deposit metal + N (or N₂) with:
   - flux / metal:N ratio (scan stoichiometric vs N-rich/poor)
   - kinetic energy ~1–few×10 eV (sputter-like) vs thermal (CVD-like)
   - substrate \(T\) (e.g. 300 K and 600–800 K)
3. Grow a few nm of film; analyze:
   - crystallinity vs amorphous (RDF / Steinhardt / coordination)
   - preferred orientation / texture hints
   - N stoichiometry and vacancy-like undercoordination
   - surface roughness / island vs layer growth (qualitative)

### Validation / caveats
- Validate short depositions: no energy blow-ups, sensible Na–Cl/HEN coordination; UMA may be less reliable for hyperthermal impacts than for bulk.
- Do **not** claim experimental grain size or coating hardness from E alone; treat as **mechanism / morphology proxies** complementary to bulk Phases A–D.

### Exit criteria
- [ ] ≥1 relaxed bicrystal (or documented GB geometry) with \(\gamma_{\mathrm{GB}}\), local chemistry vs bulk.
- [ ] ≥1 deposition-MD smoke (stoichiometric flux, one \(T\), one facet) completing without crash; RDF/coordination summary.
- [ ] Optional: matrix of facet × \(T\) × metal:N with crystallinity and N content trends.
- [ ] Explicit limits section: nano-only; not Hall–Petch grain-size series; not full PVD reactor.

**Status:** not started

---

## Recommended sequence

```text
Setup host(s) 16³
    ↓
Phase A  finite-size ΔH_f / V / LLD     ──┐
    ↓                                    ├── shared hosts
Phase B  extended SRO α(r), g(r)        ──┘
    ↓
Phase C  N-vacancy E_v^f campaign
    ↓
Phase D  thermal MD (can start after A smoke; full analysis after C or parallel if nodes allow)
    ↓
Phase E  grain / GB proxies + vapor-deposition MD
         (after A smoke; GB–vacancy link after C; growth T informed by D)
```

**Do not start Phase C at scale until** Setup + A smoke are closed.  
**Do not treat Phase D as closed** without block errors and a no-melt sanity check.  
**Do not start Phase E at scale until** Setup + A smoke are closed; treat full GB–vacancy comparison as blocked on Phase C.

---

## Deliverables (campaign-wide)

| Path (proposed) | Content |
|-----------------|--------|
| `hen/docs/hen16_science_campaign_plan.md` | This plan + status |
| `sqs_evaluation/workflow_out/hen16/` | hosts, sro, vacancies, md, gb, deposit, reports |
| `…/hen16/REPORT_hen16.md` | Final science summary tying Q1–Q5 |

---

## Out of scope (this campaign)

- Multi-tile XCCL performance / parity
- Experimental-scale polycrystal Hall–Petch series / reactor CFD
- Electronic DOS
- Charged vacancies / grand-canonical μ_N MC at scale
- Full elastic/EOS ensembles on 16³

---

## Immediate next actions

1. Close or snapshot **Setup**: decide host recipe (native 16³ MC vs expanded 3³ + reshuffle).
2. Implement `hen16` host builder + one-tile relax PBS smoke (`debug`).
3. Run Phase A on 1 host vs existing 3³ report numbers.
4. Only then scale vacancy list and MD lengths.

**When closing a phase:** update **Status** and add results path + one-line verdict in this file before opening the next.
