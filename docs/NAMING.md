# Naming: HEN vs FXPU

**HEN** = **High Entropy Nitride** — the materials project / workspace
(`…/workdir/hen`). Keep that directory name.

**FXPU** = FairChem/UMA **XPU infrastructure** (patches, env knobs, Ray workers,
PBS job scripts for scaling/parity). Do **not** brand this stack with `HEN`/`hen`.

| Kind | Old (retired) | New |
| --- | --- | --- |
| Env prefix | `HEN_*` | `FXPU_*` |
| Project root var | `HEN_ROOT` | `PROJECT_ROOT` |
| Conda activate | `activate_hen_xpu.sh` | `activate_fxpu.sh` (symlink kept) |
| Conda env | `…/envs/hen-xpu` | `…/envs/fxpu` → symlink to `hen-xpu` |
| Markers | `hen_xccl_native_v6`, … | `fxpu_xccl_native_v6`, … |
| Ray actors | `HenPhase3XPUMLIPWorker` | `FxpuPhase3XPUMLIPWorker` |
| PBS `-N` | `hen_nacl18_W24` | `nacl18_W24` (no `hen_` / `fxpu_` prefix) |

Legacy `HEN_*` exports still work via `scripts/fxpu_env_compat.py` and a few
fallbacks in `activate_fxpu.sh`. Prefer `FXPU_*` in new scripts.

Historical `pbs/out/**` and old PBS `*.o*` logs are left unchanged.
