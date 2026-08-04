#!/bin/bash
# Source from PBS:  source "${PROJECT_ROOT}/scripts/activate_fxpu.sh"
# Activates the FairChem/UMA XPU conda env (symlink envs/fxpu → hen-xpu).
# Critical: env/lib MUST be first on LD_LIBRARY_PATH so conda libur
# matches conda libsycl (else ImportError urDeviceWaitExp vs module oneAPI).

# Compat: accept legacy HEN_* names for one transition.
if [[ -z "${FXPU_ENV:-}" && -n "${HEN_XPU_ENV:-}${HEN_XPU:-}" ]]; then
  FXPU_ENV="${HEN_XPU_ENV:-$HEN_XPU}"
fi
if [[ -z "${FXPU_PREFIX:-}" && -n "${FXPU_ENV:-}" ]]; then
  FXPU_PREFIX="${FXPU_ENV}"
fi

FXPU_PREFIX="${FXPU_PREFIX:-/lus/flare/projects/MatSciAI/xiaoliyan/software/conda/envs/fxpu}"
CONDA_ROOT="${CONDA_ROOT:-/lus/flare/projects/MOFA/xiaoliyan/software/miniforge3}"

if [ -r "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${FXPU_PREFIX}"
elif [ -x "${CONDA_ROOT}/bin/conda" ]; then
  eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"
  conda activate "${FXPU_PREFIX}"
else
  export PATH="${FXPU_PREFIX}/bin:${PATH}"
  export CONDA_PREFIX="${FXPU_PREFIX}"
  for f in "${FXPU_PREFIX}/etc/conda/activate.d"/*.sh; do
    [ -r "$f" ] && source "$f"
  done
  _rest=""
  IFS=':'
  for _p in ${LD_LIBRARY_PATH:-}; do
    [ -z "${_p}" ] && continue
    case "${_p}" in
      "${FXPU_PREFIX}/lib"|"${FXPU_PREFIX}/lib/"*) continue ;;
    esac
    if [ -z "${_rest}" ]; then _rest="${_p}"; else _rest="${_rest}:${_p}"; fi
  done
  unset IFS
  export LD_LIBRARY_PATH="${FXPU_PREFIX}/lib${_rest:+:${_rest}}"
  echo "WARN: activate_fxpu.sh PATH-only (+activate.d, env/lib FIRST)" >&2
fi

command -v python >/dev/null
python -c "import torch; print('torch_ok', torch.__version__)"
