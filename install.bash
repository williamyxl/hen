#!/bin/bash
# Fresh hen-xpu env on Aurora (Intel XPU). Does NOT clone ALCF frameworks.
set -euo pipefail

source /lus/flare/projects/MOFA/xiaoliyan/software/miniforge3/etc/profile.d/conda.sh

export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"

ENV_PREFIX="${HEN_XPU_ENV:-/lus/flare/projects/MatSciAI/xiaoliyan/software/conda/envs/hen-xpu}"
mkdir -p "$(dirname "$ENV_PREFIX")"

if [[ ! -d "$ENV_PREFIX" ]]; then
  conda create -p "$ENV_PREFIX" python=3.13 -y
fi

conda activate "$ENV_PREFIX"
python -m pip install -U pip setuptools wheel
python -m pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
# fairchem-core depends on CUDA torch by default; install deps manually and keep XPU torch
python -m pip install -U ase pyyaml huggingface-hub omegaconf hydra-core scipy tqdm monty e3nn
python -m pip install -U --no-deps fairchem-core

python - <<'PY'
import torch
print("torch", torch.__version__, "xpu_attr", hasattr(torch, "xpu"))
assert "+xpu" in torch.__version__ or hasattr(torch, "xpu"), torch.__version__
import ase, fairchem.core
print("ase", ase.__version__, "fairchem ok")
PY

echo "Activate with:"
echo "  source /lus/flare/projects/MOFA/xiaoliyan/software/miniforge3/etc/profile.d/conda.sh"
echo "  conda activate $ENV_PREFIX"
