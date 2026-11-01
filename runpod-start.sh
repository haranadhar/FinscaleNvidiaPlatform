#!/usr/bin/env bash
set -euo pipefail
export USE_TF=0 USE_FLAX=0
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "$HF_HOME" out
nvidia-smi
python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA PyTorch is required"; print("CUDA:",torch.version.cuda,"GPU:",torch.cuda.get_device_name(0))'
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
echo "Ready. Activate with: source .venv/bin/activate"
