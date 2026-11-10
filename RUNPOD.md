# RunPod — FinScale Nemotron 30B

Provision a RunPod GPU Pod with CUDA-compatible PyTorch/NeMo image and **2 × 80GB GPU** or better for NVIDIA's documented Nemotron 3 Nano 30B LoRA topology. A100/H100 80GB are hardware examples, not a promise of matching SKUs, availability or tested image compatibility. Verify training recipe's Slurm/profile requirements before spending GPU hours. Choose enough persistent network volume for the checkpoint (BF16 base weights alone are roughly 63 GB), caches, filings and output; verify actual mounts.

Use a mounted persistent `/workspace` if you configured one, then:

```bash
nvidia-smi
df -h /workspace
cd /workspace
git clone --branch Nemotron --single-branch https://github.com/haranadhar/FinscaleNvidiaPlatform.git
cd FinscaleNvidiaPlatform
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/smoke_test_cuda.py
export SEC_USER_AGENT="Your Name your-email@example.com"
python scripts/download_edgar.py --limit 1
python scripts/build_sft_dataset.py --no-api
wc -l data/processed/{train,val}.jsonl
```

For real NVIDIA API-created targets inject `NVIDIA_API_KEY` through pod secrets and run `python scripts/build_sft_dataset.py` without `--no-api`. The pseudo-label fallback is not validated training supervision.

Official **30B LoRA** setup, subject to W&B/Slurm/checkpoint/artifact configuration:

```bash
git clone https://github.com/NVIDIA-NeMo/Nemotron.git /workspace/Nemotron
cd /workspace/Nemotron
uv sync
uv run nemotron nano3 sft --help
# Prepare/register official Packed-Parquet SFTDataArtifact first.
cd /workspace/FinscaleNvidiaPlatform
export NEMOTRON_REPO=/workspace/Nemotron
export NEMOTRON_RUN_PROFILE="your-valid-profile"
export RUN_NEMOTRON_SFT=1
bash scripts/run_nemotron_sft.sh
```

**The raw chat JSONL is not a direct input to the official command.** Do not enable the final script until you have validated the recipe and data preprocessing using the instructions in [README.md](README.md). Transfer results to persistent storage/object store, verify them, and stop/terminate your pod to avoid idle GPU charges. Network volume charges may continue while the pod is stopped.
