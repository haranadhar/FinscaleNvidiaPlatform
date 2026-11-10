# Lambda Labs — FinScale Nemotron 30B

1. Provision Lambda Labs NVIDIA GPU VM with **2 × 80 GB GPU** or better for the NVIDIA-documented LoRA target. GPU count/size and CUDA toolkit must be checked; a one-GPU A100 40 GB Qwen setup is **not** this 30B recipe.
2. SSH into the instance (use your actual Lambda SSH details), then:

```bash
nvidia-smi
df -h
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

The `--no-api` data is only a cheap extractive smoke test; for model-generated supervised targets, inject `NVIDIA_API_KEY` safely and run `python scripts/build_sft_dataset.py` without `--no-api`.

For **30B LoRA** follow the official NVIDIA Nano3 NeMo/Megatron-Bridge [guide](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/docs/nemotron/nano3). Prepare packed Parquet with official tooling, register the W&B SFT dataset artifact, set the cluster/Slurm profile and verify model checkpoint and LoRA configuration. Then run:

```bash
git clone https://github.com/NVIDIA-NeMo/Nemotron.git /workspace/Nemotron
cd /workspace/Nemotron
uv sync
uv run nemotron nano3 sft --help
# After preparing/validating the actual profile and packed-Parquet artifact:
cd ~/FinscaleNvidiaPlatform
export NEMOTRON_REPO=/workspace/Nemotron
export NEMOTRON_RUN_PROFILE="your-valid-profile"
export RUN_NEMOTRON_SFT=1
bash scripts/run_nemotron_sft.sh
```

Replace `~/FinscaleNvidiaPlatform` with your actual checkout directory as needed. The final command is intentionally gated and **not validated end-to-end**. Use attached/persistent storage for Hugging Face/model caches and checkpoints; export your results before deleting the VM. Stop instances when not in use to control costs.
