# RunPod — FinScale Nemotron 3 Nano 30B-A3B

This is the **RunPod** equivalent of [LAMBDA.md](LAMBDA.md). Use the RunPod pod's **verified persistent mount** (often `/workspace`). Lambda's standard Ubuntu VM uses `/home/ubuntu` instead. All training prerequisites and limitations from [README.md](README.md) apply.

## 1. Provision and verify

For the published Nemotron 3 Nano BF16 LoRA reference topology, start with **two 80GB GPUs** if available, e.g. 2 × A100 80GB or 2 × H100 80GB. The official recipe can require a suitable container, valid checkpoint, SFT data artifact, and supported execution profile; GPU count alone does not make it runnable. Consider **250GB free storage** as an initial planning estimate, not a hard NVIDIA requirement. Check network volume persistence, availability, driver/CUDA compatibility, and pricing.

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
df -h /workspace
whoami
```

## 2. Clone FinScale into your persistent RunPod workspace

```bash
test -w /workspace || { echo "/workspace is not writable; choose your writable persistent mount"; exit 1; }
cd /workspace
git clone --branch Nemotron --single-branch https://github.com/haranadhar/FinscaleNvidiaPlatform.git
cd /workspace/FinscaleNvidiaPlatform
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m py_compile scripts/*.py
python scripts/smoke_test_cuda.py
```

## 3. Restore existing teacher-labeled data instead of paying NVIDIA to regenerate

The previous successful run created **166 train + 18 validation** examples (184 total); these are **NOT** part of the repository. Copy `finscale-edgar-sft-backup.tar.gz` from your own durable backup to the pod (SSH/SCP or your storage provider's copy method), then:

```bash
cd /workspace/FinscaleNvidiaPlatform
tar -tzf /workspace/finscale-edgar-sft-backup.tar.gz
tar -C "$PWD" -xzf /workspace/finscale-edgar-sft-backup.tar.gz
python scripts/validate_edgar.py --strict
wc -l data/processed/train.jsonl data/processed/val.jsonl
```

Only if the old labeled data was not backed up, rerun SEC downloading and NVIDIA hosted-teacher generation:

```bash
cd /workspace/FinscaleNvidiaPlatform
source .venv/bin/activate
read -rp "SEC contact name and email: " SEC_USER_AGENT
export SEC_USER_AGENT
python scripts/download_edgar.py --limit 1 --pause 0.25
python scripts/validate_edgar.py --strict

read -rsp "NVIDIA API Key: " NVIDIA_API_KEY
echo
export NVIDIA_API_KEY
python scripts/build_sft_dataset.py --workers 2
wc -l data/processed/train.jsonl data/processed/val.jsonl
unset NVIDIA_API_KEY
```

Here `--workers 2` = two simultaneous hosted NVIDIA API requests; **not** two GPUs.

## 4. Install official NVIDIA Nemotron recipes and inspect training options

```bash
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
cd /workspace
git clone https://github.com/NVIDIA-NeMo/Nemotron.git /workspace/Nemotron
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
cd /workspace/Nemotron
uv sync
uv run nemotron nano3 data prep sft --help
uv run nemotron nano3 sft --help
```

## 5. Configure (NOT YET AUTOMATED) the 30B LoRA job

- Integrate restored OpenAI-style chat JSONL with the official Nano3 data-prep input.
- Use matching Nano3 tokenizer and `nano3` chat template; pack to the SFT artifact format required by the pinned recipe (current NVIDIA docs describe packed Parquet with assistant-only loss masks).
- Register the data artifact and prepare a compatible **Nemotron 3 Nano BF16** base checkpoint.
- Configure the supported environment/launcher: a RunPod pod does **not** automatically run a Slurm cluster just because it has two GPUs.
- Validate LoRA, expert parallelism EP=2 when using the NVIDIA two-80GB Nano3 recipe, sequence length/batch size, output storage and logs.
- Check `env.toml` and official NVIDIA CLI documentation; the current FinScale `configs/lora_sft.yaml` is a **non-runnable reference**.
- Only after these prerequisites exist, use the guarded wrapper with a real profile:

```bash
cd /workspace/FinscaleNvidiaPlatform
export NEMOTRON_REPO=/workspace/Nemotron
export NEMOTRON_RUN_PROFILE="REPLACE_WITH_VERIFIED_PROFILE"
export RUN_NEMOTRON_SFT=1
# Not ready for execution until the integration is completed:
# bash scripts/run_nemotron_sft.sh
```

References: [NVIDIA Nano3 training guide](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/docs/nemotron/nano3), [SFT](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/nano3/sft.md), [data prep](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/data-prep.md).

Export trained adapters/checkpoints/metrics to storage that remains accessible after termination, and verify the backups. Stopping a GPU pod may not stop storage billing.
