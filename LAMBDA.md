# Lambda Labs — FinScale Nemotron 3 Nano 30B-A3B

A step-by-step guide for replacing the former **single A100 40GB** Lambda instance with a GPU configuration appropriate for **Nemotron 3 Nano 30B-A3B BF16 LoRA**. This guide is written for Lambda's `ubuntu` account and uses **`/home/ubuntu`**, not RunPod's `/workspace`.

> **IMPORTANT — do this before terminating the old instance.** GitHub contains the project code but **does not contain the downloaded SEC filings or generated teacher labels**. The previous run created **166 training** and **18 validation** examples (184 total), under `data/processed/`, plus the filings under `data/raw/`. Those numbers are historical results, not files automatically available to a new instance.

## 0. Preserve your current 184-example dataset BEFORE terminating the old VM

On the **old** Lambda Labs instance:

```bash
cd "$HOME/FinscaleNvidiaPlatform"
test -s data/raw/edgar_10k.jsonl
test -s data/processed/train.jsonl
test -s data/processed/val.jsonl
wc -l data/processed/train.jsonl data/processed/val.jsonl

tar -C "$HOME/FinscaleNvidiaPlatform" -czf "$HOME/finscale-edgar-sft-backup.tar.gz" \
  data/raw/edgar_10k.jsonl \
  data/processed/train.jsonl \
  data/processed/val.jsonl
sha256sum "$HOME/finscale-edgar-sft-backup.tar.gz"
ls -lh "$HOME/finscale-edgar-sft-backup.tar.gz"
```

**Download the archive to your own laptop while the old VM is still running**, using a terminal on your **laptop** (replace the host/IP with the old instance's actual public address):

```bash
scp -i ~/.ssh/YOUR_LAMBDA_PRIVATE_KEY \
  ubuntu@OLD_LAMBDA_IP:/home/ubuntu/finscale-edgar-sft-backup.tar.gz \
  ~/Downloads/
sha256sum ~/Downloads/finscale-edgar-sft-backup.tar.gz  # Linux; macOS: shasum -a 256
```

For cloud backups, you can instead upload the archive to your own secure object storage; confirm the object is downloadable before deletion. The archive contains financial source text and model-generated labels; treat it as project data, **not as a GitHub commit**. **Do not terminate the old VM until you have verified a second copy**. Termination may irreversibly delete instance storage.

## 1. Choose a Lambda Labs GPU configuration

**Target:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (about 30B total parameters, roughly 3.5B active). Mixture of experts does **not** shrink base checkpoint VRAM to the number of active parameters.

- **Preferred starting point for the NVIDIA-documented 30B LoRA topology:** **2 × A100 80GB** or **2 × H100 80GB on one VM**, subject to available offerings and recipe support. NVIDIA NeMo Helix's Nano3 reference lists **2 × 80GB with expert parallelism EP=2** for LoRA.
- A single **A100 40GB**, even with reduced batch size, cannot load the unsharded BF16 30B model and complete the published recipe.
- Do **not** substitute **2 × A100 40GB** for **2 × 80GB** or assume **1 × 80GB** matches the documented 2-GPU topology.
- Different GPUs (e.g., H200/B200) can have different memory/topology requirements; validate the exact training stack before spending money.
- Pick a recent compatible NVIDIA CUDA/PyTorch/NeMo image; confirm sufficient storage for the base model, environment, datasets, and checkpoints. Treat **250 GB free** as a planning starting point, **not** a measured minimum. Check volume persistence and storage charges.

Once connected to the **new** Ubuntu VM:

```bash
whoami
pwd
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
df -h "$HOME"
free -h
python3 --version
git --version
```

Confirm the VM has **two GPU devices**, and that the memory shown is ~80GB each (or another explicitly verified supported topology), before proceeding.

## 2. Clone FinScale and install CPU / data-preparation dependencies

```bash
cd "$HOME"
git clone --branch Nemotron --single-branch \
  https://github.com/haranadhar/FinscaleNvidiaPlatform.git
cd "$HOME/FinscaleNvidiaPlatform"

git branch --show-current
git log -1 --oneline
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m py_compile scripts/*.py
python scripts/smoke_test_cuda.py
```

If `smoke_test_cuda.py` reports missing `torch` or CUDA is unavailable, investigate the chosen base image/environment rather than blindly installing a CPU-only PyTorch wheel. The SEC data preparation scripts do **not** need a GPU.

## 3. Restore the archive onto the new instance (recommended; avoids API calls and regeneration)

**From your laptop** (replace IP/key values with the **new** instance's details):

```bash
scp -i ~/.ssh/YOUR_LAMBDA_PRIVATE_KEY \
  ~/Downloads/finscale-edgar-sft-backup.tar.gz \
  ubuntu@NEW_LAMBDA_IP:/home/ubuntu/
```

Then on the **new** Lambda VM:

```bash
cd "$HOME/FinscaleNvidiaPlatform"
tar -tzf "$HOME/finscale-edgar-sft-backup.tar.gz"
tar -C "$PWD" -xzf "$HOME/finscale-edgar-sft-backup.tar.gz"
python scripts/validate_edgar.py --strict
wc -l data/processed/train.jsonl data/processed/val.jsonl
```

A restored copy of the previous successful dataset should show **166 train + 18 validation = 184** examples; if the counts differ, inspect your backup rather than assuming the new files match.

### Alternative only when backup is unavailable: download and regenerate

```bash
cd "$HOME/FinscaleNvidiaPlatform"
source .venv/bin/activate
read -rp "SEC contact name and email: " SEC_USER_AGENT
export SEC_USER_AGENT
python scripts/download_edgar.py --limit 1 --pause 0.25
python scripts/validate_edgar.py --strict

# NVIDIA-hosted teacher inference: NO second Lambda GPU is used.
read -rsp "NVIDIA API key: " NVIDIA_API_KEY
echo
export NVIDIA_API_KEY
python scripts/build_sft_dataset.py \
  --input data/raw/edgar_10k.jsonl \
  --output-dir data/processed \
  --workers 2
wc -l data/processed/train.jsonl data/processed/val.jsonl
unset NVIDIA_API_KEY
```

`--workers 2` means two concurrent **remote NVIDIA API requests**, not two physical GPUs or two Lambda nodes. The API model configured in the script is `nvidia/nemotron-3.5-lightning-30b-a3b` (**teacher**). The fine-tuning **student** is Nemotron 3 **Nano** 30B-A3B. API labels may have rate limits, quotas, and associated fees. **Do not run `--no-api` over the restored high-quality labels** unless intentionally replacing them with a cheap smoke-test dataset.

## 4. Inspect SFT labels and split quality

```bash
cd "$HOME/FinscaleNvidiaPlatform"
python - <<'PY'
import json
from collections import Counter
from pathlib import Path

splits = {}
for name in ("train", "val"):
    path = Path(f"data/processed/{name}.jsonl")
    records = [json.loads(line) for line in path.open() if line.strip()]
    if not records:
        raise SystemExit(f"Empty split: {path}")
    for r in records:
        assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant"]
        assert all(m["content"].strip() for m in r["messages"])
    tickers = {r["metadata"]["ticker"] for r in records}
    splits[name] = tickers
    print(f"{name}: {len(records)} examples; {len(tickers)} companies")
    print("label types:", dict(Counter(r["metadata"]["label_type"] for r in records)))
    print("sample answer:", records[0]["messages"][-1]["content"][:300])
print("company overlap between train and validation:",
      sorted(splits["train"] & splits["val"]))
PY
```

**Research caution:** the current FinScale generator makes a **row-level random split**. Overlapping company/filing data between train and validation can inflate evaluation claims. Create a separate company-held-out or later-filing test set for benchmark reporting. The 184 generated examples are a **small experimental dataset**, not a production financial classifier dataset.

## 5. Clone official NVIDIA Nemotron tooling in the CORRECT Lambda home directory

**Do not clone into `/workspace` on Lambda**; this previously caused `Permission denied`.

```bash
cd "$HOME"
git clone https://github.com/NVIDIA-NeMo/Nemotron.git "$HOME/Nemotron"

# install uv if not installed
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Important: avoid activating the FinScale venv while running uv from NVIDIA repo
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
cd "$HOME/Nemotron"
uv sync
uv run nemotron nano3 data prep sft --help
uv run nemotron nano3 sft --help
```

Keep the two repositories separate:
- `$HOME/FinscaleNvidiaPlatform` = FinScale code and JSONL dataset.
- `$HOME/Nemotron` = NVIDIA official Nano3 training/data-preparation recipes.

Do not move datasets or entire virtual environments between machines; restore the JSONL archive and reinstall dependencies.

## 6. Configure actual 30B SFT data prep and LoRA recipe — not yet automated by this project

**STOP BEFORE TRAINING:** the checked-in `configs/lora_sft.yaml` is a **reference with TODO placeholders**. The `scripts/run_nemotron_sft.sh` wrapper intentionally refuses to launch without a configured official recipe. Running `uv sync` and printing the CLI help **does not train anything**.

The Nano3 workflow must be connected to the FinScale dataset by:
1. Mapping `data/processed/{train,val}.jsonl` into the **official Nano3 Stage 1 data-prep input** (OpenAI-style chat messages). Check the currently cloned NVIDIA code/docs for how `DataBlend`, input sources, and dataset artifact registration are configured.
2. Running Nano3 SFT data prep with the **actual Nano3 tokenizer/chat template** and role-based loss masking to create **packed Parquet** (or another format demanded by the exact pinned recipe version). Register the expected SFT data artifact. **Do not pass raw JSONL directly to `sft --run`.**
3. Configuring `env.toml` and an execution profile supported on **your Lambda VM**. NVIDIA docs often show a **Slurm cluster profile**, while a standard Lambda standalone VM does **not** automatically provide a Slurm controller. Check the NeMo-Run local/Docker execution path or run the verified stage-specific scripts directly inside the required container; do not copy a made-up Slurm profile.
4. Downloading/importing the **compatible Nano3 BF16 base checkpoint**. Ensure it is the Nano3 model, not the Nemotron 3.5 Lightning teacher checkpoint. Check Hugging Face license/access, disk availability, cache path, and runtime architecture support.
5. Setting **LoRA** correctly for the selected Nano3/Megatron-Bridge recipe, including the published **EP=2** topology for two 80GB GPUs, and verifying tensor/pipeline parallelism, precision, sequence length, batch size, model sharding, and checkpoint output.
6. Running a smoke-test job first, then monitoring `nvidia-smi`, step loss, memory, iteration duration, and saving reproducible benchmarks.

Official reference entry points and diagnostics:

```bash
cd "$HOME/Nemotron"
git rev-parse HEAD                 # record exact recipe revision for reproducibility
uv run nemotron nano3 data prep sft --help
uv run nemotron nano3 sft --help

# Inspect these docs before choosing or implementing an executor:
sed -n '1,240p' docs/nemotron/nano3/README.md
sed -n '1,260p' docs/nemotron/nano3/sft.md
sed -n '1,220p' docs/nemo_runspec/nemo-run.md
```

**Only after** the input artifact, checkpoint, LoRA config, and **working `env.toml` profile** are completed and verified:

```bash
cd "$HOME/FinscaleNvidiaPlatform"
export NEMOTRON_REPO="$HOME/Nemotron"
export NEMOTRON_RUN_PROFILE="REPLACE_WITH_VALID_CONFIGURED_PROFILE"
export RUN_NEMOTRON_SFT=1
# Do NOT execute this until all prerequisites above are satisfied:
# bash scripts/run_nemotron_sft.sh
```

The launch command remains commented to avoid accidentally spending GPU hours on a nonworking setup. The current wrapper is *not* a complete 30B LoRA integration. Official instructions: [Nano3 recipe](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/docs/nemotron/nano3), [Stage 1 SFT](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/nano3/sft.md), [data prep](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/data-prep.md), [execution profiles](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemo_runspec/nemo-run.md).

## 7. Save everything before stopping the new VM

```bash
cd "$HOME/FinscaleNvidiaPlatform"
tar -czf "$HOME/finscale-sft-$(date +%Y%m%d).tar.gz" \
  data/raw/edgar_10k.jsonl data/processed/train.jsonl data/processed/val.jsonl
sha256sum "$HOME"/finscale-sft-*.tar.gz
```

Also export any adapters, checkpoints, logs, and metrics generated by your *actual* training recipe. Verify that your backups exist in **durable storage outside the VM** before terminating it. Do not publish secrets or private experiment artifacts to GitHub. Terminate idle GPU instances to avoid ongoing compute charges.
