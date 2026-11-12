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

## GPU choices for 30B and the DGX Spark hackathon

**Do not confuse GPU architecture with memory topology.** The hackathon's DGX Spark uses the **GB10 Grace Blackwell Superchip** with **128GB unified system memory** and an **ARM64 CPU**. A cloud **RTX PRO 6000 Blackwell 96GB** is also Blackwell, but uses **dedicated GDDR7 VRAM** and typically an **x86_64 host**; the two are not performance- or binary-equivalent. DGX Spark's unified memory is not 128GB of conventional dedicated GPU VRAM. Cross-architecture Docker images and compiled CUDA extensions may need separate ARM64 and x86_64 builds.

| GPU / configuration | Memory | Guidance for FinScale Nemotron 3 Nano 30B |
| --- | --- | --- |
| **2 × A100 80GB on one VM** | 80GB per GPU; 160GB aggregate (not pooled) | **Preferred practical Lambda choice** for the NVIDIA-documented two-80GB LoRA topology, subject to supported image, parallelism and actual availability. |
| **2 × RTX PRO 6000 Blackwell 96GB on one VM** | 96GB per GPU; 192GB aggregate (not pooled) | **Preferred Blackwell cloud comparison** when available on RunPod; validate NCCL/peer-to-peer links, CUDA/NeMo support and EP=2. Do not assume same performance as GB10. |
| **1 × RTX PRO 6000 Blackwell 96GB** | 96GB dedicated | Useful for Blackwell inference and quantized experiments; **not verified** for the published two-GPU BF16 LoRA recipe. |
| **1 × RTX 6000 Ada 48GB** | 48GB dedicated | Not suitable for the current unsharded BF16 30B training recipe; use only with a separately validated quantized/offload approach. |
| **1 × A100 40GB** | 40GB dedicated | Previous VM; good for SEC/API dataset preparation, **not** the existing BF16 30B LoRA recipe. |
| **1–2 × DGX Spark GB10** | 128GB unified per device (typical) | Hackathon hardware; optimize for ARM64, memory bandwidth, inter-node communication, and a separately validated quantized/PEFT path. Not a direct reproduction of the A100 EP=2 recipe. |

**No H100 is required or recommended for this plan.** NVIDIA's published Nano3 30B LoRA reference is a **two-80GB** configuration with expert parallelism **EP=2**; hardware fit is a prerequisite, **not** a guarantee that the current FinScale wrapper is runnable. Do not add VRAM across GPUs and assume a single process can use the sum. Verify supported device count, CUDA architecture, distributed training topology, RAM, disk, and network before renting.

**Provider discovery:** Lambda GPU inventory/pricing: https://lambda.com/pricing ; RunPod GPU inventory/pricing: https://www.runpod.io/gpu-models . Listings are not a guarantee of live availability. RTX **6000 Ada (48GB)**, older **Quadro RTX 6000 (24GB)**, and **RTX PRO 6000 Blackwell (96GB)** are **different products**.

### Record hardware and portability baseline

Run on **each** Lambda/RunPod/DGX Spark machine, and save the output with your benchmark:

```bash
uname -m
cat /etc/os-release | head -n 5
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv || nvidia-smi
nvidia-smi topo -m
nvcc --version || true
python3 - <<'PY'
import platform
print("CPU architecture:", platform.machine())
try:
    import torch
    print("torch:", torch.__version__, "CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
except ImportError:
    print("torch not installed in current Python environment")
PY
```

For 2-GPU jobs, also verify `nvidia-smi topo -m` and run NCCL/distributed smoke tests **before** training. Two GPUs on different pods or instances are **not** automatically equivalent to two GPUs inside one machine.

## Previously encountered issues and verified fixes

| Problem observed on old Lambda A100 40GB | Cause | Correct action |
| --- | --- | --- |
| `git clone ... /workspace/Nemotron` → `Permission denied` | Lambda `ubuntu` could not create directories under RunPod-style `/workspace` | Clone official NVIDIA repo into **`$HOME/Nemotron`**. |
| `cd ~/FinscaleNvidiaPlatform` → no directory | NVIDIA repo had been cloned, but user's FinScale repo had not | Clone **both** repos separately into `$HOME`. |
| `scripts/run_nemotron_sft.sh: No such file` | Executing from the official NVIDIA repo, not FinScale | `cd "$HOME/FinscaleNvidiaPlatform"` before running project scripts. |
| `VIRTUAL_ENV ... does not match project environment` | FinScale venv active while running `uv` in NVIDIA repo | `deactivate 2>/dev/null || true; unset VIRTUAL_ENV` before NVIDIA `uv sync`. |
| `Not launching: set up W&B Packed-Parquet SFT artifact and env.toml` | Training wrapper is intentionally guarded; the recipe integration is incomplete | Prepare real Nano3 packed SFT artifact, checkpoint, valid executor/profile and verified LoRA settings. **Do not merely set the bypass flag.** |
| `SyntaxError: unterminated string literal` in `download_edgar.py` | Broken generated `print` line; SEC ticker directory also overwrote ticker list | Fixed on `Nemotron` branch (commit `f420c13`). On new VM use latest branch and `python -m py_compile scripts/*.py`. |
| Confusion about `--workers 2` with one GPU | Workers are concurrent NVIDIA **hosted API requests** | `--workers 2` works on a single GPU/CPU VM; it does **not** require two GPUs. |
| A100 40GB and 30B BF16 LoRA | ~60GB base BF16 weights before training overhead, and official two-GPU topology | Use **2 × A100 80GB** or investigate **2 × RTX PRO 6000 Blackwell 96GB**, with recipe validation. |
| Replacing VM risks losing 184 labeled examples | GitHub excludes `data/raw` and `data/processed` | Back up **166 train + 18 validation**, plus raw EDGAR JSONL, off the VM **before termination**. |

**Previously completed milestone:** 30-company EDGAR extraction and NVIDIA-hosted `nvidia/nemotron-3.5-lightning-30b-a3b` teacher generation produced **166 train / 18 validation / 184 total**. **This was data generation, not local 30B fine-tuning.**

## 1. Choose a Lambda Labs GPU configuration

**Target:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (about 30B total parameters, roughly 3.5B active). Mixture of experts does **not** shrink base checkpoint VRAM to the number of active parameters.

- **Preferred Lambda choice:** **2 × A100 80GB on one VM** (where offered), matching the two-80GB NVIDIA reference topology with **EP=2**. **Do not rent H100 for this hackathon plan.** For a closer Blackwell cloud comparison, consider **2 × RTX PRO 6000 Blackwell 96GB on RunPod**, only after validating training compatibility.
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

## Portable workflow: Lambda A100 / RunPod RTX PRO Blackwell / DGX Spark GB10

The **portable artifacts** are FinScale Python code, SEC JSONL, labeled chat JSONL, evaluation definitions, and appropriately converted model adapters. The **non-portable parts** include compiled CUDA extensions, GPU-specific kernels, NCCL topology, image architecture and some training recipe settings.

```bash
# In the FinScale repo on ANY platform, after restoring the archive:
python -m py_compile scripts/*.py
python scripts/validate_edgar.py --strict
wc -l data/processed/train.jsonl data/processed/val.jsonl
# On a new instance, do NOT regenerate the labels unless backup is unavailable.
```

For the hackathon, measure separately: (1) model/adapter loading and inference on GB10, (2) single-node GPU utilization, (3) distributed communication across two GB10 nodes if supplied, and (4) the cloud training baseline. **Do not report cloud A100 throughput as DGX Spark performance.** Use model checkpoint and tokenizer compatibility checks before attempting adapter transfer.
