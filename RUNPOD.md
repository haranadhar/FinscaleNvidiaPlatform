# RunPod — FinScale Nemotron 3 Nano 30B-A3B

This is the **RunPod** equivalent of [LAMBDA.md](LAMBDA.md). Use the RunPod pod's **verified persistent mount** (often `/workspace`). Lambda's standard Ubuntu VM uses `/home/ubuntu` instead. All training prerequisites and limitations from [README.md](README.md) apply.

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

## 1. Provision and verify

For the published Nemotron 3 Nano BF16 LoRA reference topology, choose **2 × RTX PRO 6000 Blackwell 96GB** for the closest Blackwell-generation cloud comparison if available, or **2 × A100 80GB** for the documented two-80GB topology. Do not confuse RTX PRO 6000 Blackwell 96GB with RTX 6000 Ada 48GB or RTX A6000 48GB. The official recipe can require a suitable container, valid checkpoint, SFT data artifact, and supported execution profile; GPU count alone does not make it runnable. Consider **250GB free storage** as an initial planning estimate, not a hard NVIDIA requirement. Check network volume persistence, availability, driver/CUDA compatibility, and pricing.

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
df -h /workspace
whoami
```

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
