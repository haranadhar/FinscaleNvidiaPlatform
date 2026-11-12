# GPUWerk — FinScale Nemotron 30B on a DGX Spark GB10

**Target branch:** `Nemotron`  
**Provider:** GPUWerk dedicated DGX Spark **tenant container** (not a VM)  
**Student:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`  
**Teacher used for labels:** NVIDIA hosted `nvidia/nemotron-3.5-lightning-30b-a3b`  
**Goal:** reproducible GPU baseline, SEC dataset validation, honest 30B training readiness, and performance evidence for the hackathon.

> **Read this first.** This repository does **not yet contain an executable, fully integrated Nemotron 30B Nano LoRA fine-tuning recipe** for a single GB10. The current `scripts/run_nemotron_sft.sh` is intentionally guarded and `configs/lora_sft.yaml` has TODO placeholders. The commands below marked **READY** work as diagnostics or benchmarks when their dependencies are installed. The commands marked **TRAINING GATE** require completing a compatible NVIDIA training recipe and actual 30B model/artifact integration; do not report a successful 30B training run until a checkpoint, metrics and evaluation exist. A successful synthetic CUDA benchmark is **not** 30B training.

## 0. What GPUWerk actually provides

GPUWerk starts a **container** on a dedicated DGX Spark with one **GB10 Grace Blackwell GPU**, **128 GB unified CPU/GPU memory**, an **ARM64** CPU and **1 TB shared NVMe storage**. The 1 TB is **disk**, *not* GPU memory; free space depends on provider-side usage. You are `root` **inside the container**. There is no systemd, Docker socket or privileged Docker-in-Docker; **do not attempt Docker commands from NVIDIA's bare-metal Spark playbooks**, or host driver/CUDA-driver upgrades. CUDA/PyTorch software must match the tenant image and ARM64 platform. GPU memory statistics can show `N/A` on a unified-memory GB10; the monitor leaves such values blank and separately records `/proc/meminfo`.

References: [GPUWerk tenant environment](https://gpuwerk.com/docs/tenant-environment/), [GPUWerk SSH](https://gpuwerk.com/docs/ssh-keys/), [NVIDIA DGX Spark hardware](https://docs.nvidia.com/dgx/dgx-spark/hardware.html), [NVIDIA Nemotron Nano SFT](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/nano3/sft.md).

**Billing and persistence:** GPUWerk workspaces are not backed up; terminating deletes data. Stopping a reserved instance can still incur charges. Export logs/CSVs/plots/adapters before termination. Check the console's current prices and persistence policy.

### Connect

1. Log into [GPUWerk console](https://console.gpuwerk.com/) and deploy a GB10 instance using a **PyTorch/CUDA-compatible, ARM64-capable base image** if offered (rather than an auto-starting LLM inference preset).
2. Use **Open terminal in browser**, or add your public SSH key under GPUWerk account settings and connect with the **exact command shown on the instance page**.
3. For SSH from your laptop, a typical pattern is `ssh -i ~/.ssh/id_ed25519 root@YOUR_INSTANCE.gpuwerk.com`; use the actual instance hostname and key. No password is needed.

## 1. READY — Verify the instance before installing anything

Run all of the following **inside the GPUWerk terminal**:

```bash
set -o pipefail
mkdir -p /workspace/finscale-evidence/preflight
cd /workspace
whoami
pwd
uname -m
cat /etc/os-release | head -n 8
nvidia-smi -L
nvidia-smi
nvidia-smi topo -m || true
nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv || true
nvidia-smi --help-query-gpu | head -n 35 || true
free -h
df -h /workspace
df -i /workspace
command -v python3 && python3 --version
command -v git && git --version
command -v nvcc && nvcc --version || true
python3 - <<'PY'
import importlib.util, platform
print("Machine:",platform.machine())
for pkg in ("torch","transformers","peft","bitsandbytes","nemo_automodel","matplotlib"):
    print(pkg, bool(importlib.util.find_spec(pkg)))
if importlib.util.find_spec("torch"):
    import torch
    print("Torch:",torch.__version__,"CUDA:",torch.version.cuda)
    print("CUDA available:",torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:",torch.cuda.get_device_name(0))
PY
```

Copy the output to the baseline file if you want a reproducible record:

```bash
{
  date -u
  uname -a
  nvidia-smi -L
  nvidia-smi
  free -h
  df -h /workspace
  python3 --version
} > /workspace/finscale-evidence/preflight/system.txt 2>&1
```

**STOP if no GB10 is visible or `torch.cuda.is_available()` is false in the intended training Python environment.** Do not blindly `pip install torch` over a GPU-enabled ARM64 image or install a host NVIDIA driver from inside the container.

## 2. READY — Clone the correct GitHub branch

```bash
cd /workspace
git clone --branch Nemotron --single-branch https://github.com/haranadhar/FinscaleNvidiaPlatform.git
cd /workspace/FinscaleNvidiaPlatform
git branch --show-current
git rev-parse HEAD
python3 -m py_compile scripts/*.py
mkdir -p outputs/gpuwerk
```

If previously cloned, use `cd /workspace/FinscaleNvidiaPlatform && git pull --ff-only origin Nemotron` instead of trying to clone again. Do not mix up the FinScale repository with NVIDIA's separate `Nemotron` repository.

Install **CPU/data preparation** dependencies in an isolated virtual environment **without upgrading CUDA torch**:

```bash
cd /workspace/FinscaleNvidiaPlatform
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt matplotlib
python -m py_compile scripts/*.py
```

If the provider's GPU framework is not visible in this environment, use its supported native Python environment for CUDA tasks. The scripts for metrics and graphs do not require `torch`; the synthetic benchmark does.

## 3. READY — Restore or recreate SEC EDGAR SFT data

**Your previous Lambda run reportedly produced 166 train + 18 validation examples. These files are not stored in GitHub.** Prefer restoring a separate, verified archive (see [LAMBDA.md](LAMBDA.md)), not regenerating at hackathon time.

From **your laptop** with an SSH key, after substituting your real host:

```bash
scp -i ~/.ssh/id_ed25519 ~/Downloads/finscale-edgar-sft-backup.tar.gz root@YOUR_INSTANCE.gpuwerk.com:/workspace/
```

Inside GPUWerk:

```bash
cd /workspace/FinscaleNvidiaPlatform
tar -tzf /workspace/finscale-edgar-sft-backup.tar.gz
tar -xzf /workspace/finscale-edgar-sft-backup.tar.gz -C .
wc -l data/processed/train.jsonl data/processed/val.jsonl
python scripts/validate_edgar.py --strict
python - <<'PY'
import json
from pathlib import Path
for name in ("train","val"):
 p=Path("data/processed")/(name+".jsonl")
 rows=[json.loads(x) for x in p.open() if x.strip()]
 assert rows, f"Empty {name}"
 assert all([m["role"] for m in x["messages"]]==["system","user","assistant"] for x in rows)
 print(name,len(rows),"examples; companies:",len({r.get("metadata",{}).get("ticker") for r in rows}))
PY
```

The previous generator uses a **row-level train/validation split**, so a different company or filing can leak into both sets. For defensible evaluation, create a held-out-company or held-out-filing split before claiming model generalization. **184 examples is small** and not proof of a useful improvement.

### If no backup exists

```bash
cd /workspace/FinscaleNvidiaPlatform
source .venv/bin/activate
read -rp "Your SEC contact name and email: " SEC_USER_AGENT
export SEC_USER_AGENT
python scripts/download_edgar.py --limit 1 --pause 0.25
python scripts/validate_edgar.py --strict
read -rsp "NVIDIA API key (input hidden): " NVIDIA_API_KEY
echo
export NVIDIA_API_KEY
python scripts/build_sft_dataset.py --input data/raw/edgar_10k.jsonl --output-dir data/processed --workers 2
unset NVIDIA_API_KEY
wc -l data/processed/{train,val}.jsonl
```

This makes **teacher-API labels**, *not* a locally trained Nemotron model. The `--workers 2` argument means two concurrent HTTP workers, **not** two GPUs. API pricing/quotas may apply. Never paste tokens into GitHub or evidence logs.

## 4. READY — Capture continuous GPU metrics, dmon, pmon and host memory

`monitor_gpuwerk.py` probes which `nvidia-smi --query-gpu` fields actually work, collects them at 2-second intervals, and leaves unsupported fields empty. It also records host `MemAvailable`/ `MemTotal` from Linux. `dmon` and `pmon` support may vary on GB10 and inside tenant containers; save their raw output, including any unsupported-metric errors, rather than treating missing values as zero.

Use **three terminal sessions**, or a controlled single-shell background launch.

**Terminal A — structured GPU telemetry (run before training):**

```bash
cd /workspace/FinscaleNvidiaPlatform
mkdir -p outputs/gpuwerk
python3 scripts/monitor_gpuwerk.py --output outputs/gpuwerk/gpu_metrics.csv --interval 2
```

**Terminal B — NVIDIA native monitors (run before training):**

```bash
cd /workspace/FinscaleNvidiaPlatform
nvidia-smi dmon -s pucvmt -d 2 > outputs/gpuwerk/dmon.log 2>&1 &
DMON_PID=$!
nvidia-smi pmon -s um -d 2 > outputs/gpuwerk/pmon.log 2>&1 &
PMON_PID=$!
echo "dmon=$DMON_PID pmon=$PMON_PID"
# Use the terminal to run/check later:
tail -n 20 outputs/gpuwerk/dmon.log
tail -n 20 outputs/gpuwerk/pmon.log
```

Some versions reject `-s pucvmt` or `-s um`. Check `nvidia-smi dmon -h` and `nvidia-smi pmon -h` and reduce the field selector as needed. `pmon` can return empty when per-process sampling is unsupported. An empty log is not evidence of zero usage.

**Terminal C — training/benchmark commands appear below.** Keep monitoring active throughout the job. On completion, interrupt Terminal A with Ctrl+C and run `kill "$DMON_PID" "$PMON_PID" 2>/dev/null || true` in the Terminal B shell.

### Useful live diagnosis commands

```bash
watch -n 2 nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv || true
ps -eo pid,comm,%cpu,%mem,rss --sort=-rss | head -n 20
free -h
cat /proc/meminfo | grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree'
df -h /workspace
```

`watch` is optional. Don't equate a GB10 `N/A` memory metric with free GPU memory; unified memory is shared with the CPU.

## 5. READY — Run a bounded GPU compute/burn-style smoke benchmark

**Use this first, before loading the 30B checkpoint.** This performs synchronized FP16 matrix multiplies for a bounded duration. It is deliberately **not** the `gpu-burn` utility, an MLPerf benchmark, or Nemotron 30B throughput. Run only when GPU-enabled PyTorch already works.

Keep Terminals A/B monitoring, then in Terminal C:

```bash
cd /workspace/FinscaleNvidiaPlatform
python3 scripts/benchmark_gpuwerk.py --seconds 45 --size 2048 --output outputs/gpuwerk/gpu_benchmark.csv
head -n 4 outputs/gpuwerk/gpu_benchmark.csv
```

The measured time is synchronized with `torch.cuda.synchronize()`. Throughput is a theoretical FLOP count divided by measured matmul duration, and should be described as **estimated synthetic TFLOP/s**, not achieved training TFLOPs or LLM tokens/s. For an alternative tool, inspect NVIDIA [DCGM](https://docs.nvidia.com/datacenter/dcgm/latest/) and a compatible GPU-Burn build, but neither is guaranteed accessible from GPUWerk's tenant container. Do not run unbounded heat/power stress tests.

## 6. TRAINING GATE — Nemotron **30B** architecture and honest readiness check

**Model identity matters.** Your **teacher** `nemotron-3.5-lightning-30b-a3b` was used via NVIDIA's hosted API; your **student** is `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`. These are not interchangeable checkpoints. A model with ~30B total parameters and ~3B active per token can still require memory for *all* expert weights. BF16 base weights alone are approximately 60+ GB, before optimizer states, activations, temporary allocations and host/CPU needs.

NVIDIA's official Nano3 Megatron-Bridge recipe and your project's prior documentation describe a **two-80GB-GPU** LoRA reference (TP=1, EP=2). **One GB10 with unified memory is not a supported drop-in substitute for two A100 80GB GPUs.** A single-GB10 BF16 30B LoRA run can run out of memory while *loading*, before a training step occurs. 4-bit QLoRA may be feasible with specific compatible NeMo AutoModel/transformers/Mamba builds, but is **experimental** until you validate it against this exact student checkpoint on the tenant image.

### Check the training stack first

```bash
cd /workspace/FinscaleNvidiaPlatform
python3 -m py_compile scripts/*.py
sed -n '1,140p' scripts/run_nemotron_sft.sh
sed -n '1,150p' configs/lora_sft.yaml
nvidia-smi -L
python3 - <<'PY'
from importlib.util import find_spec
for x in ["torch","transformers","peft","bitsandbytes","nemo_automodel","megatron"]:
 try: print(x, bool(find_spec(x)))
 except Exception as e: print(x,repr(e))
PY
```

A successful `torch` installation does *not* prove that `bitsandbytes`, `mamba_ssm` or custom MoE kernels support your ARM64 sm_121 device.

### Official model recipe inspection (not a training launch)

```bash
cd /workspace
git clone https://github.com/NVIDIA-NeMo/Nemotron.git NVIDIA-Nemotron
cd /workspace/NVIDIA-Nemotron
git rev-parse HEAD
ls docs/nemotron/nano3/
# If the provider image supports the project's uv/Python environment:
command -v uv || echo "uv not installed; inspect upstream installation first"
# After installing uv according to the official Nemotron repo:
# uv sync
# uv run nemotron nano3 data prep sft --help
# uv run nemotron nano3 sft --help
# uv run nemotron nano3 sft --dry-run
```

The official SFT CLI consumes **packed Parquet artifacts**, compatible pretrained model checkpoints, and NeMo-Run configuration. Raw FinScale `train.jsonl` is **not directly accepted** by the existing guarded wrapper. If using `uv`, first `deactivate 2>/dev/null || true; unset VIRTUAL_ENV` to avoid mixed Python environments.

**Required before claiming you trained the Nano3 student:**

1. Choose and pin an **ARM64/sm_121-compatible** NVIDIA training software stack and checkpoint revision.
2. Convert validated FinScale messages to correct **role-masked, tokenized, packed Parquet** (assistant tokens only contribute loss); verify output statistics.
3. Register/import the checkpoint and SFT dataset for the same **Nano3 text-only** recipe (not Nano3 Omni, Lightning, or an unrelated Llama/Qwen configuration).
4. Set up a **single-GB10-compatible** LoRA/QLoRA method; validate expert targeting, Mamba kernels, quantization exclusions, CPU/GPU loading peak, checkpointing and batch/sequence length. For BF16, check whether distributed EP=2 is actually required. Do **not** use the two-GPU configuration on a single GPU.
5. Run a **short real training smoke test**, e.g. 5–10 optimization steps, and confirm: nonzero gradients, loss changes, adapter/checkpoint saved, no OOM, and before/after evaluation. Then extend to a longer run.

The repo's **current** `configs/lora_sft.yaml` has a placeholder `_target_` and is not safe as an executable configuration. Do not set `RUN_NEMOTRON_SFT=1` just to bypass the safety gate.

### If you already have a fully validated official recipe

NVIDIA documents the following execution interfaces. These commands are **templates only** until the data/checkpoint/profile are fully configured for your machine:

```bash
# Inspect without launching:
cd /workspace/NVIDIA-Nemotron
# Requires a working uv environment:
uv run nemotron nano3 sft --dry-run

# The upstream NeMo-Run CLI only after configuring an actual executor/profile,
# valid packed SFT artifacts, compatible weights and a verified single-GB10 recipe:
# uv run nemotron nano3 data prep sft --run YOUR_CONFIGURED_PROFILE
# uv run nemotron nano3 sft --run YOUR_CONFIGURED_PROFILE
```

On a single GB10 tenant container, the above `--run` mode is **not assumed operational**; NVIDIA's documented examples use Slurm-based profiles and external cluster executors. Use an appropriately adapted direct training entry point if validated for GPUWerk.

**Experimental community report:** [NVIDIA Developer Forum — single GB10 QLoRA Nemotron Nano 30B](https://forums.developer.nvidia.com/t/fine-tuning-nemotron-3-nano-30b-a3b-on-asus-ascent-gx10-gb10-dgx-spark-cuda-oom-at-load-box-freezes-4-bit-qlora-fix/372434). This reports a 4-bit, Mamba-aware workaround on an **ASUS GX10**, *not GPUWerk*. It is third-party experience and may depend on patches/versions; audit the attached instructions before trying on your rented instance.

## 7. READY — Capture a *real* training command with monitors

Once you have a working training command (whether an adapted official Nano3 recipe or a validated QLoRA script), record the exact command and git revision; avoid putting API keys in command lines. Start monitor/dmon/pmon (Section 4) **before** execution.

```bash
cd /workspace/FinscaleNvidiaPlatform
mkdir -p outputs/gpuwerk
date -u +%FT%TZ | tee outputs/gpuwerk/training_started_utc.txt
git rev-parse HEAD | tee outputs/gpuwerk/finscale_commit.txt
# REPLACE ONLY AFTER you have actually validated the 30B training entrypoint:
# set -o pipefail
# python /absolute/path/to/validated_nano3_training.py ... 2>&1 | tee outputs/gpuwerk/train.log
# echo "training exit code: ${PIPESTATUS[0]}" | tee outputs/gpuwerk/training_exit_code.txt
```

**Do not label benchmark data as training results unless this real optimizer step loop completed.** For the graphs of learning behavior, export your real trainer logs into **`outputs/gpuwerk/train_metrics.csv`** with the columns below (a step and loss are required, the other columns are optional):

```csv
step,loss,tokens_per_second,step_time_s
```

Don't invent data rows, tokens/s, test accuracy or a loss curve. For an actual run, preserve **adapter/checkpoint files, trainer configuration, optimizer-step count, loss CSV and evaluation outputs**. Peak memory/utilization alone is insufficient to claim training was successful.

## 8. READY — Render performance charts from *actual* CSVs

After stopping the monitoring commands, in FinScale's Python environment:

```bash
cd /workspace/FinscaleNvidiaPlatform
source .venv/bin/activate
python -m pip install matplotlib
python scripts/plot_gpuwerk.py --dir outputs/gpuwerk
ls -lh outputs/gpuwerk
```

Charts are created **only where real numeric samples exist**; unsupported GB10 metrics are omitted, never filled with zero. Files include:

| Artifact | What it proves |
| --- | --- |
| `gpu_metrics.csv` | Timestamped GPU/host telemetry, blanks for unsupported fields |
| `dmon.log` | Native NVIDIA GPU device monitor output or error |
| `pmon.log` | Per-process monitor output or unsupported warning |
| `gpu_benchmark.csv` | Measured synthetic matrix throughput |
| `gpu_utilization.png` | GPU compute utilization during your recorded interval (when supported) |
| `gpu_memory.png` | Reported GPU memory only (when supported; may be blank for GB10) |
| `host_memory.png` | Free shared host memory |
| `gpu_power.png`, `gpu_temperature.png` | Device power/thermals only when supported |
| `synthetic_matmul.png` | GPU stress/synthetic throughput, not training tokens/s |
| `train_metrics.csv`, `training_loss.png` | **Only if you supplied real optimizer-step metrics** |
| `training_tokens_per_second.png` | **Only if measured and supplied by actual trainer** |
| `telemetry_summary.csv` | Means and sample counts for supported GPU/host telemetry |

Stop the monitor before running the graph command so the CSV is finalized. Capture summary statistics only for supported metric fields.

## 9. Hackathon comparisons and evaluation

Suggested results table — fill in only after actual runs:

| Experiment | Hardware | Model | Mode | Real optimizer steps? | Avg GPU util | Peak reported memory / host usage | Training loss / eval | Throughput | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | GPUWerk 1×GB10 | Synthetic FP16 matmul | Stress baseline | No | TBD | TBD | N/A | Estimated TFLOP/s | `gpu_benchmark.csv` |
| B | GPUWerk 1×GB10 | Nemotron Nano 30B | Inference | No | TBD | TBD | SEC task quality | Tokens/s measured separately | Model server log |
| C | GPUWerk 1×GB10 | Nemotron Nano 30B | LoRA/QLoRA SFT | **Only if validated** | TBD | TBD | Train/val and held-out test | Samples/s and tokens/s | Adapter + trainer metrics |
| D | Hackathon 2×GB10 | Nemotron Nano 30B | Multi-node if supported | **Only if actually run** | TBD | TBD | Held-out test | Per-node, scaling efficiency | NCCL logs + paired traces |

**Do not use GPUWerk single-Spark numbers to claim two-node GB10 performance.** Also, two independently rented GPUWerk Sparks do **not** automatically provide routable low-latency cluster fabric/NCCL communication. Validate connectivity, topology, permissions and orchestration separately before asserting distributed training. The hackathon's provided two nodes and CIQ/Fuzzball require a distinct orchestration/benchmark setup.

For any real SEC task report: retain the same held-out set, prompt format, max tokens, quantization mode and evaluation script across models. Report exact hardware, driver, PyTorch/CUDA revision, model SHA and software changes. With 184 small teacher-generated examples, avoid claims of production accuracy or model superiority without an independent held-out evaluation.

## 10. Export evidence BEFORE ending GPUWerk billing

From your laptop (use the actual hostname shown by GPUWerk):

```bash
scp -i ~/.ssh/id_ed25519 -r root@YOUR_INSTANCE.gpuwerk.com:/workspace/FinscaleNvidiaPlatform/outputs/gpuwerk ~/Downloads/finscale-gpuwerk-evidence
```

Inside GPUWerk, optionally create an archive:

```bash
cd /workspace/FinscaleNvidiaPlatform
tar -czf /workspace/finscale-gpuwerk-evidence.tar.gz -C outputs gpuwerk
sha256sum /workspace/finscale-gpuwerk-evidence.tar.gz
```

If there are real training adapters/checkpoints, export those **separately** before terminating. **Do not commit credentials, SEC datasets, checkpoints, or large metric files to GitHub**. The repository's `.gitignore` already excludes `outputs/*`.

## 11. Troubleshooting

| Symptom | Action |
| --- | --- |
| `docker: cannot connect to Docker daemon` | Expected in GPUWerk tenant; use directly installed compatible ARM64 software, not nested Docker |
| `systemctl: command not found` | Expected; tenant has no systemd |
| GPU memory query prints `N/A` | Expected possibility on integrated GB10; use host memory trend and framework memory stats |
| `nvidia-smi dmon/pmon` exits | Inspect `dmon.log` / `pmon.log`; reduce selector based on `-h`; no zero-filled replacements |
| `torch.cuda.is_available() = False` | Verify provider CUDA-enabled image and Python interpreter; do not install host driver |
| `exec format error` | ARM64 vs x86_64 mismatch |
| Training OOM on load | Stop; 128GB unified memory is not dedicated 128GB VRAM. Do not use 2-GPU EP=2 recipe on one GPU |
| `mamba_ssm` / MoE / Triton custom kernel error | Pin an ARM64 sm_121-compatible verified training stack; avoid swapping unrelated model recipes |
| `RUN_NEMOTRON_SFT=1` fails | Existing wrapper lacks packed dataset/checkpoint/profile wiring; this remains a **training integration task** |
| `data/processed/train.jsonl` missing | Restore old archive or regenerate labels from SEC/NVIDIA teacher API |
| `gpu_benchmark.py` no torch | Use provider's CUDA-enabled Python; no synthetic throughput claim if it cannot run |
| Graph missing | No numeric samples for that metric (or trainer CSV missing); this is deliberate |

### Primary documentation

- [GPUWerk tenant container and persistence](https://gpuwerk.com/docs/tenant-environment/)
- [GPUWerk SSH](https://gpuwerk.com/docs/ssh-keys/)
- [NVIDIA DGX Spark specifications](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [NVIDIA DGX Spark NeMo fine-tuning playbook (bare-metal/container guidance; Docker steps do NOT apply inside GPUWerk)](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/nemo-fine-tune/README.md)
- [Official Nemotron 3 Nano SFT](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/nano3/sft.md)
- [Community report: QLoRA Nemotron 30B on a GB10](https://forums.developer.nvidia.com/t/fine-tuning-nemotron-3-nano-30b-a3b-on-asus-ascent-gx10-gb10-dgx-spark-cuda-oom-at-load-box-freezes-4-bit-qlora-fix/372434)
