# FinScale — Hackathon Master Runbook
**Version:** 2026-11-13 | **Branch:** Nemotron | **Hardware:** 1–2 NVIDIA DGX Spark GB10 | **Orchestrator:** CIQ Fuzzball v4 | **Status:** pre-training / distributed integration

> **Evidence discipline:** As of this document, GB10 CUDA forward/backward of causal-conv1d, Mamba imports, tokenizer/dataset loading, and 30B weights download are confirmed. **Nemotron 30B optimizer-step training, adapter saving, inference evaluation and Fuzzball two-node model training are not confirmed.** Do not confuse synthetic GPU operations or NCCL communication with model training. Commands below are an executable *attempt* conditioned on the documented gates, not a guarantee of GB10 fit or Fuzzball compatibility.

## 1. Executive pitch: why FinScale?

Public-company SEC 10-K filings contain lengthy disclosures on business operations, market risks, liquidity, legal exposure, supply chains and cybersecurity. Analysts must read unstructured documents, locate relevant statements and produce consistent, auditable assessments. FinScale proposes to reduce repetitive reading by converting filing text into **traceable, structured risk signals**. Its intended output schema is:
- **Risk class:** `market`, `credit`, `liquidity`, `operational`, `regulatory_legal`, `cybersecurity`, `supply_chain`, `other`.
- **Severity:** `low`, `medium`, `high`.
- **Evidence:** verbatim source quote, filing/ticker/section metadata and model confidence or abstention where support is insufficient.

Potential users: investment analysts, compliance and risk teams, researchers and financial intelligence systems. The value proposition is **faster first-pass triage, repeatable labeling, source-grounded audit trails, and private/local deployment**. It does **not** replace expert financial analysis or establish regulatory compliance; these benefits must be validated against a manually reviewed test set.

### Why fine-tune instead of prompting alone?

Prompt-only models may vary in label names, JSON validity, severity thresholds and source quotation. A domain-specific supervised fine-tune can teach a consistent taxonomy, response structure and filing terminology; it may reduce prompt length and inference variability. **This is a hypothesis to measure** against a frozen base-model prompting baseline, not a proven outcome. Evaluate class macro-F1, per-class precision/recall, severity agreement, JSON validity, evidence quote fidelity, hallucination rate, inference latency and cost.

### Why LoRA?

A 30B-parameter BF16 model has ~60 GB of weights before optimizer state and activations. Full fine-tuning requires gradients and optimizer states for all trainable parameters and is impractical on a single GB10 with ~128 GB unified memory. **Low-Rank Adaptation (LoRA)** freezes pretrained weights and learns small low-rank updates `ΔW = (α/r) B A` for selected linear layers, reducing trainable parameter count and optimizer memory. For the current reference: `r=8`, `alpha=16`, base BF16, learning rate `1e-5`, microbatch 1. LoRA still needs memory for base weights, activations, gradients and kernels, and **does not guarantee the 30B training graph fits**. Measure trainable-parameter percentage and actual memory.

### Why NVIDIA Nemotron 3 Nano 30B-A3B?

The selected student checkpoint is `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, a hybrid Mamba/MoE architecture with approximately 30B total parameters and roughly 3B activated per token (model-family naming, not a memory guarantee). It fits the NVIDIA GB10 + local AI + multi-node engineering theme and tests a challenging, realistic hybrid architecture rather than a tiny demonstration-only model. **30B total weights must still be resident or sharded**; 3B active parameters do not mean 3B memory footprint. Model-specific `mamba-ssm`, `causal-conv1d` and Triton/CUDA compatibility are essential.

### What has actually been adapted?

- SEC EDGAR 10-K documents collected from a roughly 30-company universe.
- NVIDIA hosted teacher `nvidia/nemotron-3.5-lightning-30b-a3b` generated task examples.
- Current GB10 JSONL: **164 training + 18 validation = 182** samples, each in chat `messages` format with filing metadata.
- NeMo AutoModel 0.8.0 experimental local single-GPU LoRA recipe derived from the official `nemotron_nano_v3_singlegpu_lora.yaml`.
- Student 30B checkpoint downloaded to `/workspace/models/nemotron-30b` (13 shards; ~59 GiB).
- ARM64/CUDA13/SM12.1 Mamba and causal-convolution extensions compiled and imported.

**Critical scientific gap:** Current teacher examples appear to be summary/extractive quote tasks; they have **not been demonstrated** to provide the eight-class/severity schema. Before claiming a risk classifier, inspect and if necessary relabel a representative sample, build stratified held-out evaluation and prevent filing/company leakage.

## 2. Architecture

```text
SEC EDGAR 10-K -> document extraction/chunking -> NVIDIA teacher labeling
    -> quality checks + filing-level split -> chat JSONL -> tokenizer
    -> Nemotron 3 Nano 30B BF16 (frozen) + LoRA adapter
    -> local GB10 training -> adapter/checkpoint -> structured inference
    -> risk/evidence dashboard + held-out evaluation
                           |
             Fuzzball v4 orchestrator
                  /               \
        DGX Spark GB10 #0     DGX Spark GB10 #1
             rank 0                rank 1
                  \___ NCCL / torch.distributed ___/
                  performance + rank logs + checkpoint
```

The distributed stage may require an alternative **validated** FSDP/EP training recipe; the single-GPU recipe is not automatically two-node-capable. Do not assume the event's Spark machines share one memory pool.

## 3. Known challenges, diagnostic and resolution

| Observed problem | Diagnosis / correction | Status |
|---|---|---|
| GPUWerk tenant container | No nested Docker, systemd or host driver access; use Python venv/source builds | Understood |
| Python 3.12 ARM64 wheels | Prebuilt Mamba/Causal CUDA extensions unavailable | Built from source |
| Mamba C++ compile errors | PyTorch 2.14 requires C++20; Mamba setup.py passed C++17 | Changed four `-std=c++17` flags to `-std=c++20`; wheel built |
| HTTP 404 during Mamba build | Fallback wheel download, not the primary compiler error | Resolved via source compile |
| CUDA arch | GB10 SM12.1 requires compatible toolkit and extension build | CUDA13.0, TORCH_CUDA_ARCH_LIST=12.1 |
| causal-conv1d | Local aarch64 wheel, BF16 forward+backward on CUDA | **PASS** |
| mamba-ssm | Local aarch64 wheel; Nemotron training-kernel import | **IMPORT PASS**; kernel execution unverified |
| `hf` CLI missing | Use `huggingface_hub.snapshot_download` Python API | 13 shards downloaded |
| Hugging Face unauthenticated warning | Optional HF_TOKEN for authenticated download; never commit token | Download completed |
| Dataset truncation value `right` | Invalid Transformers TruncationStrategy | Use `longest_first` |
| Tokenizer no pad ID | EOS fallback warning | Investigate padding and masking |
| 512-token example | **0 supervised labels**; assistant response may be truncated | **BLOCKER** |
| Official AutoModel single-GPU recipe | Uses HellaSwag; checkpoint disabled; H100 reference | Adapt carefully and test |
| Lambda A100 40GB | Insufficient for current unsharded BF16 30B approach | Use GB10 |
| AWS G5 / AL2023 | 8GB root partition exhausted; PyTorch install failed | Separate volume/resize before install |
| Previous two-node EC2 NCCL | Port 29500 timeout, incorrect interface assumptions, missing probe script | Validate real interface and permitted ports |
| Old SFT launcher | Requires W&B packed Parquet/env.toml and is guarded | Use official NeMo AutoModel recipe instead |

### Rebuild extension (only if necessary on fresh ARM64 machine)

```bash
source /workspace/automodel-test-env/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.1 MAX_JOBS=4
cd /workspace/mamba-source-check/extracted/mamba
cp -n setup.py setup.py.original
python3 - <<'PY'
from pathlib import Path
p=Path("setup.py"); s=p.read_text()
n=s.count("-std=c++17")
if n not in (0,4): raise SystemExit(f"Unexpected C++17 flag count: {n}")
p.write_text(s.replace("-std=c++17","-std=c++20"))
print("C++17 flags replaced:",n)
PY
python3 -m pip wheel . --no-build-isolation --no-deps --no-cache-dir \
  --wheel-dir /workspace/mamba-source-check/wheels
python3 -m pip install --no-deps /workspace/mamba-source-check/wheels/mamba_ssm-*.whl
```

Do not reuse these binary wheels on x86, mismatched PyTorch ABI, different Python or incompatible CUDA environment. Do not rebuild on the event machines unless truly needed; obtain compatible prepared image from organizers.

## 4. GB10 single-node — exact chronological commands

### 4.1 Preflight (GPUWerk paths; adapt on physical Spark)

```bash
source /workspace/automodel-test-env/bin/activate
cd /workspace/FinscaleNvidiaPlatform
git pull --ff-only origin Nemotron
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
export TORCH_CUDA_ARCH_LIST=12.1 MAX_JOBS=4
mkdir -p results/nemotron-gb10
{ date -u; hostname; uname -m; nvidia-smi; nvcc --version; free -h; df -h /workspace; git rev-parse HEAD; } \
  2>&1 | tee results/nemotron-gb10/preflight.txt
python3 - <<'PY'
import torch, causal_conv1d, mamba_ssm
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
print("torch",torch.__version__,"CUDA",torch.version.cuda)
print("device",torch.cuda.get_device_name(0),"capability",torch.cuda.get_device_capability(0))
print("extension imports: PASS")
PY
wc -l data/processed/{train,val}.jsonl
ls /workspace/models/nemotron-30b/model-*.safetensors | wc -l
```

**Expected from previous machine:** 164/18 rows, 13 shards. Re-run checks on event hosts.

### 4.2 Inspect true labels and supervised tokens

```bash
python3 - <<'PY'
import json
p="data/processed/train.jsonl"
with open(p) as f: row=json.loads(next(f))
print("roles:",[x.get("role") for x in row["messages"]])
print("assistant characters:",len(row["messages"][-1].get("content","")))
print("metadata:",row.get("metadata",{}))
print("assistant excerpt:",row["messages"][-1].get("content","")[:350])
PY
python3 scripts/check_gb10_supervision.py \
  --model /workspace/models/nemotron-30b \
  --data data/processed/train.jsonl \
  --lengths 512 1024 2048 4096 --limit 20 \
  2>&1 | tee results/nemotron-gb10/supervision.log
```

**STOP if no supervised tokens.** A tokenizer sample loading successfully is not a training pass. If positives appear only at 4096, train at 4096 *only if memory allows*. If no positives even at 4096, inspect `ChatDataset` formatting and response-mask logic before any model load.

### 4.3 Generate experimental one-step recipe

```bash
python3 scripts/prepare_gb10_config.py --seq-length 2048 --steps 1
grep -nE 'pretrained_model_name_or_path|path_or_dataset_id|seq_length|max_steps|enabled:|_target_' \
  configs/nemotron_gb10_finscale.yaml
```

**Change `--seq-length` to the shortest *verified* length with meaningful supervised labels.** This is a recipe generator, not a guarantee the official AutoModel train loop accepts every dataset configuration.

### 4.4 Actual 30B LoRA training attempt (no five-minute timeout)

```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.1
set -o pipefail
/usr/bin/time -v automodel configs/nemotron_gb10_finscale.yaml --nproc-per-node 1 \
  2>&1 | tee results/nemotron-gb10/one_step.log
echo "training exit code: ${PIPESTATUS[0]}"
```

**Pass requires:** model loaded; positive labels; finite loss; backward completed; LoRA optimizer step completed. If out-of-memory, inspect actual unified memory usage, activation checkpointing, sequence length and kernel logs. Do not silently claim success. This recipe has `checkpoint.enabled: false`; one step alone does not produce an adapter.

### 4.5 Longer run, checkpoint and evaluation

After one-step success, regenerate with `--steps 10` or `--steps 20`. **Before launching**, inspect installed AutoModel checkpoint configuration and enable adapter checkpointing using its supported schema. Verify checkpoint files exist, adapter reloads, and actual held-out model inference outputs are produced. Keep logs and model/data/config hashes. Avoid blindly flipping `checkpoint.enabled` without checking required output paths and save cadence.

### 4.6 Evidence and safe backup

```bash
git rev-parse HEAD
sha256sum data/processed/{train,val}.jsonl
tar -czf /workspace/finscale-data-backup.tar.gz \
  data/raw/edgar_10k.jsonl data/processed/train.jsonl data/processed/val.jsonl
sha256sum /workspace/finscale-data-backup.tar.gz
```

Transfer the archive, compiled wheels, logs, adapters and charts to persistent storage. Do not upload API keys or 59GiB model weights to GitHub.

## 5. CIQ Fuzzball v4 — concrete commands and starter workflows

**Docs:** https://docs.ciq.com/fuzzball/4.2/appendices/workflow-syntax ; https://docs.ciq.com/fuzzball/4.2/parallel-and-distributed/generic ; https://docs.ciq.com/fuzzball/parallel-and-distributed/nccl .

The actual hackathon control plane, container image URI, group, storage mounts, GPU resource name and network rules must be provided by event staff. CLI and server versions must match. **Do not install a new Fuzzball control plane.**

### 5.1 Log in and inspect

```bash
fuzzball --version
fuzzball context --help
fuzzball context list
fuzzball context login --help
fuzzball workflow --help
fuzzball workflow validate --help
fuzzball workflow start --help
```

If event context already exists, use it. Otherwise ask organizers for their **exact** context-create and login instructions; do not guess auth realm or credentials. CLI commands `fuzzball workflow validate FILE`, `fuzzball workflow start FILE`, and `fuzzball workflow stop UUID` are documented; see sources above.

### 5.2 Single-node Fuzzball smoke

Edit the placeholder `docker://REPLACE_WITH_ORGANIZER_ARM64_CUDA_IMAGE` in `fuzzball/gb10_single_smoke.fz`. It requests one GPU and runs preflight in the organizer-provided image.

```bash
fuzzball workflow validate fuzzball/gb10_single_smoke.fz
fuzzball workflow start fuzzball/gb10_single_smoke.fz
# Copy returned UUID. Open workflow status/logs in Fuzzball Web UI.
```

If GPU resource naming differs, have staff adjust `resource.devices` and image. An ordinary host Docker daemon is **not** required; Fuzzball manages containers.

### 5.3 Fuzzball two-node NCCL test

Edit image and code mount assumptions in `fuzzball/gb10_two_node_nccl.fz`. It uses **`multinode.nodes: 2`, `implementation: generic`, `procs-per-node: 1`**, and the documented `/multinode/generic-helper` `exec_all` to launch one `torchrun` per host. It runs `scripts/nccl_multinode_proof.py` and requires the project code and compatible Python/PyTorch to exist in the container. Stage the code via the organizer-approved image or mounted volume before starting.

```bash
fuzzball workflow validate fuzzball/gb10_two_node_nccl.fz
fuzzball workflow start fuzzball/gb10_two_node_nccl.fz
# Record UUID, two hostnames, ranks 0/1, world_size 2, all-reduce sum 3.
```

Do **not** assume the job will run on the event cluster without adapting resource annotations and storage. A valid YAML file is not proof of scheduling or connectivity. Use the Fuzzball Web UI for allocation and per-rank log screenshots.

### 5.4 Two-node *model training* — separate validation

The previous step proves NCCL communication only. A true two-node training job requires a supported NeMo AutoModel distributed recipe (FSDP/EP/DP), identical weights/data, global batch configuration, correct checkpoint semantics, and a **completed optimizer step on both ranks**. Do **not** run the single-GPU YAML with `--nnodes=2` and call it supported. Inspect installed `/workspace/Automodel/examples` for multi-GPU Nemotron recipe and validate the training strategy on GB10. Once validated, replace the probe command in the Fuzzball Fuzzfile with the proven `automodel` distributed launch, and preserve the exact edited Fuzzfile. **No validated two-node Nemotron training recipe is currently included.**

### 5.5 If direct terminal access is available (without Fuzzball)

As a separate fallback only, run two concurrent shells with the same reachable rank-0 IP:

```bash
# Node 0
torchrun --nnodes=2 --nproc-per-node=1 --node-rank=0 \
 --master-addr=NODE0_PRIVATE_IP --master-port=29500 \
 scripts/nccl_multinode_proof.py
# Node 1
torchrun --nnodes=2 --nproc-per-node=1 --node-rank=1 \
 --master-addr=NODE0_PRIVATE_IP --master-port=29500 \
 scripts/nccl_multinode_proof.py
```

Replace `NODE0_PRIVATE_IP` with actual address. Do not hardcode `NCCL_SOCKET_IFNAME=ens5`; determine real NIC and use staff-authorized TCP/RDMA paths. Port 29500 is rendezvous, not the entire NCCL data-plane port range. Do not open all ports.

## 6. Benchmarking and graphical evidence

For each **real optimizer step**, record:
`run_id,nodes,model,precision,seq_length,global_batch_size,step_seconds,tokens,loss`.

Measure actual processed nonpadding tokens, exclude cold loading/JIT for steady-state, and compare equal global batches and sequence lengths. Repeat >=3 steady-state steps per topology. Compute:
- `tokens_per_second = measured_tokens / measured_step_seconds`
- `speedup = two_node_tokens_per_second / single_node_tokens_per_second`
- `parallel_efficiency = speedup / 2 * 100`

Never fill unknown metrics with guesses. Use:

```bash
python3 scripts/plot_training_comparison.py \
  --csv results/training_steps.csv \
  --out results/benchmark_charts
```

This produces `throughput.png` and `step_latency.png`. Capture rank logs, Fuzzball allocation screenshot, model and dataset SHA, adapter checkpoint, inference outputs, loss trend and evaluation quality metrics. On unified-memory GB10, `nvidia-smi` memory readings can be N/A; do not report them as precise dedicated VRAM usage.

## 7. End-to-end demo and judging narrative

1. **Problem:** SEC filings are large, unstructured and hard to review consistently.
2. **Input:** select ticker and show exact filing/section.
3. **Teacher/data:** show traceable quote + actual label; disclose current training-target limitations.
4. **Fine-tuning:** show frozen Nemotron 30B plus small LoRA adapter and measured training logs.
5. **Local AI:** show on-device inference with source-grounded output.
6. **Orchestration:** show Fuzzball allocating two distinct GB10 hosts.
7. **Communication:** show ranks 0/1, distinct hostnames and NCCL sum 3.
8. **Training:** only if successful, show two-host optimizer step and adapter checkpoint.
9. **Performance:** measured single vs two node throughput, speedup, scaling efficiency and latency charts.
10. **Reproducibility:** Git SHA, model revision, config, data split/hash, workflow UUID, logs and limitations.

**Fallback if distributed 30B cannot finish:** demonstrate authentic two-node distributed training of a smaller model and 30B single-node inference/training separately; explicitly label the models and topologies. This is stronger than misrepresenting a communication test as 30B distributed training.

## 8. Tomorrow's priorities and time allocation

| Priority | Action | Evidence |
|---|---|---|
| P0 | Fuzzball access + two GB10 allocation + shared model/data/image | Workflow ID, two hostnames |
| P0 | Supervised-token fix + one GB10 30B optimizer step | Nonzero labels, finite loss, step log |
| P0 | Two-node NCCL via Fuzzball | rank0/rank1 logs, all-reduce sum 3 |
| P1 | Supported distributed model recipe and real two-rank optimizer step | Two-rank training logs |
| P1 | Adapter checkpoint and held-out inference | Saved files + reproducible prediction |
| P1 | Three+ comparable measured steps/topology | CSV + graphs |
| P2 | Polished dashboard, presentation, extra tuning | Demo screenshot + summary |

### Staff questions that unblock everything

Ask organizers: **What is the Fuzzball v4.3 context/login? What ARM64 CUDA image and GPU resource annotations are available? Can one workflow reserve two separate GB10 nodes? How are project files/model shards mounted? What network interface/ports or RDMA fabric are permitted? Which official example launches PyTorch `torchrun` across both nodes? Where are outputs persisted?**

## References

- CIQ Fuzzfile v4 syntax: https://docs.ciq.com/fuzzball/4.2/appendices/workflow-syntax
- CIQ generic multinode helper: https://docs.ciq.com/fuzzball/4.2/parallel-and-distributed/generic
- CIQ NCCL training: https://docs.ciq.com/fuzzball/parallel-and-distributed/nccl
- CIQ workflow start: https://docs.ciq.com/fuzzball/4.0/quick-start/uqs-starting-and-stopping-workflows
- NVIDIA NeMo AutoModel: https://github.com/NVIDIA-NeMo/Automodel
- Nemotron Nano3 SFT: https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/nano3/sft.md
- Project: https://github.com/haranadhar/FinscaleNvidiaPlatform/tree/Nemotron
