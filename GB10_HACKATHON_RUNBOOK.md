# FinScale GB10: verified recovery, single-node LoRA and hackathon runbook

**Status as of Nov 13, 2026:** NVIDIA Nemotron 3 Nano 30B-A3B BF16 weights downloaded; CUDA extension tests passed; **NO 30B LoRA optimizer step has yet been observed**. Do not claim trained adapters or throughput until measured. Use branch `Nemotron`. This runbook supersedes older statements that weights have not been downloaded *on the GPUWerk tenant*; weights and labeled JSONL are **not stored in GitHub**.

## 1. Actual evidence from GPUWerk GB10

| Item | Observed |
|---|---|
| GPU | NVIDIA GB10, compute capability 12.1, ARM64 |
| Torch | 2.14.1+cu130; CUDA 13.0 |
| CUDA toolkit | 13.0; nvcc 13.0.88, sm_121 |
| NeMo AutoModel | editable 0.8.0+6f44dbf, `/workspace/Automodel` |
| Test venv | `/workspace/automodel-test-env` (uses original `/workspace/gb10-env` packages) |
| causal-conv1d | 1.7.0, locally built ARM64 wheel (~198 MiB), import + real CUDA BF16 forward/backward **PASS** |
| mamba-ssm | 2.3.2.post1, locally built ARM64 wheel (~334 MiB), import + `mamba_split_conv1d_scan_combined` import **PASS**; fused kernel execution **NOT YET TESTED** |
| Teacher-labeled FinScale data | `data/processed/train.jsonl` 164, `val.jsonl` 18 (182 total) on GPUWerk |
| Student checkpoint | `/workspace/models/nemotron-30b`, 13 safetensors shards, ~59 GiB on disk; download completed in 17m56s |
| Storage after download | 726 GB free / 836 GB filesystem |
| First tokenization test | 164 rows; sample 0: 511 input tokens, **0 supervised tokens** at seq_length 512 |
| Training | **NOT RUN**. Zero supervised tokens must be fixed before training |

**Caution:** The teacher-labeled dataset currently contains SEC financial-document summary/quote tasks, not necessarily the original eight-category risk classification and severity labels. Confirm target output schema before claiming risk classification quality. Train/validation split may share companies or filings; use filing-level holdout for defensible evaluation.

## 2. Failures, causes and exact resolutions

1. **GPUWerk is a tenant container:** no Docker-in-Docker/systemd/host driver control; do not run bare-metal DGX Spark Docker instructions inside it. Hackathon physical DGX Spark access may differ.
2. **ARM64 extension wheel unavailable:** `mamba-ssm` and `causal-conv1d` required local source builds; `flash_attn` was missing and is not established as required for this path.
3. **Mamba build failed after many kernels compiled:** PyTorch 2.14 headers use C++20 (`std::strong_ordering`, `starts_with`, `requires`), while Mamba source `setup.py` explicitly passed `-std=c++17`. The prebuilt wheel download HTTP 404 was not the compiler root cause. Change **four** `-std=c++17` occurrences in Mamba 2.3.2.post1 `setup.py` to `-std=c++20`, preserve all other flags, set `TORCH_CUDA_ARCH_LIST=12.1`, `MAX_JOBS=4`, build with `pip wheel . --no-build-isolation --no-deps --no-cache-dir`. Built and installed successfully. Save wheel files before deleting instance.
4. **`hf: command not found`:** Hugging Face Python library was installed, CLI absent. Use `huggingface_hub.snapshot_download(repo_id=...,local_dir=...,max_workers=4)`. Unauthenticated request warning did not prevent the download.
5. **`truncation: right` invalid:** Transformers accepts `longest_first`, `only_first`, `only_second`, `do_not_truncate` or boolean. Updated to `longest_first`.
6. **Critical remaining blocker:** at 512-token truncation, first training example has 0 supervised tokens. Likely assistant response was truncated out; **must verify across dataset**, not assume. Do not run training with all labels masked.
7. **`pad_token_id` missing:** tokenizer falls back to EOS id 11; investigate masking/padding before interpreting losses. Do not silently modify tokenizer vocabulary.
8. **Earlier cloud experiments:** Lambda A100 40GB unsuitable for this unsharded BF16 30B recipe; source download Python/f-string issue; W&B packed-Parquet/env.toml requirement in older guarded launcher; missing API key; AL2023 8GB root partition, PyTorch install no space; NCCL port 29500 TCP timeouts, incorrect NIC `ens5` assumption, missing `nccl_probe.py`; never open all 0–65535 ports as a workaround.

## 3. Restore/check the single GB10 environment

```bash
source /workspace/automodel-test-env/bin/activate
cd /workspace/FinscaleNvidiaPlatform
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.1
export MAX_JOBS=4
python - <<'PY'
import torch, causal_conv1d, mamba_ssm
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print('Mamba imports OK; kernel execution still needs validation')
PY
wc -l data/processed/{train,val}.jsonl
ls /workspace/models/nemotron-30b/model-*.safetensors | wc -l
```

If recreating a fresh Spark environment, **do not blindly install Torch 2.14.1**: first inspect the preinstalled driver, CUDA, Python and ARM64 wheels. GPUWerk paths may not exist at the hackathon. For the extension build, use the saved wheels only if Python ABI, PyTorch ABI, CUDA and architecture match; otherwise rebuild.

## 4. Correct the supervised-token problem BEFORE real training

From the repository root, run the checked-in diagnostic:

```bash
python scripts/check_gb10_supervision.py --model /workspace/models/nemotron-30b --data data/processed/train.jsonl --lengths 512 1024 2048 4096 --limit 20
```

If 512 gives 0 labels but 2048/4096 gives >0, increase YAML `dataset.seq_length` cautiously; longer sequences increase memory/compute. If *all* give 0, inspect template, assistant roles and label masking; do not launch training. Validate `labels != -100` for a batch, not just one row.

## 5. Prepare a **real one-step** single-GB10 LoRA run

The official upstream recipe is `/workspace/Automodel/examples/llm_finetune/nemotron/nemotron_nano_v3_singlegpu_lora.yaml`. It uses HellaSwag, disables checkpointing, and was tested on H100; GB10 compatibility is **not proven**. The locally created `configs/nemotron_gb10_finscale.yaml` is **not yet committed to GitHub**. Use `scripts/prepare_gb10_config.py` to regenerate from the actual installed upstream recipe:

```bash
source /workspace/automodel-test-env/bin/activate
cd /workspace/FinscaleNvidiaPlatform
python scripts/prepare_gb10_config.py --seq-length 2048 --steps 1
cat configs/nemotron_gb10_finscale.yaml
```

Only after positive supervised-token validation:

```bash
mkdir -p results/nemotron-gb10
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.1
set -o pipefail
/usr/bin/time -v automodel configs/nemotron_gb10_finscale.yaml --nproc-per-node 1 \
  2>&1 | tee results/nemotron-gb10/one_step.log
```

**This is an attempted real training step, not a promise it will fit** in 128GB unified memory or that all kernels will execute. Keep the log and verify a **nonzero supervised label count**, finite loss, backward and optimizer-step completion. Watch OOM, process kills and Triton/CUDA compilation errors. If failing, preserve logs and use the fallback small model rather than claiming success. Run without a 5-minute timeout if there is time.

For a longer run, after the first optimizer step succeeds, change `max_steps` to 10 or 20, enable checkpointing **only after inspecting the installed recipe's checkpoint schema**, and verify saved adapter files and reload. The default upstream `checkpoint.enabled: false` means **no adapter is saved**. Do not advertise a checkpoint until one exists.

## 6. Backup and portability

```bash
cd /workspace/FinscaleNvidiaPlatform
mkdir -p results/nemotron-gb10
tar -czf /workspace/finscale-gb10-data.tar.gz \
  data/raw/edgar_10k.jsonl data/processed/train.jsonl data/processed/val.jsonl
sha256sum /workspace/finscale-gb10-data.tar.gz
ls -lh /workspace/mamba-source-check/wheels/
```

Copy dataset archive, wheel files, run logs, adapters, CSVs and charts to persistent storage before terminating GPUWerk. Do not commit 59GiB weights, secrets, private API keys, or full source filings to GitHub. On a new node, download the model from Hugging Face and verify its revision.

## 7. Single vs two-node evidence gates

| Gate | What counts as proof |
|---|---|
| G0 | Identical code/model/dataset/precision/sequence length, machine and network metadata |
| G1 | Both nodes visible, unique hostnames, GPUs and IPs |
| G2 | Bidirectional connectivity on coordinator port and actual NCCL data ports |
| G3 | `torchrun` two ranks with successful NCCL all-reduce; **not** proof of multi-node model training |
| G4 | Fuzzball job shows two distinct hosts and worker ranks in logs |
| G5 | Model-specific distributed forward/backward/optimizer step succeeds across both hosts |
| G6 | Same total work comparison, measured steady-state tokens/s and step times, repeated >=3 times |
| G7 | Checkpoint saved/reloaded, inference/evaluation outputs and reproducible run manifest |

**Crucial:** Two independent single-node jobs, two GPUs on one host, or a two-node NCCL all-reduce alone do **not** prove distributed Nemotron training. If model training fails, label NCCL evidence as communication-only.

Continue with [Fuzzball two-node runbook](FINSCALE_FUZZBALL_TWO_NODE.md) and [benchmark/visualization instructions](FINSCALE_FUZZBALL_TWO_NODE.md#benchmarking-and-visual-evidence).
