# FinScale — Nemotron 3 Nano 30B-A3B + SEC EDGAR

**Branch:** `Nemotron` | **Student model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | **Dataset:** SEC EDGAR 10-K filings from a target universe of 30 US public companies.

**Start here on Lambda Labs:** **[LAMBDA.md](LAMBDA.md)** — this now contains the full, ordered commands for (a) backing up the existing teacher-labeled dataset, (b) provisioning a replacement 2 × 80GB GPU VM, (c) restoring labeled JSONL data, and (d) checking the NVIDIA official Nano3 recipe prerequisites. **RunPod:** [RUNPOD.md](RUNPOD.md).

> **Existing successful run (previous Lambda A100 40GB):** NVIDIA API teacher `nvidia/nemotron-3.5-lightning-30b-a3b` generated **166 train + 18 validation = 184 examples**. This run is **not** a completed 30B student LoRA training job. **These JSONL files are NOT in GitHub. Back up `data/raw/` and `data/processed/` from the old VM before terminating it.** If the old VM is already gone and no backup exists, regenerate the dataset from SEC EDGAR and NVIDIA API on the replacement VM.

## Hardware guidance

- **Previous:** single A100 40GB (suitable for preparation/teacher API labeling, *not* the published BF16 Nano3 30B LoRA recipe).
- **Recommended Lambda starting configuration:** **two 80GB GPUs** (e.g. **2 × A100 80GB** or **2 × H100 80GB**, if offered). The NVIDIA Nemotron 3 Nano NeMo Helix reference documents **LoRA: 2 × 80GB, TP=1, EP=2, PP=1**. This is a reference topology, not a guarantee that every stock Lambda image runs the recipe unchanged.
- **Do not confuse teacher vs. student:** API teacher = **Nemotron 3.5 Lightning 30B-A3B**; locally fine-tuned student = **Nemotron 3 Nano 30B-A3B**. `--workers 2` controls concurrent NVIDIA hosted API calls, **not** GPU count.
- Verify GPU devices, VRAM, storage, CUDA drivers, compatible training image, official recipe execution method, and costs before GPU training.

## Project scripts

| Path | Purpose | Status |
| --- | --- | --- |
| `scripts/download_edgar.py` | Fetch one or more SEC 10-K filings per company | Operational after syntax/ticker-list fix |
| `scripts/validate_edgar.py` | Audit coverage of 30 requested companies | Operational |
| `scripts/build_sft_dataset.py` | Extract filing sections and produce teacher-labeled OpenAI-style messages | Worked on previous Lambda instance |
| `scripts/smoke_test_cuda.py` | Check CUDA/PyTorch | Environment diagnostic |
| `scripts/run_nemotron_sft.sh` | Guarded handoff to NVIDIA official Nano3 SFT CLI | **Not an end-to-end launcher** |
| `configs/lora_sft.yaml` | Illustrative settings with TODO placeholders | **Not runnable** |
| `requirements.txt` | CPU and NVIDIA hosted-teacher API dependencies | Not a full GPU training environment |

## Default 30-company universe

```text
AAPL NVDA MSFT AMZN GOOGL META TSLA JPM BAC WFC
GS MS V MA PYPL NFLX DIS WMT COST HD
UNH JNJ PFE ABBV XOM CVX BA CAT AMD INTC
```

## Fast path on a NEW Lambda Ubuntu VM

**Before the replacement:** preserve the old labeled JSONL files via the backup/upload instructions in [LAMBDA.md](LAMBDA.md).

```bash
# Fresh replacement Lambda instance, after confirming two supported GPUs:
cd "$HOME"
git clone --branch Nemotron --single-branch https://github.com/haranadhar/FinscaleNvidiaPlatform.git
cd "$HOME/FinscaleNvidiaPlatform"
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m py_compile scripts/*.py
nvidia-smi -L

# Once you copy the backup archive into /home/ubuntu:
tar -C "$PWD" -xzf "$HOME/finscale-edgar-sft-backup.tar.gz"
python scripts/validate_edgar.py --strict
wc -l data/processed/train.jsonl data/processed/val.jsonl
```

If you do **not** have the archive, see [LAMBDA.md §3](LAMBDA.md#3-restore-the-archive-onto-the-new-instance-recommended-avoids-api-calls-and-regeneration) for fresh SEC/teacher API instructions.

## Official NVIDIA Nano3 30B LoRA: setup vs. an actual training run

```bash
# Run this OUTSIDE the activated FinScale virtual environment
deactivate 2>/dev/null || true
unset VIRTUAL_ENV
cd "$HOME"
git clone https://github.com/NVIDIA-NeMo/Nemotron.git "$HOME/Nemotron"

# Install uv first if unavailable, following LAMBDA.md
cd "$HOME/Nemotron"
uv sync
uv run nemotron nano3 data prep sft --help
uv run nemotron nano3 sft --help
```

**STOP:** the current FinScale repo does **not** yet connect your JSONL into an official packed-Parquet SFT artifact, import a compatible Nano3 base checkpoint, or define a verified Lambda executor/`env.toml` profile and LoRA configuration. Merely buying a 2 × 80GB VM and setting `RUN_NEMOTRON_SFT=1` **will not** train the model. The guarded wrapper is intentionally non-runnable until this integration is completed. Refer to the official NVIDIA [Nano3 training documentation](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/docs/nemotron/nano3), [data prep](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/data-prep.md), and [NeMo-Run execution](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemo_runspec/nemo-run.md).

## Data quality and research considerations

- The 184 API-labeled examples are a **small** supervised set and need answer-level spot checks.
- The current generator splits examples by **row**, not held-out company or held-out filing; compute a separate held-out benchmark before claiming generalization.
- Never commit SEC files, W&B/Hugging Face/NVIDIA keys, local checkpoints, output archives, or large artifacts into source control.
- Original FinScale Qwen pipeline remains on `main`; `Nemotron` is the adapted Nemotron-focused branch.
- Data preparation originated from [Manish's FinScale branch](https://github.com/mkchoithani/finscale/tree/edgar-nemotron-runpod).

**Reference model:** https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.
