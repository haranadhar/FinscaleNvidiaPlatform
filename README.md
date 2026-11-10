# FinScale — Nemotron 3 Nano 30B-A3B + SEC EDGAR

**Target model:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (approximately 31.6B total parameters, 3B-class active). This `Nemotron` branch replaces the previous Qwen pipeline; the `main` branch is unchanged.

**Scope:** collect SEC 10-K disclosures for **30 companies**, construct grounded chat SFT data, and prepare an NVIDIA NeMo/Megatron-Bridge **30B** LoRA experiment on either **Lambda Labs or RunPod**. GPU training requires additional NVIDIA recipe configuration; this repository does **not** claim a verified, one-command 30B training run.

## Files
- `scripts/download_edgar.py`: collects up to `--limit` 10-K filings **per company**; defaults to 30 tickers.
- `scripts/build_sft_dataset.py`: section extraction and NVIDIA API-generated supervision; `--no-api` creates *pseudo-labels*, not verified summaries.
- `scripts/smoke_test_cuda.py`: CUDA visibility.
- `scripts/run_nemotron_sft.sh`: deliberately guarded hand-off to the **official** NVIDIA-NeMo/Nemotron Nano3 recipe; requires a registered Packed-Parquet dataset, W&B, valid cluster profile, and compatible checkpoint.
- `configs/lora_sft.yaml`: **reference template, not runnable**. Match fields to the pinned official recipe.
- `requirements.txt`: data-preparation and API dependencies; do not overwrite preinstalled CUDA PyTorch.
- `LAMBDA.md`, `RUNPOD.md`: provider-specific setup and storage.
- `data/raw/edgar_10k.jsonl`, `data/processed/{train,val}.jsonl`: created at runtime, excluded from Git.

## Default 30-company universe
```text
AAPL NVDA MSFT AMZN GOOGL META TSLA JPM BAC WFC
GS MS V MA PYPL NFLX DIS WMT COST HD
UNH JNJ PFE ABBV XOM CVX BA CAT AMD INTC
```
The actual number of retrieved filings can be lower (SEC availability, access, or malformed filings). A ticker list is not a pre-downloaded dataset.

## 1. Select GPU provider
**Lambda Labs:** [LAMBDA.md](LAMBDA.md). **RunPod:** [RUNPOD.md](RUNPOD.md).

For Nemotron **30B LoRA**, NVIDIA NeMo Helix documents a starting configuration of **2 × 80GB GPUs** (EP=2); full SFT lists **8 × 80GB GPUs**. The official recipe may need Slurm and a cluster environment, even on machines with enough VRAM. Do not assume a single A100 40GB or GB10 can train BF16 30B LoRA. Provider availability/prices and driver/image compatibility must be checked.

## 2. Clone **the Nemotron branch**, not main
```bash
git clone --branch Nemotron --single-branch https://github.com/haranadhar/FinscaleNvidiaPlatform.git
cd FinscaleNvidiaPlatform
nvidia-smi
python3 --version
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/smoke_test_cuda.py
```
Use a provider NVIDIA PyTorch/CUDA image if you later run GPU workflows; the source code download/section extraction is CPU-capable.

## 3. Download SEC EDGAR for 30 companies
Set a truthful SEC contact identity. Avoid synthetic or anonymous User-Agent values. Filing requests must respect SEC fair-access limits.
```bash
export SEC_USER_AGENT="Researcher Name researcher@yourdomain.com"
python scripts/download_edgar.py --limit 1 --pause 0.25
python - <<'PY'
import json,collections
p='data/raw/edgar_10k.jsonl'
rows=[json.loads(x) for x in open(p) if x.strip()]
print('filings:',len(rows),'unique companies:',len(set(r['ticker'] for r in rows)))
print('tickers:',sorted(set(r['ticker'] for r in rows)))
PY
```
To fetch two filings per ticker instead, use `--limit 2` (up to 60 filings). To override the 30 defaults, pass `--tickers NVDA AAPL ...` or `--tickers-file tickers.txt` (one ticker per line).

## 4. Build grounded SFT examples
The following default uses the NVIDIA model API as a *teacher*. API access may cost money and requires network connectivity and a valid key. The teacher model in Manish's generator is **Nemotron 3.5 Lightning 30B-A3B**; the **student training target remains Nemotron 3 Nano 30B-A3B**.
```bash
export NVIDIA_API_KEY="your_nvidia_api_key"    # preferably inject via provider secrets
python scripts/build_sft_dataset.py --input data/raw/edgar_10k.jsonl --output-dir data/processed --workers 2
wc -l data/processed/train.jsonl data/processed/val.jsonl
```
Alternatively, to test the extraction workflow without calling NVIDIA's API:
```bash
python scripts/build_sft_dataset.py --no-api --input data/raw/edgar_10k.jsonl --output-dir data/processed
```
**Caution:** `--no-api` produces simplistic extractive pseudo-labels. Dataset split is row-level, and a company can appear in both train and validation. Do not cite validation as an independent company-held-out benchmark without constructing a separate company-level split.

## 5. NVIDIA official Nano3 30B LoRA / SFT handoff
NVIDIA's official model and recipe tooling is **not** the older 8B Hugging Face fallback. Do not run `scripts/train_hf_lora.py` for this 30B target.
```bash
git clone https://github.com/NVIDIA-NeMo/Nemotron.git /workspace/Nemotron
cd /workspace/Nemotron
uv sync
uv run nemotron nano3 sft --help
```
Follow [NVIDIA's Nano3 guide](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/docs/nemotron/nano3) for W&B, NeMo/Slurm profile, checkpoint import, **Packed-Parquet SFTDataArtifact** preprocessing, and the correct LoRA settings in your pinned revision. Raw `data/processed/train.jsonl` is **not** directly accepted by the official Nano3 SFT command. Convert/register it with the official tooling; then inspect `configs/lora_sft.yaml` and the guarded wrapper:
```bash
cd /workspace/FinscaleNvidiaPlatform
NEMOTRON_REPO=/workspace/Nemotron bash scripts/run_nemotron_sft.sh
# Deliberately exits until the prerequisites have been configured.
# Only after successfully registering artifacts/configuring a valid profile:
export NEMOTRON_RUN_PROFILE="your-configured-cluster-profile"
export RUN_NEMOTRON_SFT=1
NEMOTRON_REPO=/workspace/Nemotron bash scripts/run_nemotron_sft.sh
```
**No GPU 30B training has been executed or verified in this integration.** Do not set `RUN_NEMOTRON_SFT=1` until the configuration is complete. Capture GPU utilization, steps/sec, loss, and single-vs-multi-node measurements as real results.

## Model, data and security notes
- NVIDIA checkpoint: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
- NVIDIA official developer repo: https://github.com/NVIDIA-NeMo/Nemotron
- Training / hardware guide: https://docs.nvidia.com/nemo-helix/documentation/customizer-reference/models/llama-nemotron/
- Never commit API tokens, real SEC datasets, adapters, model weights or checkpoints.
- The legacy Qwen scripts, benchmark metrics and charts were intentionally removed from this branch's current tree; see `main` for earlier experiments.
- Attribution: SEC/Nemotron preparation scripts adapted from Manish's `mkchoithani/finscale` branch `edgar-nemotron-runpod`.
