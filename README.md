# FinscaleNvidiaPlatform
# FinScale: Financial Risk Classification Using NVIDIA GPU and LoRA Fine-Tuning

## Overview

**FinScale** is an AI-powered financial risk classification project that fine-tunes the Qwen2.5-3B-Instruct large language model (LLM) using Low-Rank Adaptation (LoRA) to identify and classify financial risks from corporate financial disclosures and SEC filings.

The project leverages NVIDIA GPU acceleration, PyTorch, Hugging Face Transformers, and Parameter-Efficient Fine-Tuning (PEFT) to classify financial disclosures into structured JSON outputs.

The classification framework identifies financial risk categories, severity levels, supporting evidence, and explanations.

The project is designed to evaluate the effectiveness of domain-specific LLM fine-tuning for financial risk analysis, with a focus on classification accuracy, structured output reliability, and generalization across previously unseen companies.

### Key Capabilities

- Financial risk identification from SEC EDGAR filings.
- Automated financial disclosure labeling using Qwen2.5-7B-Instruct.
- LoRA-based fine-tuning of Qwen2.5-3B-Instruct.
- NVIDIA GPU-accelerated training using PyTorch and PEFT.
- Structured JSON generation with category, severity, evidence, and explanation.
- Base model versus fine-tuned model evaluation.
- Independent evaluation using held-out company disclosures.
- Training metrics, evaluation reports, and experiment reproducibility.

## Architecture and Technology Stack

### Model Architecture

| Component | Technology |
|---|---|
| Base LLM | Qwen2.5-3B-Instruct |
| Data Labeling Model | Qwen2.5-7B-Instruct |
| Fine-Tuning Approach | LoRA |
| Training Framework | PyTorch |
| Fine-Tuning Library | Hugging Face PEFT |
| Model Framework | Hugging Face Transformers |
| GPU Acceleration | NVIDIA CUDA |
| Dataset Processing | Hugging Face Datasets |
| GPU Environment | Lambda Labs |
| Initial GPU Configuration | NVIDIA A100-SXM4-40GB |
| Operating System | Ubuntu Linux |
| Python | Python 3.10 |
| Experiment Tracking | JSON metrics and training logs |

### High-Level Workflow

1. SEC EDGAR financial filing acquisition.
2. Financial disclosure extraction and preprocessing.
3. Automated financial risk labeling.
4. Dataset validation and preparation.
5. Training, validation, and internal test splitting.
6. Qwen2.5-3B-Instruct LoRA fine-tuning.
7. LoRA adapter generation and model evaluation.
8. Independent held-out company evaluation.
9. Performance comparison and experiment reporting.

## Financial Risk Classification Categories

FinScale currently supports the following eight financial risk categories:

| Category | Description |
|---|---|
| market | Market fluctuations and financial market exposure |
| credit | Credit exposure and counterparty risks |
| liquidity | Liquidity and cash-flow risks |
| operational | Operational disruptions and business risks |
| regulatory_legal | Regulatory compliance and legal risks |
| cybersecurity | Cybersecurity threats and data security risks |
| supply_chain | Supply chain disruptions and dependencies |
| other | Other financial and business risks |

**Severity levels:** Low, Medium, High.

### Expected Model Output

The model generates structured JSON containing:

- `category`
- `severity`
- `evidence`
- `explanation`

The evidence field identifies a supporting passage from the financial disclosure, while the explanation provides the reasoning behind the classification.

## NVIDIA GPU Environment

The initial NVIDIA implementation was executed using Lambda Labs GPU infrastructure.

**Reference GPU configuration:**

- GPU: NVIDIA A100-SXM4-40GB
- GPU Memory: 40 GB
- OS: Ubuntu Linux
- Python: 3.10
- Framework: PyTorch
- CUDA: NVIDIA CUDA-enabled PyTorch environment

The implementation uses the PyTorch/PEFT backend rather than the original Apple Silicon MLX backend.

### Repository Structure

```text
finscale/
│
├── train.py
├── evaluate.py
├── requirements.txt
│
├── scripts/
│   ├── download_edgar.py
│   ├── make_data_nvidia.py
│   ├── validate_data.py
│   └── prepare_heldout.py
│
├── data/
│   ├── train_raw.jsonl
│   ├── eval_raw.jsonl
│   └── nvidia/
│       ├── train.jsonl
│       ├── valid.jsonl
│       ├── test.jsonl
│       └── heldout_test.jsonl
│
└── out/
    ├── adapters/
    ├── metrics.json
    ├── eval_results.json
    ├── eval_results.md
    └── train_peft.log
```

*Note: Some NVIDIA-specific helper scripts are maintained separately from the original upstream repository and must be available in the project copy.*

## Installation and Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/mkchoithani/finscale.git
cd finscale

git status
```

For an existing repository:

```bash
cd ~/finscale
git pull --ff-only
git status
```

### 2. Verify NVIDIA GPU and CUDA

```bash
nvidia-smi
python3 --version

python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available')"
```

Verify that PyTorch detects the assigned NVIDIA GPU and CUDA is available.

Do not reinstall PyTorch if the Lambda Labs image already provides a working CUDA-enabled installation.

### 3. Create Python Virtual Environment

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate

python -m pip install --upgrade pip
```

The `--system-site-packages` option allows the virtual environment to access the existing CUDA-enabled PyTorch installation.

### 4. Install Compatible Dependencies

The following package combination was used to resolve compatibility issues during the NVIDIA implementation.

```bash
python -m pip install \
  "numpy==1.24.4" \
  "huggingface-hub==0.36.2" \
  "tokenizers==0.20.3" \
  "transformers==4.46.3" \
  "datasets==3.3.2" \
  "peft" \
  "accelerate" \
  "safetensors" \
  "pillow" \
  "jinja2>=3.1.0"
```

Disable optional TensorFlow and Flax integrations for the PyTorch training workflow:

```bash
export USE_TF=0
export USE_FLAX=0
```

Validate dependencies:

```bash
python -m pip check
```

Verify imports and CUDA:

```bash
python -c "import torch, transformers, datasets; from transformers import Trainer, TrainingArguments; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('Transformers:', transformers.__version__); print('Datasets:', datasets.__version__)"
```

## Data Acquisition and Preparation

### 1. SEC EDGAR Financial Disclosures

FinScale uses financial disclosure text obtained from SEC EDGAR filings.

The SEC downloader is available at:

```text
scripts/download_edgar.py
```

Before execution, configure the User-Agent with an appropriate application identification and valid contact email.

Inspect the supported options:

```bash
python scripts/download_edgar.py --help
```

Follow SEC access guidelines and request-rate limits.

### 2. Financial Risk Data Generation

The NVIDIA-compatible data-generation workflow uses Qwen2.5-7B-Instruct to generate financial risk labels from SEC filing excerpts.

The data-generation helper is:

```text
scripts/make_data_nvidia.py
```

Execute:

```bash
python -u scripts/make_data_nvidia.py
```

The workflow generates two separate datasets:

| Dataset | Purpose |
|---|---|
| `data/train_raw.jsonl` | Financial risk records used for training |
| `data/eval_raw.jsonl` | Independent held-out evaluation records |

The initial workflow reserves AMZN and XOM disclosures for the held-out evaluation dataset.

The generated records contain ticker, excerpt, and label information.

### 3. Validate Generated Data

Validate the JSONL records before model training.

```bash
python scripts/validate_data.py
```

Validation includes:

- JSON structure and record integrity.
- Required ticker, excerpt, and label fields.
- Classification category validation.
- Severity validation.
- Evidence and explanation availability.
- Dataset record counts and company tickers.

Ensure that held-out companies remain separate from the training dataset.

## Model Fine-Tuning Using NVIDIA GPU

### 1. Training Configuration

FinScale fine-tunes Qwen2.5-3B-Instruct using LoRA through the PyTorch/PEFT backend.

The initial NVIDIA training configuration:

| Parameter | Configuration |
|---|---|
| Base Model | Qwen/Qwen2.5-3B-Instruct |
| Backend | PyTorch / PEFT |
| Fine-Tuning | LoRA |
| Epochs | 1 |
| Batch Size | 1 |
| Maximum Sequence Length | 2048 |
| Learning Rate | 1e-5 |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| Evaluation Steps | 25 |

### 2. Execute Training

Activate the Python environment:

```bash
cd ~/finscale
source .venv/bin/activate

export USE_TF=0
export USE_FLAX=0
```

Execute GPU-accelerated training:

```bash
python -u train.py \
  --backend peft \
  --input data/train_raw.jsonl \
  --data-dir data/nvidia \
  --model Qwen/Qwen2.5-3B-Instruct \
  --epochs 1 \
  --batch-size 1 \
  --max-seq-length 2048 \
  --learning-rate 1e-5 \
  --lora-r 8 \
  --lora-alpha 16 \
  --eval-steps 25 \
  --out-dir out 2>&1 | tee out/train_peft.log
```

Monitor GPU utilization:

```bash
nvidia-smi
```

Inspect training logs:

```bash
tail -n 80 out/train_peft.log
```

### 3. Verify LoRA Adapter Generation

After training completes, verify the generated adapter files:

```bash
find out/adapters -maxdepth 2 -type f
```

Expected artifacts:

```text
out/adapters/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer_config.json
└── tokenizer.json
```

Review training metrics:

```bash
cat out/metrics.json
```

## Model Evaluation

FinScale evaluates the original base model against the fine-tuned model to measure the impact of domain-specific financial risk training.

### 1. Internal Test Dataset Evaluation

The internal test dataset is generated from the training source using the project's data-splitting workflow.

Execute:

```bash
python -u evaluate.py \
  --backend peft \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter out/adapters \
  --test data/nvidia/test.jsonl \
  --out-dir out
```

The evaluation compares:

- Valid JSON generation.
- Financial risk category accuracy.
- Category macro-F1 score.
- Severity classification accuracy.

### 2. Held-Out Company Evaluation

To measure generalization, FinScale maintains a separate evaluation dataset containing previously reserved company disclosures.

The initial held-out companies are:

- Amazon (AMZN)
- Exxon Mobil (XOM)

These records are maintained separately in:

```text
data/eval_raw.jsonl
```

Convert the held-out raw JSONL records into the chat-formatted evaluation schema:

```bash
python scripts/prepare_heldout.py
```

This generates:

```text
data/nvidia/heldout_test.jsonl
```

Run the independent evaluation:

```bash
python -u evaluate.py \
  --backend peft \
  --model Qwen/Qwen2.5-3B-Instruct \
  --adapter out/adapters \
  --test data/nvidia/heldout_test.jsonl \
  --out-dir out
```

**Important:** The held-out company evaluation is a separate evaluation stage from the internal test split. Preserve the results independently and verify that AMZN and XOM do not appear in the training data.

## Initial NVIDIA Experiment Results

The following results were observed during the initial NVIDIA GPU implementation using the internal test dataset.

### Base Model vs. Fine-Tuned Model

| Evaluation Metric | Base Model | Fine-Tuned Model |
|---|---:|---:|
| Valid JSON | 60.0% | 100.0% |
| Category Accuracy | 30.0% | 40.0% |
| Category Macro-F1 | 0.262 | 0.266 |
| Severity Accuracy | 65.0% | 65.0% |

### Results Interpretation

- **Structured output reliability:** Valid JSON generation improved from 60% to 100%.
- **Category classification:** Accuracy improved from 30% to 40%.
- **Category Macro-F1:** Increased from 0.262 to 0.266, indicating limited improvement across risk categories.
- **Severity classification:** Accuracy remained unchanged at 65%.

These results demonstrate improved structured output generation following LoRA fine-tuning. However, classification performance requires further optimization and validation.

The initial results are directional and should not be considered production-level performance.

The held-out AMZN/XOM evaluation must be completed and reported independently before drawing conclusions about generalization.

## Experiment Artifacts and Reproducibility

Each experiment should retain the following artifacts:

```text
experiment/
├── package_versions.txt
├── train.log
├── train_metrics.json
├── internal_test_results.json
├── internal_test_results.md
├── heldout_results.json
├── heldout_results.md
└── README.md
```

Capture package versions:

```bash
python -m pip freeze > out/package_versions.txt
```

Record the following information for every experiment:

- GPU model and configuration.
- Base model identifier.
- Dataset version and record counts.
- Training configuration and LoRA parameters.
- Python and dependency versions.
- Training logs and evaluation metrics.
- Internal test and held-out evaluation results.

Do not commit Hugging Face tokens, SEC API credentials, or other private credentials to GitHub.

## Troubleshooting

| Issue | Resolution |
|---|---|
| CUDA unavailable | Verify the Lambda GPU instance and CUDA-enabled PyTorch installation. |
| NumPy/Transformers import errors | Use the compatible dependency versions specified above. |
| `tf_keras` missing | Set `USE_TF=0` and `USE_FLAX=0`. |
| `BucketNotFoundError` | Verify compatibility between Datasets and Hugging Face Hub. |
| `mlx_lm` missing | Explicitly select `--backend peft`. |
| `adapter_config.json` missing | Verify successful training and adapter generation before evaluation. |
| SyntaxError in notebook | Execute shell commands using `!` or `%%bash`, or use a terminal. |
| Training process failure | Inspect `out/train_peft.log` and GPU utilization. |

**Important:** MLX is intended for Apple Silicon execution. All NVIDIA GPU training and evaluation operations must explicitly use the PEFT backend.

## Future Roadmap

### Model and Classification Improvements

- Expand financial disclosure datasets.
- Improve underrepresented financial risk categories.
- Increase human-reviewed training labels.
- Experiment with different LoRA ranks and learning rates.
- Extend training epochs and evaluate larger datasets.
- Improve category-level precision, recall, and macro-F1.

### NVIDIA GPU and Distributed Training

- Extend the current single-node NVIDIA GPU implementation.
- Evaluate NVIDIA GB10 Blackwell systems.
- Explore multi-node distributed training using PyTorch and NCCL.
- Investigate orchestration through NVIDIA Fuzzball.
- Measure distributed training performance and scaling efficiency.

### AI Model Deployment

- Develop an inference-serving architecture.
- Explore OpenAI-compatible API endpoints.
- Integrate financial risk classification into downstream financial analytics workflows.
- Evaluate scalability and inference performance.

## Project Repository

GitHub: [(https://github.com/haranadhar/FinscaleNvidiaPlatform)]

## Data and Licensing

The project uses public, non-proprietary financial disclosures and generated classification examples.

The Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct models are subject to their respective licensing terms published on Hugging Face.

SEC EDGAR data usage must comply with applicable SEC access policies and requirements.

---

**FinScale — NVIDIA GPU-Accelerated Financial Risk Classification Using Large Language Models and Parameter-Efficient Fine-Tuning.**
