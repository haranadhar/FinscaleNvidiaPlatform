import json
import random
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

CATEGORIES = [
    "market", "credit", "liquidity", "operational",
    "regulatory_legal", "cybersecurity", "supply_chain", "other"
]
SEVERITIES = ["low", "medium", "high"]

SYSTEM = (
    "You are a financial risk analyst. Classify the risk disclosure excerpt. "
    "Respond with ONLY a JSON object with keys: "
    '"category" (one of ' + ", ".join(CATEGORIES) + '), '
    '"severity" (one of low, medium, high), '
    '"evidence" (a short verbatim quote from the excerpt, max 25 words), '
    '"explanation" (1-2 sentences justifying the label, grounded in evidence).'
)

CHUNKS_PER_FILE = 25
MIN_WORDS, MAX_WORDS = 60, 220


def chunk_text(text):
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    chunks, current = [], []

    for paragraph in paragraphs:
        if len(paragraph.split()) > MAX_WORDS:
            if current:
                words = sum(len(x.split()) for x in current)
                if MIN_WORDS <= words <= MAX_WORDS:
                    chunks.append(" ".join(current))
                current = []
            continue

        current.append(paragraph)
        words = sum(len(x.split()) for x in current)

        if words >= MIN_WORDS:
            if words <= MAX_WORDS:
                chunks.append(" ".join(current))
            current = []

    return chunks


def parse_label(raw, excerpt):
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return None

    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    category = result.get("category")
    severity = result.get("severity")
    evidence = result.get("evidence", "").strip()
    explanation = result.get("explanation", "").strip()

    if category not in CATEGORIES or severity not in SEVERITIES:
        return None
    if not evidence or not explanation:
        return None

    normalize = lambda s: re.sub(r"\s+", " ", s.lower()).strip(" .\"'")
    if normalize(evidence) not in normalize(excerpt):
        return None

    if len(evidence.split()) > 25:
        return None

    return {
        "category": category,
        "severity": severity,
        "evidence": evidence,
        "explanation": explanation
    }


def main():
    random.seed(7)

    files = sorted(Path("data/raw").glob("*.txt"))
    if not files:
        raise SystemExit(
            "No SEC text files found in data/raw. "
            "Run python scripts/download_edgar.py first."
        )

    print("Loading NVIDIA-compatible model:", MODEL_ID)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    by_company = {}
    for file in files:
        company = file.stem.split("_")[0]
        by_company.setdefault(company, []).append(file)

    tickers = sorted(by_company)
    random.shuffle(tickers)

    n_eval = max(2, len(tickers) // 5)
    eval_tickers = set(tickers[:n_eval])

    print("Held-out evaluation companies:", sorted(eval_tickers))

    rows = {"train": [], "eval": []}

    for ticker, company_files in by_company.items():
        split = "eval" if ticker in eval_tickers else "train"

        for file in company_files:
            chunks = chunk_text(file.read_text(errors="replace"))
            random.shuffle(chunks)

            for excerpt in tqdm(chunks[:CHUNKS_PER_FILE], desc=file.name):
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": f"Excerpt:\n{excerpt}"
                    }
                ]

                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                inputs = tokenizer(
                    prompt,
                    return_tensors="pt"
                ).to(model.device)

                with torch.inference_mode():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=250,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )

                generated = output[0][inputs["input_ids"].shape[1]:]
                raw = tokenizer.decode(
                    generated,
                    skip_special_tokens=True
                )

                label = parse_label(raw, excerpt)
                if label is None:
                    continue

                rows[split].append({
                    "ticker": ticker,
                    "excerpt": excerpt,
                    "label": label
                })

    Path("data").mkdir(exist_ok=True)

    for split, items in rows.items():
        output_path = Path(f"data/{split}_raw.jsonl")
        with output_path.open("w") as output_file:
            for item in items:
                output_file.write(json.dumps(item) + "\n")

        print(f"{split}: {len(items)} records -> {output_path}")

    print("Dataset generation complete.")


if __name__ == "__main__":
    main()
