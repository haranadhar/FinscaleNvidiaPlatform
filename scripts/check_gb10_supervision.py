#!/usr/bin/env python3
"""Check that FinScale ChatDataset produces real supervised labels."""
import argparse
import torch
from transformers import AutoTokenizer
from nemo_automodel.components.datasets.llm.chat_dataset import ChatDatasetConfig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    found = False
    for length in args.lengths:
        ds = ChatDatasetConfig(
            path_or_dataset_id=args.data,
            seq_length=length,
            truncation="longest_first",
        ).build(tokenizer=tok)
        positives = 0
        print(f"seq_length={length} samples={len(ds)}", flush=True)
        for idx in range(min(args.limit, len(ds))):
            row = ds[idx]
            labels = torch.as_tensor(row["labels"])
            supervised = int((labels != -100).sum().item())
            positives += supervised > 0
            print(f"  row={idx} tokens={len(row['input_ids'])} supervised={supervised}", flush=True)
        print(f"positive_rows={positives}", flush=True)
        if positives:
            found = True
            print(f"USABLE: at least one supervised example at seq_length={length}", flush=True)
            break
    if not found:
        raise SystemExit("FAIL: no supervised labels in checked examples. Do not train.")
if __name__ == "__main__":
    main()
