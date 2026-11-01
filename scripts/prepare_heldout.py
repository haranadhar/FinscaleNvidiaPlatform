#!/usr/bin/env python3
"""Convert held-out raw FinScale records into evaluate.py chat JSONL."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import parse_records, chat_example

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,default=Path("data/eval_raw.jsonl"))
    ap.add_argument("--output",type=Path,default=Path("data/nvidia/heldout_test.jsonl"))
    args=ap.parse_args()
    records,errors=parse_records(args.input)
    if errors: raise ValueError("Invalid input: "+"; ".join(errors[:10]))
    if not records: raise ValueError("No held-out records")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf-8") as f:
        for r in records: f.write(json.dumps(chat_example(r),ensure_ascii=False)+"\n")
    print(f"Wrote {len(records)} held-out records to {args.output}")
if __name__=="__main__": main()
