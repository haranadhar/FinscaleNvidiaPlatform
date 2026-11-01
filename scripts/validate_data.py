#!/usr/bin/env python3
"""Validate FinScale raw JSONL and ensure train/eval ticker isolation."""
import argparse, collections, json
from pathlib import Path
CATS=set("market credit liquidity operational regulatory_legal cybersecurity supply_chain other".split())
SEVS={"low","medium","high"}
def check(path):
    tickers=set(); counts=collections.Counter(); errors=[]
    for n,line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(),1):
        if not line.strip(): continue
        try:
            row=json.loads(line); label=row["label"]
            assert isinstance(row["ticker"],str) and row["ticker"].strip()
            assert isinstance(row["excerpt"],str) and row["excerpt"].strip()
            assert label["category"] in CATS
            assert str(label["severity"]).lower() in SEVS
            assert isinstance(label.get("evidence"),str) and label["evidence"].strip()
            assert isinstance(label.get("explanation"),str) and label["explanation"].strip()
            tickers.add(row["ticker"]);counts[label["category"]]+=1
        except (ValueError,KeyError,TypeError,AssertionError) as e: errors.append(f"{path}:{n}: invalid record ({e})")
    return tickers,counts,errors
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",type=Path,default=Path("data/train_raw.jsonl"))
    ap.add_argument("--eval",type=Path,default=Path("data/eval_raw.jsonl"))
    args=ap.parse_args()
    t,tc,te=check(args.train);e,ec,ee=check(args.eval)
    errors=te+ee
    if t&e: errors.append("Company leakage: "+", ".join(sorted(t&e)))
    print(json.dumps({"train_tickers":sorted(t),"eval_tickers":sorted(e),"train_categories":tc,"eval_categories":ec,"errors":errors},indent=2))
    if errors: raise SystemExit(1)
if __name__=="__main__": main()
