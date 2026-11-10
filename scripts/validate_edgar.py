#!/usr/bin/env python3
"""Audit 10-K JSONL company coverage before preparing SFT examples."""
import argparse
from collections import Counter
import json
from pathlib import Path

DEFAULT_TICKERS = (
    "AAPL NVDA MSFT AMZN GOOGL META TSLA JPM BAC WFC "
    "GS MS V MA PYPL NFLX DIS WMT COST HD "
    "UNH JNJ PFE ABBV XOM CVX BA CAT AMD INTC"
).split()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/edgar_10k.jsonl"))
    parser.add_argument("--expected", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when expected tickers are missing.")
    args = parser.parse_args()
    found = Counter()
    invalid = 0
    with args.input.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                ticker = str(entry.get("ticker", "")).strip().upper()
                if not ticker or entry.get("form") != "10-K" or not entry.get("text"):
                    invalid += 1
                    continue
                found[ticker] += 1
            except (ValueError, TypeError):
                invalid += 1
                print(f"Malformed JSONL line {line_number}")
    expected = set(t.upper() for t in args.expected)
    missing = sorted(expected - found.keys())
    unexpected = sorted(found.keys() - expected)
    print(json.dumps({
        "requested_companies": len(expected),
        "retrieved_companies": len(found),
        "filings": sum(found.values()),
        "filings_by_ticker": dict(sorted(found.items())),
        "missing_tickers": missing,
        "unexpected_tickers": unexpected,
        "invalid_records": invalid,
    }, indent=2))
    if args.strict and (missing or invalid):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
