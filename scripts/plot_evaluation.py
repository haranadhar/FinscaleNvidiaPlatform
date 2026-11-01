#!/usr/bin/env python3
"""Generate presentation-ready FinScale evaluation charts from actual JSON results.

Usage:
  python scripts/plot_evaluation.py --results out/heldout/eval_results.json
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIELDS = [
    ("valid_json_rate", "Valid JSON", True),
    ("category_accuracy", "Category accuracy", True),
    ("category_macro_f1", "Category macro-F1", False),
    ("severity_accuracy", "Severity accuracy", True),
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("out/heldout/eval_results.json"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if not args.results.is_file():
        parser.error(f"Missing {args.results}; finish evaluate.py first.")
    result = json.loads(args.results.read_text(encoding="utf-8"))
    base, tuned = result["base"], result["fine_tuned"]
    count = base.get("examples", "?")
    labels = [name for _, name, _ in FIELDS]
    values_base = [float(base[key]) * 100 for key, _, _ in FIELDS]
    values_tuned = [float(tuned[key]) * 100 for key, _, _ in FIELDS]
    output_dir = args.output_dir or (args.results.parent / "charts")
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(FIELDS))
    fig, ax = plt.subplots(figsize=(12, 6.5))
    width = 0.35
    bars_base = ax.bar(x - width / 2, values_base, width, label="Base Qwen2.5-3B", color="#64748B")
    bars_tuned = ax.bar(x + width / 2, values_tuned, width, label="FinScale LoRA", color="#0F9D8A")
    for bars, scores in ((bars_base, base), (bars_tuned, tuned)):
        for bar, (key, _, is_pct) in zip(bars, FIELDS):
            value = float(scores[key])
            label = f"{value * 100:.1f}%" if is_pct else f"{value:.3f}"
            ax.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        fontsize=10, fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Score (%) — macro-F1 shown ×100")
    ax.set_title("FinScale | Held-out financial risk evaluation\nBase vs LoRA Fine-Tuned Qwen2.5-3B", fontsize=16, weight="bold", pad=20)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.015, f"Held-out evaluation | {count} examples | Source: {args.results.name}",
             ha="center", fontsize=10, color="#475569")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ("png", "pdf"):
        target = output_dir / f"finscale_comparison.{ext}"
        fig.savefig(target, dpi=300, bbox_inches="tight")
        print(f"Saved {target}")
    plt.close(fig)

if __name__ == "__main__":
    main()
