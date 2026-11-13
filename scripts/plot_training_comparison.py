#!/usr/bin/env python3
"""Plot measured, not invented, single-vs-two-node FinScale training metrics.
CSV columns: run_id,nodes,model,precision,seq_length,global_batch_size,step_seconds,tokens,loss
Each row is a completed optimizer step. Use repeated comparable runs.
"""
import argparse, csv, statistics
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="results/benchmark_charts")
    a = p.parse_args()
    rows = list(csv.DictReader(open(a.csv, newline="")))
    if not rows:
        raise SystemExit("CSV empty")
    required = {"run_id","nodes","model","precision","seq_length","global_batch_size","step_seconds","tokens","loss"}
    if not required.issubset(rows[0]):
        raise SystemExit("Missing columns: " + ", ".join(sorted(required - set(rows[0]))))
    groups = defaultdict(list)
    for row in rows:
        seconds = float(row["step_seconds"])
        tokens = float(row["tokens"])
        if seconds <= 0 or tokens <= 0:
            raise SystemExit("Nonpositive measured time or tokens")
        groups[int(row["nodes"])].append({**row,"tps":tokens/seconds,"seconds":seconds,"loss":float(row["loss"])})
    # Do not compare differing workloads as if equivalent.
    signatures = {tuple((r[k] for k in ("model","precision","seq_length","global_batch_size"))) for rs in groups.values() for r in rs}
    if len(signatures) != 1:
        raise SystemExit("Incomparable workloads; align model/precision/seq_length/global_batch_size")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ns = sorted(groups)
    med_tps = [statistics.median(r["tps"] for r in groups[n]) for n in ns]
    med_sec = [statistics.median(r["seconds"] for r in groups[n]) for n in ns]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar([str(n) for n in ns], med_tps)
    ax.set(xlabel="GB10 nodes", ylabel="Measured training tokens/second", title="FinScale: steady-state training throughput")
    for i,v in enumerate(med_tps): ax.text(i,v,f"{v:.1f}",ha="center",va="bottom")
    fig.tight_layout(); fig.savefig(out/"throughput.png",dpi=180);plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar([str(n) for n in ns], med_sec)
    ax.set(xlabel="GB10 nodes", ylabel="Median optimizer-step seconds", title="FinScale: training step latency")
    fig.tight_layout();fig.savefig(out/"step_latency.png",dpi=180);plt.close(fig)
    if 1 in groups and 2 in groups:
        speedup = statistics.median(r["tps"] for r in groups[2])/statistics.median(r["tps"] for r in groups[1])
        print(f"Measured 2-node / 1-node throughput speedup: {speedup:.3f}x")
        print(f"Scaling efficiency: {100*speedup/2:.1f}%")
    print("Charts:",out/"throughput.png",out/"step_latency.png")
    for n in ns:
        print(f"{n} nodes: {len(groups[n])} measured steps, median {statistics.median(r['tps'] for r in groups[n]):.2f} tok/s")
    if any(len(groups[n]) < 3 for n in ns):
        print("WARNING: fewer than 3 steps for a node count; results are preliminary.")
if __name__ == "__main__":
    main()
