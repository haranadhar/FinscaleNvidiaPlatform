#!/usr/bin/env python3
"""Create evidence charts only from real collected CSVs; no fabricated series."""
import argparse, csv, pathlib, statistics
def records(path):
    with open(path,newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def num(v):
    try: return float(v)
    except (ValueError,TypeError): return None
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dir",required=True)
    a=p.parse_args(); d=pathlib.Path(a.dir)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lines=[];gpu=d/"gpu_metrics.csv"
    if gpu.exists():
        rows=records(gpu);indices=sorted({r.get("gpu_index","") for r in rows})
        plots=[("gpu_util_pct","GPU compute utilization","Percent","gpu_utilization.png"),("gpu_memory_used_mib","GPU memory reported by nvidia-smi","MiB (may be unsupported on GB10)","gpu_memory.png"),("host_mem_available_kib","Host available shared memory","KiB","host_memory.png"),("gpu_power_w","Reported GPU power","Watts","gpu_power.png"),("gpu_temp_c","GPU temperature","Celsius","gpu_temperature.png")]
        for col,title,ylabel,name in plots:
            fig,ax=plt.subplots(figsize=(10,4))
            found=False
            for i in indices:
                series=[(num(r.get("elapsed_s")),num(r.get(col))) for r in rows if r.get("gpu_index","")==i]
                series=[(x,y) for x,y in series if x is not None and y is not None]
                if series:
                    ax.plot(*zip(*series),label="GPU "+i)
                    found=True
                    lines.append((col,i,round(statistics.mean(y for x,y in series),3),len(series)))
            if found:
                ax.set(title=title,xlabel="Elapsed seconds",ylabel=ylabel);ax.grid(alpha=.3);ax.legend()
                fig.tight_layout();fig.savefig(d/name,dpi=140)
            plt.close(fig)
    bm=d/"gpu_benchmark.csv"
    if bm.exists():
        rows=records(bm)
        xy=[(num(r.get("elapsed_s")),num(r.get("estimated_tflops"))) for r in rows]
        xy=[(x,y) for x,y in xy if x is not None and y is not None]
        if xy:
            fig,ax=plt.subplots(figsize=(10,4))
            ax.plot(*zip(*xy))
            ax.set(title="Synthetic FP16 matmul throughput (not Nemotron tokens/s)",xlabel="Elapsed seconds",ylabel="Estimated TFLOP/s")
            ax.grid(alpha=.3);fig.tight_layout();fig.savefig(d/"synthetic_matmul.png",dpi=140);plt.close(fig)
    train=d/"train_metrics.csv"
    if train.exists():
        rows=records(train)
        for col,out in [("loss","training_loss.png"),("tokens_per_second","training_tokens_per_second.png"),("step_time_s","step_time.png")]:
            xy=[(num(r.get("step")),num(r.get(col))) for r in rows];xy=[(x,y) for x,y in xy if x is not None and y is not None]
            if xy:
                fig,ax=plt.subplots(figsize=(10,4));ax.plot(*zip(*xy))
                ax.set(title="Actual "+col.replace("_"," "),xlabel="Training step",ylabel=col)
                ax.grid(alpha=.3);fig.tight_layout();fig.savefig(d/out,dpi=140);plt.close(fig)
    with (d/"telemetry_summary.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["field","gpu_index","mean","sample_count"]);w.writerows(lines)
    print("Graphs and telemetry summary created in",d)
    print("NOTE: training charts require real train_metrics.csv: step,loss,tokens_per_second,step_time_s")
if __name__=="__main__":main()
