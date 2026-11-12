#!/usr/bin/env python3
"""Bounded synthetic CUDA matmul benchmark (not the GPU-Burn project)."""
import argparse, csv, datetime, pathlib, time
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seconds",type=float,default=45)
    p.add_argument("--size",type=int,default=2048)
    p.add_argument("--output",default="outputs/gpuwerk/gpu_benchmark.csv")
    a=p.parse_args()
    if a.seconds<=0 or a.seconds>300: p.error("seconds must be 0 < seconds <= 300")
    if a.size<256 or a.size>4096: p.error("size must be 256..4096")
    import torch
    if not torch.cuda.is_available(): raise SystemExit("CUDA torch unavailable; use provider-compatible PyTorch environment")
    device="cuda:0"
    x=torch.randn(a.size,a.size,device=device,dtype=torch.float16)
    y=torch.randn(a.size,a.size,device=device,dtype=torch.float16)
    for _ in range(5):z=x@y
    torch.cuda.synchronize()
    dest=pathlib.Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    with dest.open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["timestamp_utc","elapsed_s","matmul_ms","estimated_tflops"])
        while time.monotonic()-start<a.seconds:
            t=time.perf_counter()
            z=x@y
            torch.cuda.synchronize()
            ms=(time.perf_counter()-t)*1000
            flops=2*a.size**3
            w.writerow([datetime.datetime.now(datetime.timezone.utc).isoformat(),round(time.monotonic()-start,3),round(ms,3),round(flops/(ms/1000)/1e12,5)])
            f.flush()
    print("Saved",dest,"(synthetic matmul, not real model throughput)")
if __name__=="__main__":main()
