#!/usr/bin/env python3
"""Portable, best-effort NVIDIA GPU/host telemetry. Missing metrics stay blank."""
import argparse, csv, datetime, json, os, pathlib, subprocess, time

FIELDS = ["index","name","utilization.gpu","utilization.memory","memory.used","memory.total","temperature.gpu","power.draw","clocks.sm"]
HEAD = ["timestamp_utc","elapsed_s","gpu_index","gpu_name","gpu_util_pct","memory_util_pct","gpu_memory_used_mib","gpu_memory_total_mib","gpu_temp_c","gpu_power_w","sm_clock_mhz","host_mem_available_kib","host_mem_total_kib"]
def run(args):
    try:
        x=subprocess.run(args, capture_output=True, text=True, timeout=6)
        return x.stdout.strip() if x.returncode==0 else ""
    except (OSError,subprocess.TimeoutExpired):
        return ""
def host_mem():
    values={}
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            k,_,v=line.partition(":")
            if k in ("MemTotal","MemAvailable"): values[k]=v.strip().split()[0]
    except OSError: pass
    return values
def sample(fields):
    out=run(["nvidia-smi","--query-gpu="+",".join(fields),"--format=csv,noheader,nounits"])
    if not out: return []
    return list(csv.reader(out.splitlines(), skipinitialspace=True))
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--interval",type=float,default=2)
    p.add_argument("--duration",type=float,default=0,help="0 = run until Ctrl+C")
    a=p.parse_args()
    if a.interval < 0.5: p.error("interval must be >=0.5 seconds")
    dest=pathlib.Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    # Query separately: a single unsupported field can cause the whole query to fail.
    supported=[f for f in FIELDS if sample([f])]
    if "index" not in supported: raise SystemExit("nvidia-smi index query failed: no GPU access")
    names=dict(zip(supported,supported))
    print("Supported GPU telemetry fields:", ", ".join(supported),flush=True)
    print("Unavailable fields are blank; DGX Spark shares CPU/GPU unified memory.",flush=True)
    start=time.monotonic()
    with dest.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(HEAD)
        try:
            while True:
                elapsed=time.monotonic()-start
                if a.duration and elapsed>=a.duration: break
                now=datetime.datetime.now(datetime.timezone.utc).isoformat()
                rows=sample(supported);mem=host_mem()
                for vals in rows:
                    dat=dict(zip(supported,[x.strip() for x in vals]))
                    def v(x):return dat.get(x,"")
                    w.writerow([now,round(elapsed,3),v("index"),v("name"),v("utilization.gpu"),v("utilization.memory"),v("memory.used"),v("memory.total"),v("temperature.gpu"),v("power.draw"),v("clocks.sm"),mem.get("MemAvailable",""),mem.get("MemTotal","")])
                f.flush()
                time.sleep(max(0,a.interval-(time.monotonic()-start-elapsed)))
        except KeyboardInterrupt: pass
    print(f"Saved telemetry: {dest}")
if __name__=="__main__":main()
