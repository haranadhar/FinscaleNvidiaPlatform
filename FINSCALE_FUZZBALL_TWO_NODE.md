# FinScale: Fuzzball + two DGX Spark GB10 nodes — hackathon execution plan

**Target:** two distinct DGX Spark GB10 hosts, CIQ Fuzzball scheduler, NCCL connectivity, a real distributed Nemotron training optimizer step, fair performance comparison, reproducible dashboard. **Not yet executed or verified.** Fuzzball v4.3 deployment/job syntax depends on the event cluster; do **not** assume a local Fuzzball server, Docker daemon, cloud install, license or specific CLI flags.

Start with [GB10 single-node runbook](GB10_HACKATHON_RUNBOOK.md). Two-node experiments should begin only after the single-node supervised-token issue and one-step model execution are addressed.

## A. What to obtain from hackathon staff (before modifying infrastructure)

Ask for: (1) Fuzzball Web UI/CLI endpoint and login, (2) project/namespace/cluster identifier, (3) exact supported workflow YAML example for **two separate DGX Spark hosts**, (4) node image and persistent/shared dataset/model storage mount, (5) Fuzzball GPU resource naming and rank-launch integration, (6) cross-node interface and permitted TCP ports, (7) whether NCCL over Spark interconnect is configured, (8) checkpoint artifact location. Do not deploy a second Fuzzball control plane on AWS, pay a separate license, or assume the event requires it. Follow staff-provided Fuzzball 4.3 example for actual workflow syntax.

Useful docs: https://docs.ciq.com/fuzzball/ . Save submitted workflow YAML, scheduler job ID, and the Web UI screenshot showing both host allocations.

## B. Stage identical environment on BOTH GB10s

Use the event's supported ARM64/CUDA image. Copy project branch and dataset to a shared mount or stage separately; record hashes. The 59GiB model is not in GitHub and must be staged separately, ideally on staff-provided shared storage. GPUWerk-specific `/workspace/automodel-test-env` and `/workspace/models/nemotron-30b` will not exist on event machines automatically.

```bash
git clone --branch Nemotron --single-branch https://github.com/haranadhar/FinscaleNvidiaPlatform.git
cd FinscaleNvidiaPlatform
git rev-parse HEAD
uname -m
hostname -f
nvidia-smi -L
python3 -c 'import torch; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))'
sha256sum data/processed/train.jsonl  # after restoring same dataset to both hosts
```

Record node name, IP, CUDA/driver, PyTorch, NeMo AutoModel git SHA, dataset/model SHA, CPU/memory, actual NIC speed, and clock sync. Verify **both hosts are distinct**.

## C. Network preflight (not training)

Do not hard-code `ens5`: inspect `ip -br addr`, `ip route get PEER_IP`, and event networking. On node 0 (example only, staff must authorize port):

```bash
python3 -m http.server 29500 --bind 0.0.0.0
```

On node 1:

```bash
curl --connect-timeout 5 http://NODE0_PRIVATE_IP:29500/
```

Stop HTTP server before torchrun. This tests one rendezvous TCP port **only**. NCCL may use additional dynamic data ports; work with staff on documented firewall ranges, do not open 0–65535. Set `NCCL_SOCKET_IFNAME` only after determining the actual reachable interface. `NCCL_IB_DISABLE=1` only if fabric diagnostics warrant TCP fallback; disabling RDMA may reduce throughput.

## D. Two-host NCCL all-reduce — first concrete proof

On both hosts, in same env, run with actual node0 address and interface. Example assumes **one GB10 GPU per host** and that both shells can reach each other directly:

Node 0:
```bash
export MASTER_ADDR=NODE0_PRIVATE_IP MASTER_PORT=29500
export NCCL_DEBUG=INFO
export FINSCALE_EVIDENCE_DIR=/SHARED/results/nccl
torchrun --nnodes=2 --nproc-per-node=1 --node-rank=0 \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  scripts/nccl_multinode_proof.py 2>&1 | tee node0-nccl.log
```

Node 1 (run concurrently):
```bash
export MASTER_ADDR=NODE0_PRIVATE_IP MASTER_PORT=29500
export NCCL_DEBUG=INFO
export FINSCALE_EVIDENCE_DIR=/SHARED/results/nccl
torchrun --nnodes=2 --nproc-per-node=1 --node-rank=1 \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  scripts/nccl_multinode_proof.py 2>&1 | tee node1-nccl.log
```

Replace `/SHARED` with real shared mount or use separate output paths and copy files together afterward. **Expected:** rank 0 and 1 from different hostnames, `world_size=2`, all-reduce sum=3, two rank JSON files. This demonstrates distributed communication, **not yet distributed model training**. The Fuzzball equivalent must launch these same two workers on separate hosts using the staff-provided workflow specification. Capture the actual workflow and rank logs.

## E. Distributed Nemotron training (higher-risk experiment)

1. **Baseline:** first complete single-GB10 LoRA optimizer step using [single-node runbook](GB10_HACKATHON_RUNBOOK.md). Preserve loss, time, tokens, checkpoint, config and environment.
2. **Select a supported multi-node recipe:** NVIDIA's single-GPU Nano3 YAML has `dp_size=1, ep_size=1` and cannot become a validated two-host recipe just by changing `--nproc-per-node`. Inspect official AutoModel examples and the hackathon's Fuzzball launcher. Distributed FSDP/DP or EP across hosts requires correct model partitioning, process groups and checkpointing. Prefer a supported two-rank recipe; do not invent one. Validate memory fit and all-reduce before scheduling the 30B job.
3. **Fuzzball launch:** adapt the event-provided workflow to request two GB10 nodes, identical image/mounts, one rank per node, rendezvous via rank-0 host, network env and logs. Verify job ID, two node allocations and per-rank CUDA execution.
4. **True training proof:** two rank logs show the same run ID, world size 2, distinct hosts, model loading, positive supervised tokens, forward, backward, optimizer step, finite loss, and saved/reloadable adapter. If no optimizer step finishes, report distributed setup/communication only.
5. **Fallback:** if 30B distributed path is blocked by memory, kernel or time, demonstrate *actual distributed training* of a smaller supported model on both hosts, label the fallback accurately, and separately show Nemotron 30B single-node results. Do not represent the fallback as distributed 30B.

**Checkpoint requirement:** upstream single-GPU recipe defaults to `checkpoint.enabled: false`; enable saving only after confirming the installed AutoModel checkpoint schema, and verify actual files and adapter reload. Model weights alone are not a trained adapter.

## F. Benchmarking and visual evidence

Collect at least three measured steady-state optimizer steps for each topology, excluding cold model load, first-run JIT compilation, dataset preparation and checkpoint upload. Prefer repeated independent runs with fixed seeds. Keep **same model, precision, sequence length, effective global batch size, dataset and number of training tokens**. Two nodes may have a different optimum batch size; report that separately as a *throughput-at-capacity* experiment, not a like-for-like speedup.

Per completed step capture `run_id,nodes,model,precision,seq_length,global_batch_size,step_seconds,tokens,loss`. Count **actual non-padding input/target tokens processed**, document counting convention; do not substitute configured maximum length. Store `rank`, hostname, CUDA/driver, NIC, optimizer step, and wall-clock timestamps separately. GPU memory on GB10 unified memory can show N/A; record CPU RAM and `nvidia-smi` alongside, do not invent VRAM usage.

Metrics:
- Throughput = processed tokens / measured optimizer-step wall time (tokens/s).
- Speedup = two-node throughput / single-node throughput (same workload).
- Parallel efficiency = speedup / 2 × 100%.
- Communication = measured NCCL collective times; separately report end-to-end optimizer-step latency.
- Quality = held-out loss and actual task metrics, with dataset split/leakage disclosed.

Prepare a CSV from **real log measurements only**, then render:

```bash
python scripts/plot_training_comparison.py \
  --csv results/training_steps.csv \
  --out results/benchmark_charts
```

Charts: `throughput.png` and `step_latency.png`. Include a final slide with **(1) architecture diagram, (2) Fuzzball two-host allocation, (3) rank0/rank1 NCCL logs, (4) 1 vs 2-node measured tokens/s and speedup, (5) checkpoint and inference result, (6) caveats**. The chart script rejects incompatible workload signatures and empty/fabricated data.

## G. Demo sequence and ownership

| Time order | Demo | Pass condition |
|---|---|---|
| 1 | SEC document + labeled record | Real filing, traceable label and source quote |
| 2 | Single-node Nemotron 30B | Real forward/backward/optimizer step and finite loss |
| 3 | Fuzzball scheduler | Two distinct GB10 hosts visibly allocated |
| 4 | NCCL communication | Two hostnames, two ranks, sum=3 |
| 5 | Two-node training | Actual two-rank model optimizer step and checkpoint |
| 6 | Dashboard | Measured tokens/s, step latency, scaling efficiency, task evaluation |
| 7 | Reproducibility | Commit SHA, dataset hash, config, log, workflow, adapter and rerun instructions |

Roles: infrastructure lead owns GB10/CUDA/NCCL/Fuzzball and measurements; model engineer owns LoRA/dataset/quality; app engineer owns API/demo; fourth member owns evidence capture, charts and presentation. Prioritize a working and honestly measured demo over speculative 30B scaling claims.

**Never claim a two-node 30B training win until the two-host optimizer step and reproducible evidence exist.**
