#!/usr/bin/env python3
"""Generate an experimental FinScale config from the installed official Nano3 recipe."""
import argparse
from pathlib import Path
import yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="/workspace/Automodel/examples/llm_finetune/nemotron/nemotron_nano_v3_singlegpu_lora.yaml")
    ap.add_argument("--model", default="/workspace/models/nemotron-30b")
    ap.add_argument("--data", default="data/processed/train.jsonl")
    ap.add_argument("--output", default="configs/nemotron_gb10_finscale.yaml")
    ap.add_argument("--seq-length", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=1)
    args = ap.parse_args()
    if args.steps < 1 or args.seq_length < 128:
        ap.error("steps must be >=1 and seq-length >=128")
    source = Path(args.recipe)
    cfg = yaml.safe_load(source.read_text())
    expected = "nemo_automodel.components.datasets.llm.hellaswag.HellaSwag"
    if cfg.get("dataset", {}).get("_target_") != expected:
        raise SystemExit("Official recipe changed: inspect dataset schema manually")
    cfg["model"]["pretrained_model_name_or_path"] = str(Path(args.model).resolve())
    cfg["dataset"] = {
        "_target_": "nemo_automodel.components.datasets.llm.chat_dataset.ChatDatasetConfig",
        "path_or_dataset_id": str(Path(args.data).resolve()),
        "seq_length": args.seq_length,
        "truncation": "longest_first",
    }
    cfg["step_scheduler"]["max_steps"] = args.steps
    cfg["step_scheduler"]["global_batch_size"] = 1
    cfg["step_scheduler"]["local_batch_size"] = 1
    # Keep upstream checkpoint schema untouched; enable only after validated.
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print("Generated:", output)
    print("Checkpoint enabled:", cfg.get("checkpoint", {}).get("enabled"))
    print("NOTE: Validate positive supervised labels before training")
if __name__ == "__main__":
    main()
