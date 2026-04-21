#!/usr/bin/env python3
"""Run every eval_*.py in this directory against a single checkpoint.

Each per-benchmark script already exposes a `--checkpoint / --output / --device`
CLI and writes a JSON metrics file. This driver invokes them in-sequence as
subprocesses (separate processes = clean GPU memory between benchmarks +
isolates per-eval crashes), then aggregates the JSON outputs into one summary.

Usage:
    python scripts/evaluation/eval_all.py \\
        --checkpoint experiments/runs/lm-75m-2x4090/.../final.pt \\
        --output-dir experiments/results/lm-75m-2x4090 \\
        --fast                      # small per-task sample caps (~1 h on 2x4090)

    # Select / skip tasks
    python scripts/evaluation/eval_all.py --checkpoint X --tasks mmlu hellaswag
    python scripts/evaluation/eval_all.py --checkpoint X --skip gpqa hle
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parents[1]


@dataclass
class EvalTask:
    name: str                            # short task id (filename stem without eval_)
    script: str                          # filename under scripts/evaluation/
    metric_keys: List[str]               # candidate keys in the JSON to report as the headline number
    sample_flag: Optional[str] = None    # CLI flag controlling sample cap (None = unsupported)
    fast_sample_cap: Optional[int] = None  # value to pass with sample_flag under --fast
    supports_tokenizer: bool = True
    extra_args: Optional[List[str]] = None  # always-on extra args


# Output basenames match each per-script default so existing consumers still work.
TASKS: List[EvalTask] = [
    EvalTask("mmlu",           "eval_mmlu.py",           ["accuracy", "acc"],
             sample_flag="--max-samples-per-subject", fast_sample_cap=1),
    EvalTask("mmlu_pro",       "eval_mmlu_pro.py",       ["accuracy", "acc"],
             sample_flag="--max-samples", fast_sample_cap=500),
    EvalTask("hellaswag",      "eval_hellaswag.py",      ["acc_norm", "accuracy", "acc"],
             sample_flag="--max-samples", fast_sample_cap=500),
    EvalTask("anli",           "eval_anli.py",           ["accuracy", "acc", "r1_accuracy"],
             sample_flag="--max-samples", fast_sample_cap=500),
    EvalTask("commonsenseqa",  "eval_commonsenseqa.py",  ["accuracy", "acc"],
             sample_flag="--max-samples", fast_sample_cap=500),
    EvalTask("coqa",           "eval_coqa.py",           ["f1", "exact_match"],
             sample_flag="--max-docs", fast_sample_cap=50),
    EvalTask("glue",           "eval_glue.py",           ["average", "accuracy"],
             sample_flag="--max-samples", fast_sample_cap=200),
    EvalTask("gpqa",           "eval_gpqa.py",           ["accuracy", "acc"],
             sample_flag="--max-samples", fast_sample_cap=200),
    EvalTask("hle",            "eval_hle.py",            ["accuracy", "acc"],
             sample_flag="--max-samples", fast_sample_cap=200),
    EvalTask("ifbench_test",   "eval_ifbench_test.py",   ["accuracy", "strict_accuracy", "acc"],
             sample_flag="--max-samples", fast_sample_cap=200),
    EvalTask("ifeval",         "eval_ifeval.py",         ["strict_accuracy", "accuracy", "acc"],
             sample_flag="--max-samples", fast_sample_cap=200),
    EvalTask("mixeval_easy",   "eval_mixeval_easy.py",   ["accuracy", "acc"],
             sample_flag="--max-samples", fast_sample_cap=200),
    EvalTask("squad_v2",       "eval_squad_v2.py",       ["f1", "exact_match"],
             sample_flag="--max-samples", fast_sample_cap=200),
    EvalTask("sst2",           "eval_sst2.py",           ["accuracy", "acc"],
             sample_flag="--max-samples", fast_sample_cap=500, supports_tokenizer=False),
    EvalTask("gsm8k",          "eval_gsm8k.py",          ["accuracy", "pass@1"],
             sample_flag="--max-samples", fast_sample_cap=50, supports_tokenizer=False),
]


def build_cmd(task: EvalTask, args: argparse.Namespace, output_path: Path) -> List[str]:
    cmd = [sys.executable, str(EVAL_DIR / task.script),
           "--checkpoint", args.checkpoint,
           "--device", args.device,
           "--output", str(output_path)]
    if task.supports_tokenizer and args.tokenizer:
        cmd += ["--tokenizer", args.tokenizer]
    if args.fast and task.sample_flag and task.fast_sample_cap is not None:
        cmd += [task.sample_flag, str(task.fast_sample_cap)]
    if task.extra_args:
        cmd += task.extra_args
    return cmd


def extract_metric(data, keys: List[str]) -> Optional[float]:
    if isinstance(data, list):
        data = data[-1] if data and isinstance(data[-1], dict) else {}
    if not isinstance(data, dict):
        return None
    for k in keys:
        v = data.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    overall = data.get("overall")
    if isinstance(overall, dict):
        for k in keys:
            v = overall.get(k)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def run_task(task: EvalTask, args: argparse.Namespace) -> dict:
    out_path = Path(args.output_dir) / f"{task.name}_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(task, args, out_path)
    print(f"\n[eval_all] >>> {task.name}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True)
        elapsed = time.time() - t0
    except subprocess.CalledProcessError as e:
        return {"task": task.name, "status": "failed", "returncode": e.returncode,
                "elapsed_s": time.time() - t0, "output": str(out_path)}

    data = {}
    if out_path.exists():
        try:
            data = json.loads(out_path.read_text())
        except Exception as exc:
            return {"task": task.name, "status": "parse_error", "error": str(exc),
                    "elapsed_s": elapsed, "output": str(out_path)}
    try:
        metric = extract_metric(data, task.metric_keys)
    except Exception as exc:
        return {"task": task.name, "status": "metric_error", "error": str(exc),
                "elapsed_s": elapsed, "output": str(out_path)}
    return {"task": task.name, "status": "ok", "metric": metric,
            "metric_keys_tried": task.metric_keys, "elapsed_s": elapsed,
            "output": str(out_path), "raw": data}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all eval_*.py against one checkpoint and aggregate results.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer path/name (leave empty to let each script use its default).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="experiments/results",
                        help="Directory for per-task JSON outputs + summary.")
    parser.add_argument("--summary", default=None,
                        help="Path to aggregated summary JSON (default: <output-dir>/eval_all_summary.json).")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help=f"Subset of tasks to run. Default = all. Known: {[t.name for t in TASKS]}")
    parser.add_argument("--skip", nargs="+", default=[], help="Tasks to skip.")
    parser.add_argument("--fast", action="store_true",
                        help="Apply plan's fast-eval sample caps (~1h on 2x4090) instead of full runs.")
    parser.add_argument("--continue-on-error", action="store_true", default=True,
                        help="Keep going after a task fails (default: on).")
    args = parser.parse_args()

    by_name = {t.name: t for t in TASKS}
    if args.tasks:
        unknown = [n for n in args.tasks if n not in by_name]
        if unknown:
            parser.error(f"Unknown tasks: {unknown}. Known: {list(by_name)}")
        selected = [by_name[n] for n in args.tasks]
    else:
        selected = list(TASKS)
    selected = [t for t in selected if t.name not in set(args.skip)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary) if args.summary else output_dir / "eval_all_summary.json"

    results: List[dict] = []
    t_start = time.time()
    for task in selected:
        res = run_task(task, args)
        results.append(res)
        # incremental write so partial progress is durable
        summary_path.write_text(json.dumps({
            "checkpoint": args.checkpoint, "tokenizer": args.tokenizer,
            "device": args.device, "fast": False,
            "tasks": results,
            "total_elapsed_s": time.time() - t_start,
        }, indent=2))

    # Pretty table
    print("\n=== eval_all summary ===")
    print(f"{'task':<18} {'status':<12} {'metric':>10}  {'time_s':>8}")
    for r in results:
        metric = r.get("metric")
        metric_str = f"{metric:.4f}" if isinstance(metric, (int, float)) else "-"
        print(f"{r['task']:<18} {r['status']:<12} {metric_str:>10}  {r.get('elapsed_s', 0):>8.1f}")
    print(f"\nSummary written to {summary_path}")

    num_failed = sum(1 for r in results if r["status"] != "ok")
    return 0 if num_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
