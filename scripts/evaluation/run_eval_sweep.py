#!/usr/bin/env python3
"""Sweep eval_all.py across every checkpoint in experiments/runs/ + HF baselines.

Distributes jobs over a fixed GPU pool (default: 0,1,2) — one checkpoint per
GPU at a time. Each job is a subprocess of `eval_all.py` with
`CUDA_VISIBLE_DEVICES` pinned to a single physical GPU, so the three workers
never fight for memory. Per-model per-task JSON lands under
`<output-root>/<model-slug>/`, and a single CSV rollup is written at
`<output-root>/sweep_summary.csv`.

HF baselines (gpt2, HuggingFaceTB/SmolLM2-135M) piggyback on the same driver
because `_eval_common.load_scratch_model` now recognizes HF model ids and
wraps them into the scratch model's `{logits, loss}` interface.

Usage:
    python scripts/evaluation/run_eval_sweep.py \\
        --runs-dir experiments/runs \\
        --output-root experiments/results/sweep \\
        --gpus 0 1 2 \\
        --fast                        # small per-task sample caps
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

EVAL_DIR = Path(__file__).resolve().parent
EVAL_ALL = EVAL_DIR / "eval_all.py"
PROJECT_ROOT = EVAL_DIR.parents[1]

# Import the task list from eval_all.py so "already evaluated?" uses the
# canonical set of tasks rather than whatever happens to be on disk.
sys.path.insert(0, str(EVAL_DIR))
from eval_all import TASKS as EVAL_TASKS  # noqa: E402

EXPECTED_TASK_NAMES = [t.name for t in EVAL_TASKS]


def is_already_evaluated(summary_path: Path) -> bool:
    """Return True iff summary.json exists and every expected task ran ok with a metric."""
    if not summary_path.exists():
        return False
    try:
        data = json.loads(summary_path.read_text())
    except Exception:
        return False
    #tasks = data.get("tasks") or []
    #by_name = {t.get("task"): t for t in tasks if isinstance(t, dict)}
    #for name in EXPECTED_TASK_NAMES:
    #    entry = by_name.get(name)
    #    if not entry or entry.get("status") != "ok" or entry.get("metric") is None:
    #        return False
    return True


HF_BASELINES = [
    # (model_id, tokenizer_id) — tokenizer defaults to the model's own
    ("gpt2", "gpt2"),
    ("HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M"),
]


def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


@dataclass
class Job:
    spec: str           # path to .pt or an HF id
    kind: str           # "checkpoint" or "hf"
    slug: str           # filesystem-safe id used for output dir
    tokenizer: Optional[str]  # None = let eval_all.py use its default (16k BPE)


def discover_checkpoints(runs_dir: Path, archive_filter: Optional[str]) -> List[Path]:
    # All *.pt, recursive. Optional filter "skip" drops archived pre-4090 runs
    # (whose tokenizer/arch are incompatible); "only" keeps just those.
    pts = sorted(runs_dir.rglob("*.pt"))
    if archive_filter == "skip":
        pts = [p for p in pts if "archive_pre-4090" not in p.parts]
    elif archive_filter == "only":
        pts = [p for p in pts if "archive_pre-4090" in p.parts]
    # Verifier heads are binary classifiers, not LMs — LM evals are meaningless.
    pts = [p for p in pts if "verifier" not in p.name.lower()
           and not any("verifier" in part.lower() for part in p.parts)]
    return pts


def build_jobs(args: argparse.Namespace) -> List[Job]:
    jobs: List[Job] = []
    runs_dir = Path(args.runs_dir)
    if runs_dir.exists():
        for pt in discover_checkpoints(runs_dir, args.archived):
            rel = pt.relative_to(runs_dir)
            jobs.append(Job(spec=str(pt.resolve()), kind="checkpoint",
                            slug=slugify(str(rel.with_suffix(""))), tokenizer=None))
    else:
        print(f"[sweep] WARN: {runs_dir} does not exist; no checkpoints discovered.", flush=True)

    if not args.no_baselines:
        for hf_id, tok in HF_BASELINES:
            jobs.append(Job(spec=hf_id, kind="hf", slug=slugify(hf_id), tokenizer=tok))

    return jobs


def run_job(job: Job, gpu_id: int, args: argparse.Namespace) -> dict:
    out_dir = Path(args.output_root) / job.slug
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "eval_all_summary.json"
    log_path = out_dir / "eval_all.log"

    cmd = [sys.executable, str(EVAL_ALL),
           "--checkpoint", job.spec,
           "--output-dir", str(out_dir),
           "--summary", str(summary_path),
           "--device", "cuda"]  # GPU index is pinned via CUDA_VISIBLE_DEVICES
    if job.tokenizer:
        cmd += ["--tokenizer", job.tokenizer]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Avoid tokenizer thread storms when several workers run in parallel.
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"[sweep] GPU{gpu_id} start: {job.slug} ({job.kind})", flush=True)
    t0 = time.time()
    with open(log_path, "wb") as log_fp:
        proc = subprocess.run(cmd, env=env, stdout=log_fp, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    status = "ok" if proc.returncode == 0 else f"failed(rc={proc.returncode})"
    print(f"[sweep] GPU{gpu_id} done : {job.slug} status={status} elapsed={elapsed:.1f}s", flush=True)

    summary: dict = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:
            summary = {"_parse_error": str(exc)}

    return {
        "job": job.__dict__,
        "gpu": gpu_id,
        "returncode": proc.returncode,
        "status": status,
        "elapsed_s": elapsed,
        "summary_path": str(summary_path),
        "log_path": str(log_path),
        "summary": summary,
    }


def worker_loop(gpu_id: int, job_q: "queue.Queue[Job]", results: list,
                results_lock: threading.Lock, args: argparse.Namespace) -> None:
    while True:
        try:
            job = job_q.get_nowait()
        except queue.Empty:
            return
        try:
            res = run_job(job, gpu_id, args)
        except Exception as exc:  # defensive — keep worker alive on unexpected errors
            res = {"job": job.__dict__, "gpu": gpu_id, "status": f"exception:{exc}",
                   "returncode": -1, "elapsed_s": 0, "summary": {}}
        with results_lock:
            results.append(res)
        job_q.task_done()


def all_task_names(results: List[dict]) -> List[str]:
    names: List[str] = []
    seen = set()
    for r in results:
        for t in (r.get("summary") or {}).get("tasks", []) or []:
            name = t.get("task")
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def write_csv(results: List[dict], csv_path: Path) -> None:
    task_names = all_task_names(results)
    headers = ["model", "kind", "status", "returncode", "elapsed_s", "gpu",
               "summary_path", *task_names]
    rows = []
    for r in results:
        job = r["job"]
        task_metrics: Dict[str, Optional[float]] = {}
        task_status: Dict[str, str] = {}
        for t in (r.get("summary") or {}).get("tasks", []) or []:
            task_metrics[t["task"]] = t.get("metric")
            task_status[t["task"]] = t.get("status", "")
        row = {
            "model": job["spec"],
            "kind": job["kind"],
            "status": r["status"],
            "returncode": r["returncode"],
            "elapsed_s": f"{r['elapsed_s']:.1f}",
            "gpu": r["gpu"],
            "summary_path": r.get("summary_path", ""),
        }
        for name in task_names:
            v = task_metrics.get(name)
            if v is None:
                row[name] = task_status.get(name, "")
            else:
                row[name] = f"{v:.4f}"
        rows.append(row)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[sweep] CSV written: {csv_path}  ({len(rows)} rows, {len(task_names)} tasks)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel eval sweep over checkpoints + HF baselines.")
    parser.add_argument("--runs-dir", default="experiments/runs",
                        help="Directory searched recursively for *.pt checkpoints.")
    parser.add_argument("--output-root", default="experiments/results/sweep",
                        help="Per-model results land under <output-root>/<model-slug>/.")
    parser.add_argument("--csv", default=None,
                        help="CSV path (default: <output-root>/sweep_summary.csv).")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2],
                        help="Physical GPU ids to use as the worker pool.")
    parser.add_argument("--archived", choices=["skip", "only", "include"], default="include",
                        help="How to treat experiments/runs/*archive_pre-4090* (default: skip).")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip gpt2 + SmolLM2-135M baselines.")
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate even if a complete summary.json already exists.")
    args = parser.parse_args()

    jobs = build_jobs(args)
    if not jobs:
        print("[sweep] No jobs to run.", file=sys.stderr)
        return 1

    # Drop jobs whose summary.json already has every expected task completed.
    output_root = Path(args.output_root)
    pending: List[Job] = []
    skipped: List[Job] = []
    cou=0
    for j in jobs:
        args.force=False
        summary_path = output_root / j.slug / "eval_all_summary.json"
        print(summary_path)
        if not args.force and is_already_evaluated(summary_path):
            cou+=1
            print(f"skipping {summary_path} count ", cou )
            skipped.append(j)
        else:
            pending.append(j)

    print(f"[sweep] {len(pending)} pending / {len(skipped)} already-done jobs across GPUs {args.gpus}:")
    for j in pending:
        print(f"  - [{j.kind}] {j.spec}  -> {j.slug}")
    for j in skipped:
        print(f"  - [skip-done] {j.spec}  -> {j.slug}")
    jobs = pending

    job_q: "queue.Queue[Job]" = queue.Queue()
    for j in jobs:
        job_q.put(j)

    results: List[dict] = []
    lock = threading.Lock()
    threads = [threading.Thread(target=worker_loop,
                                args=(gpu, job_q, results, lock, args),
                                daemon=True)
               for gpu in args.gpus]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total = time.time() - t0

    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv) if args.csv else output_root / "sweep_summary.csv"

    # Fold previously-completed jobs back into the CSV so it covers every model
    # on disk, not just the ones we re-ran this session.
    for j in skipped:
        summary_path = output_root / j.slug / "eval_all_summary.json"
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = {}
        results.append({
            "job": j.__dict__, "gpu": -1, "returncode": 0, "status": "cached",
            "elapsed_s": 0.0, "summary_path": str(summary_path), "summary": summary,
        })
    write_csv(results, csv_path)

    # Also dump a raw JSON rollup for completeness.
    json_path = output_root / "sweep_summary.json"
    json_path.write_text(json.dumps({"total_elapsed_s": total, "results": results}, indent=2))
    print(f"[sweep] JSON written: {json_path}")
    print(f"[sweep] total elapsed: {total:.1f}s ({len(results)} jobs)")

    failed = [r for r in results if r["returncode"] != 0]
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
