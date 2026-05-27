#!/usr/bin/env python3
"""Sweep eval_all.py across every checkpoint in experiments/runs/ + HF baselines.

Distributes jobs over a fixed GPU pool (default: 0,1,2,3) with two eval jobs per
GPU by default. Each job is a subprocess of `eval_all.py` with
`CUDA_VISIBLE_DEVICES` pinned to a single physical GPU, so two workers may
intentionally share each selected GPU. Per-model per-task JSON lands under
`<output-root>/<model-slug>/`, and a single CSV rollup is written at
`<output-root>/sweep_summary.csv`.

HF baselines (gpt2, HuggingFaceTB/SmolLM2-135M) piggyback on the same driver
because `_eval_common.load_scratch_model` now recognizes HF model ids and
wraps them into the scratch model's `{logits, loss}` interface.

Usage:
    python scripts/evaluation/run_eval_sweep.py \\
        --runs-dir experiments/runs \\
        --output-root experiments/results/sweep \\
        --gpus 0 1 2 3 \\
        --jobs-per-gpu 2 \\
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
from typing import Dict, List, Optional, Set, Tuple

EVAL_DIR = Path(__file__).resolve().parent
EVAL_ALL = EVAL_DIR / "eval_all.py"
PROJECT_ROOT = EVAL_DIR.parents[1]

# Import the task list from eval_all.py so "already evaluated?" uses the
# canonical set of tasks rather than whatever happens to be on disk.
sys.path.insert(0, str(EVAL_DIR))
from eval_all import TASKS as EVAL_TASKS  # noqa: E402

EXPECTED_TASK_NAMES = [t.name for t in EVAL_TASKS]


def summary_is_complete(data: dict, expected_tasks: List[str]) -> bool:
    """Return True iff every expected task ran successfully.

    Some eval scripts complete but do not expose a headline metric under the
    keys eval_all.py probes, yielding `"metric": null`. That should not force a
    full rerun: the task output and raw summary are still present.
    """
    tasks = data.get("tasks") or []
    by_name = {t.get("task"): t for t in tasks if isinstance(t, dict)}
    for name in expected_tasks:
        entry = by_name.get(name)
        if not entry or entry.get("status") != "ok":
            return False
    return True


def load_summary(summary_path: Path) -> Optional[dict]:
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text())
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def is_already_evaluated(summary_path: Path, expected_tasks: List[str]) -> bool:
    """Return True iff summary.json exists and every expected task ran ok."""
    data = load_summary(summary_path)
    return bool(data is not None and summary_is_complete(data, expected_tasks))


def spec_identity_keys(spec: Optional[str]) -> Set[str]:
    """Build comparable ids for a checkpoint path or HF model id."""
    if not spec:
        return set()
    keys = {spec}
    path = Path(spec)
    if path.suffix == ".pt":
        abs_path = path if path.is_absolute() else PROJECT_ROOT / path
        try:
            abs_path = abs_path.resolve()
        except Exception:
            pass
        keys.add(str(abs_path))
        try:
            keys.add(str(abs_path.relative_to(PROJECT_ROOT)))
        except ValueError:
            pass
    return keys


SummaryRecord = Tuple[Path, dict, Set[str]]


def load_summary_records(output_root: Path) -> List[SummaryRecord]:
    """Read existing eval_all summaries once for cache detection."""
    if not output_root.exists():
        return []
    records: List[SummaryRecord] = []
    for summary_path in output_root.rglob("eval_all_summary.json"):
        data = load_summary(summary_path)
        if data is None:
            continue
        records.append((summary_path, data, spec_identity_keys(data.get("checkpoint"))))
    return records


def summary_matches_job(data: dict, summary_keys: Set[str], job: Job,
                        args: argparse.Namespace) -> bool:
    if not (summary_keys & spec_identity_keys(job.spec)):
        return False
    if bool(data.get("fast", False)) != bool(args.fast):
        return False
    if data.get("tokenizer") != job.tokenizer:
        return False
    return True


def find_completed_summary(job: Job, output_root: Path, args: argparse.Namespace,
                           expected_tasks: List[str],
                           records: List[SummaryRecord]) -> Optional[Path]:
    """Find a completed summary for this job, even if the slug changed."""
    expected_path = output_root / job.slug / "eval_all_summary.json"

    # Prefer the canonical path for this invocation.
    for summary_path, data, summary_keys in records:
        if summary_path == expected_path:
            if (
                summary_is_complete(data, expected_tasks)
                and summary_matches_job(data, summary_keys, job, args)
            ):
                return summary_path
            break

    # Fall back to matching by checkpoint/HF id. This avoids reruns when the
    # same output root was produced with a different --runs-dir, which changes
    # the filesystem slug but not the evaluated model.
    for summary_path, data, summary_keys in records:
        if (
            summary_is_complete(data, expected_tasks)
            and summary_matches_job(data, summary_keys, job, args)
        ):
            return summary_path
    return None


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


def worker_label(gpu_id: int, slot_id: int) -> str:
    return f"GPU{gpu_id}.{slot_id}"


def run_job(job: Job, gpu_id: int, slot_id: int, args: argparse.Namespace) -> dict:
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
    if args.fast:
        cmd += ["--fast"]
    if args.tasks:
        cmd += ["--tasks", *args.tasks]
    if args.skip:
        cmd += ["--skip", *args.skip]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Avoid tokenizer thread storms when several workers run in parallel.
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    label = worker_label(gpu_id, slot_id)
    print(f"[sweep] {label} start: {job.slug} ({job.kind})", flush=True)
    t0 = time.time()
    with open(log_path, "wb") as log_fp:
        proc = subprocess.run(cmd, env=env, stdout=log_fp, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    status = "ok" if proc.returncode == 0 else f"failed(rc={proc.returncode})"
    print(f"[sweep] {label} done : {job.slug} status={status} elapsed={elapsed:.1f}s", flush=True)

    summary: dict = {}
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:
            summary = {"_parse_error": str(exc)}

    return {
        "job": job.__dict__,
        "gpu": gpu_id,
        "gpu_slot": slot_id,
        "returncode": proc.returncode,
        "status": status,
        "elapsed_s": elapsed,
        "summary_path": str(summary_path),
        "log_path": str(log_path),
        "summary": summary,
    }


def worker_loop(gpu_id: int, slot_id: int, job_q: "queue.Queue[Job]", results: list,
                results_lock: threading.Lock, args: argparse.Namespace) -> None:
    while True:
        try:
            job = job_q.get_nowait()
        except queue.Empty:
            return
        try:
            res = run_job(job, gpu_id, slot_id, args)
        except Exception as exc:  # defensive — keep worker alive on unexpected errors
            res = {"job": job.__dict__, "gpu": gpu_id, "gpu_slot": slot_id,
                   "status": f"exception:{exc}", "returncode": -1,
                   "elapsed_s": 0, "summary": {}}
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
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3],
                        help="Physical GPU ids to use as the worker pool.")
    parser.add_argument("--jobs-per-gpu", type=int, default=2,
                        help="Concurrent eval_all.py jobs to run on each selected GPU.")
    parser.add_argument("--archived", choices=["skip", "only", "include"], default="include",
                        help="How to treat experiments/runs/*archive_pre-4090* (default: include).")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip gpt2 + SmolLM2-135M baselines.")
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate even if a complete summary.json already exists.")
    parser.add_argument("--fast", action="store_true",
                        help="Pass --fast through to eval_all.py.")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Task subset to pass through to eval_all.py.")
    parser.add_argument("--skip", nargs="+", default=[],
                        help="Tasks to skip when calling eval_all.py.")
    args = parser.parse_args()
    if args.jobs_per_gpu < 1:
        parser.error("--jobs-per-gpu must be >= 1")

    jobs = build_jobs(args)
    if not jobs:
        print("[sweep] No jobs to run.", file=sys.stderr)
        return 1
    expected_tasks = args.tasks if args.tasks else EXPECTED_TASK_NAMES
    expected_tasks = [name for name in expected_tasks if name not in set(args.skip)]

    # Drop jobs whose summary.json already has every expected task completed.
    output_root = Path(args.output_root)
    summary_records = load_summary_records(output_root)
    pending: List[Job] = []
    skipped: List[Tuple[Job, Path]] = []
    for j in jobs:
        completed_summary = None if args.force else find_completed_summary(
            j, output_root, args, expected_tasks, summary_records
        )
        if completed_summary is not None:
            skipped.append((j, completed_summary))
        else:
            pending.append(j)

    worker_count = len(args.gpus) * args.jobs_per_gpu
    print(f"[sweep] {len(pending)} pending / {len(skipped)} already-done jobs "
          f"across {worker_count} worker slots ({args.jobs_per_gpu}/GPU) on GPUs {args.gpus}:")
    for j in pending:
        print(f"  - [{j.kind}] {j.spec}  -> {j.slug}")
    for j, summary_path in skipped:
        expected_path = output_root / j.slug / "eval_all_summary.json"
        suffix = "" if summary_path == expected_path else f"  (cached at {summary_path})"
        print(f"  - [skip-done] {j.spec}  -> {j.slug}{suffix}")
    jobs = pending

    job_q: "queue.Queue[Job]" = queue.Queue()
    for j in jobs:
        job_q.put(j)

    results: List[dict] = []
    lock = threading.Lock()
    threads = [threading.Thread(target=worker_loop,
                                args=(gpu, slot, job_q, results, lock, args),
                                daemon=True)
               for gpu in args.gpus
               for slot in range(args.jobs_per_gpu)]
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
    for j, summary_path in skipped:
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = {}
        results.append({
            "job": j.__dict__, "gpu": -1, "gpu_slot": -1, "returncode": 0,
            "status": "cached", "elapsed_s": 0.0,
            "summary_path": str(summary_path), "summary": summary,
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
