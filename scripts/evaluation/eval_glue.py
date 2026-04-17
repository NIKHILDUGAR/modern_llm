#!/usr/bin/env python3
"""GLUE (Wang et al., 2019) multi-task evaluation.

Covers the 8 standard GLUE subtasks:
    cola   (acceptability,    accuracy + Matthews corrcoef)
    sst2   (sentiment,        accuracy)
    mrpc   (paraphrase,       accuracy + F1)
    qqp    (paraphrase,       accuracy + F1)
    mnli   (entailment,       matched + mismatched accuracy)
    qnli   (QA-as-NLI,        accuracy)
    rte    (entailment,       accuracy)
    stsb   (similarity,       Pearson/Spearman over bucketed prediction)

All classification tasks use log-likelihood over verbalized labels (lm-eval
template). STS-B is continuous [0, 5]; we score over 6 integer buckets and
report the soft expectation. Each subtask reports its standard metric, and we
emit an unweighted average (GLUE score) across tasks.

Usage:
    python scripts/evaluation/eval_glue.py \\
        --checkpoint experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _eval_common import DEFAULT_TOKENIZER, load_scratch_model, mc_argmax, score_completion

DEFAULT_CKPT = "experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt"


# ---------- Metric helpers (self-contained, no sklearn dep) ----------

def accuracy(preds: List[int], labels: List[int]) -> float:
    if not preds:
        return 0.0
    return sum(int(p == l) for p, l in zip(preds, labels)) / len(preds)


def f1_binary(preds: List[int], labels: List[int]) -> float:
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r)


def matthews_corrcoef(preds: List[int], labels: List[int]) -> float:
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


def pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denom = math.sqrt(sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y))
    return num / denom if denom > 0 else 0.0


def spearman(x: List[float], y: List[float]) -> float:
    def rank(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        ranks = [0.0] * len(v)
        for r, i in enumerate(idx):
            ranks[i] = r
        return ranks
    return pearson(rank(x), rank(y))


# ---------- Subtask evaluators ----------

def _classify(model, tokenizer, device, ds, prompt_fn: Callable, label_words: List[str]) -> Tuple[List[int], List[int]]:
    preds, labels = [], []
    for ex in tqdm(ds, desc=prompt_fn.__name__):
        prompt = prompt_fn(ex)
        pred = mc_argmax(model, tokenizer, prompt, label_words, device, length_normalize=False)
        preds.append(pred)
        labels.append(int(ex["label"]))
    return preds, labels


def eval_cola(model, tokenizer, device, max_samples):
    ds = load_dataset("glue", "cola", split="validation")
    if max_samples: ds = ds.select(range(min(max_samples, len(ds))))
    def fmt_cola(ex): return f"{ex['sentence']}\nQuestion: Is this sentence grammatically acceptable?\nAnswer:"
    preds, labels = _classify(model, tokenizer, device, ds, fmt_cola, [" no", " yes"])
    return {"accuracy": accuracy(preds, labels), "matthews_corrcoef": matthews_corrcoef(preds, labels)}


def eval_sst2(model, tokenizer, device, max_samples):
    ds = load_dataset("glue", "sst2", split="validation")
    if max_samples: ds = ds.select(range(min(max_samples, len(ds))))
    def fmt_sst2(ex): return f"Review: {ex['sentence'].strip()}\nSentiment:"
    preds, labels = _classify(model, tokenizer, device, ds, fmt_sst2, [" negative", " positive"])
    return {"accuracy": accuracy(preds, labels)}


def eval_mrpc(model, tokenizer, device, max_samples):
    ds = load_dataset("glue", "mrpc", split="validation")
    if max_samples: ds = ds.select(range(min(max_samples, len(ds))))
    def fmt_mrpc(ex): return f"Sentence 1: {ex['sentence1']}\nSentence 2: {ex['sentence2']}\nDo they mean the same thing?\nAnswer:"
    preds, labels = _classify(model, tokenizer, device, ds, fmt_mrpc, [" no", " yes"])
    return {"accuracy": accuracy(preds, labels), "f1": f1_binary(preds, labels)}


def eval_qqp(model, tokenizer, device, max_samples):
    ds = load_dataset("glue", "qqp", split="validation")
    if max_samples: ds = ds.select(range(min(max_samples, len(ds))))
    def fmt_qqp(ex): return f"Question 1: {ex['question1']}\nQuestion 2: {ex['question2']}\nAre these duplicate questions?\nAnswer:"
    preds, labels = _classify(model, tokenizer, device, ds, fmt_qqp, [" no", " yes"])
    return {"accuracy": accuracy(preds, labels), "f1": f1_binary(preds, labels)}


def eval_mnli(model, tokenizer, device, max_samples):
    out = {}
    for split_key, split_name in [("matched", "validation_matched"), ("mismatched", "validation_mismatched")]:
        ds = load_dataset("glue", "mnli", split=split_name)
        if max_samples: ds = ds.select(range(min(max_samples, len(ds))))
        def fmt_mnli(ex): return f"{ex['premise']}\nQuestion: {ex['hypothesis']} True, False, or Neither?\nAnswer:"
        # glue label order: 0=entailment, 1=neutral, 2=contradiction
        preds, labels = _classify(model, tokenizer, device, ds, fmt_mnli, [" True", " Neither", " False"])
        out[f"accuracy_{split_key}"] = accuracy(preds, labels)
    return out


def eval_qnli(model, tokenizer, device, max_samples):
    ds = load_dataset("glue", "qnli", split="validation")
    if max_samples: ds = ds.select(range(min(max_samples, len(ds))))
    def fmt_qnli(ex): return f"{ex['sentence']}\nQuestion: {ex['question']}\nDoes the sentence answer the question?\nAnswer:"
    # qnli: 0=entailment(yes), 1=not_entailment(no)
    preds, labels = _classify(model, tokenizer, device, ds, fmt_qnli, [" yes", " no"])
    return {"accuracy": accuracy(preds, labels)}


def eval_rte(model, tokenizer, device, max_samples):
    ds = load_dataset("glue", "rte", split="validation")
    if max_samples: ds = ds.select(range(min(max_samples, len(ds))))
    def fmt_rte(ex): return f"{ex['sentence1']}\nQuestion: {ex['sentence2']} True or False?\nAnswer:"
    # rte: 0=entailment(True), 1=not_entailment(False)
    preds, labels = _classify(model, tokenizer, device, ds, fmt_rte, [" True", " False"])
    return {"accuracy": accuracy(preds, labels)}


def eval_stsb(model, tokenizer, device, max_samples):
    ds = load_dataset("glue", "stsb", split="validation")
    if max_samples: ds = ds.select(range(min(max_samples, len(ds))))
    # soft expectation over 0..5 buckets
    buckets = [f" {i}" for i in range(6)]
    preds_cont, labels_cont = [], []
    for ex in tqdm(ds, desc="eval_stsb"):
        prompt = f"Sentence 1: {ex['sentence1']}\nSentence 2: {ex['sentence2']}\nSimilarity (0-5):"
        log_probs = []
        for b in buckets:
            log_probs.append(score_completion(model, tokenizer, prompt, b, device))
        m = max(log_probs)
        exps = [math.exp(lp - m) for lp in log_probs]
        Z = sum(exps)
        expected = sum(i * (e / Z) for i, e in enumerate(exps))
        preds_cont.append(expected)
        labels_cont.append(float(ex["label"]))
    return {"pearson": pearson(preds_cont, labels_cont),
            "spearman": spearman(preds_cont, labels_cont)}


SUBTASKS = {
    "cola": eval_cola,
    "sst2": eval_sst2,
    "mrpc": eval_mrpc,
    "qqp": eval_qqp,
    "mnli": eval_mnli,
    "qnli": eval_qnli,
    "rte": eval_rte,
    "stsb": eval_stsb,
}


def summarize(per_task: dict) -> float:
    """Unweighted mean of each task's 'primary' metric (matches GLUE convention)."""
    primary = []
    if "cola" in per_task: primary.append(per_task["cola"]["matthews_corrcoef"])
    if "sst2" in per_task: primary.append(per_task["sst2"]["accuracy"])
    if "mrpc" in per_task: primary.append((per_task["mrpc"]["accuracy"] + per_task["mrpc"]["f1"]) / 2)
    if "qqp" in per_task: primary.append((per_task["qqp"]["accuracy"] + per_task["qqp"]["f1"]) / 2)
    if "mnli" in per_task: primary.append((per_task["mnli"]["accuracy_matched"] + per_task["mnli"]["accuracy_mismatched"]) / 2)
    if "qnli" in per_task: primary.append(per_task["qnli"]["accuracy"])
    if "rte" in per_task: primary.append(per_task["rte"]["accuracy"])
    if "stsb" in per_task: primary.append((per_task["stsb"]["pearson"] + per_task["stsb"]["spearman"]) / 2)
    return sum(primary) / len(primary) if primary else 0.0


def main():
    parser = argparse.ArgumentParser(description="GLUE evaluation (all 8 subtasks)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--tasks", nargs="+", default=list(SUBTASKS.keys()),
                        choices=list(SUBTASKS.keys()))
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Cap per-task sample count (QQP/MNLI are huge)")
    parser.add_argument("--output", type=str, default="experiments/results/glue_metrics.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    model, tokenizer = load_scratch_model(args.checkpoint, args.device, args.tokenizer)

    per_task = {}
    for task in args.tasks:
        print(f"\n--- GLUE {task} ---")
        per_task[task] = SUBTASKS[task](model, tokenizer, args.device, args.max_samples)
        print(f"{task}: {per_task[task]}")

    avg = summarize(per_task)
    results = {
        "glue_score": avg,
        "per_task": per_task,
        "model": str(args.checkpoint),
    }
    print(f"\nGLUE average: {avg:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
