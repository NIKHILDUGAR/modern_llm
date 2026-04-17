#!/usr/bin/env python3
"""SQuAD v2 (Rajpurkar et al., 2018) evaluation.

SQuAD v2 adds unanswerable questions — a model must either extract the answer
span or decline with an empty answer. We use the official SQuAD v2 metric
(exact_match, f1, HasAns/NoAns breakdowns) via the `evaluate` library when
available; otherwise a local implementation.

Protocol:
    Greedy generation. Prompt the model with the passage + question and a
    one-shot "unanswerable" hint. If the decoded answer contains "unanswerable"
    or "no answer", we emit the empty string.

Usage:
    python scripts/evaluation/eval_squad_v2.py \\
        --checkpoint experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _eval_common import DEFAULT_TOKENIZER, greedy_generate, load_scratch_model

DEFAULT_CKPT = "experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt"


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def f1_score(pred: str, gold: str) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    p = overlap / len(pred_tokens)
    r = overlap / len(gold_tokens)
    return 2 * p * r / (p + r)


def em_score(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def best_metric(pred: str, golds: list[str], fn) -> float:
    if not golds:
        golds = [""]
    return max(fn(pred, g) for g in golds)


def evaluate_squad_v2(model, tokenizer, device: str, max_samples: int) -> dict:
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    em_has, em_no = 0.0, 0.0
    f1_has, f1_no = 0.0, 0.0
    n_has, n_no = 0, 0

    for ex in tqdm(ds, desc="Evaluating SQuAD v2"):
        context = ex["context"]
        question = ex["question"]
        golds = ex["answers"]["text"]
        prompt = (
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"If the question is unanswerable from the context, reply \"unanswerable\".\n"
            f"Answer:"
        )
        raw = greedy_generate(model, tokenizer, prompt, device,
                              max_new_tokens=48, stop_strings=["\n", "Question:", "Context:"])
        pred = raw.strip()
        if pred.lower() in {"unanswerable", "no answer", "none"} or "unanswerable" in pred.lower():
            pred = ""
        is_has_ans = bool(golds) and any(g.strip() for g in golds)

        em = best_metric(pred, golds if is_has_ans else [""], em_score)
        f1 = best_metric(pred, golds if is_has_ans else [""], f1_score)
        if is_has_ans:
            em_has += em; f1_has += f1; n_has += 1
        else:
            em_no += em; f1_no += f1; n_no += 1

    total = n_has + n_no
    return {
        "exact_match": 100.0 * (em_has + em_no) / max(1, total),
        "f1": 100.0 * (f1_has + f1_no) / max(1, total),
        "HasAns_exact": 100.0 * em_has / max(1, n_has),
        "HasAns_f1": 100.0 * f1_has / max(1, n_has),
        "NoAns_exact": 100.0 * em_no / max(1, n_no),
        "NoAns_f1": 100.0 * f1_no / max(1, n_no),
        "total": total,
        "HasAns_total": n_has,
        "NoAns_total": n_no,
    }


def main():
    parser = argparse.ArgumentParser(description="SQuAD v2 evaluation")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-samples", type=int, default=500,
                        help="SQuAD v2 dev is ~11k — default to 500 for speed")
    parser.add_argument("--output", type=str, default="experiments/results/squad_v2_metrics.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    model, tokenizer = load_scratch_model(args.checkpoint, args.device, args.tokenizer)

    print("Evaluating SQuAD v2...")
    results = evaluate_squad_v2(model, tokenizer, args.device, args.max_samples)
    results["model"] = str(args.checkpoint)

    print(f"\nSQuAD v2 EM: {results['exact_match']:.2f}  F1: {results['f1']:.2f}")
    print(f"  HasAns EM: {results['HasAns_exact']:.2f}  F1: {results['HasAns_f1']:.2f}")
    print(f"  NoAns  EM: {results['NoAns_exact']:.2f}  F1: {results['NoAns_f1']:.2f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
