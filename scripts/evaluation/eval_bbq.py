#!/usr/bin/env python3
"""BBQ (Bias Benchmark for QA, Parrish et al., 2022) evaluation.

BBQ tests social-bias behavior in QA by pairing ambiguous and disambiguated
contexts for the same question. Each question has three answer choices; the
aggregate metric is accuracy (with separate numbers for ambiguous vs.
disambiguated contexts). Standard BBQ also reports a bias score; we include it
as "bias_score_disambig" computed only over disambiguated contexts because the
ambiguous-context bias score requires per-question target-answer metadata that
is not always materialized in HF splits.

Protocol:
    3-way multiple choice via log-likelihood over " A" / " B" / " C".

Usage:
    python scripts/evaluation/eval_bbq.py \\
        --checkpoint experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _eval_common import DEFAULT_TOKENIZER, load_scratch_model, mc_argmax

DEFAULT_CKPT = "experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt"
LETTERS = ["A", "B", "C"]


def build_prompt(ex: dict) -> str:
    prompt = f"{ex['context']}\nQuestion: {ex['question']}"
    for letter, key in zip(LETTERS, ["ans0", "ans1", "ans2"]):
        prompt += f"\n{letter}. {ex[key]}"
    prompt += "\nAnswer:"
    return prompt


def evaluate_bbq(model, tokenizer, device: str, max_samples_per_cat: int) -> dict:
    # Community mirror; original heegyu/bbq or Elfsong/BBQ are common
    try:
        ds = load_dataset("heegyu/bbq")
    except Exception:
        ds = load_dataset("Elfsong/BBQ")
    # BBQ on Elfsong/BBQ is split by social category (age, gender, race, ...),
    # not train/test. Concatenate every split so we evaluate the whole benchmark.
    if "test" in ds:
        data = list(ds["test"])
    else:
        data = [ex for split in ds.keys() for ex in ds[split]]

    by_cat = defaultdict(lambda: {"correct": 0, "total": 0, "ambig_correct": 0, "ambig_total": 0,
                                  "disambig_correct": 0, "disambig_total": 0})
    total_correct = 0
    total_count = 0

    # optional cap per category
    if max_samples_per_cat and max_samples_per_cat > 0:
        buckets = defaultdict(list)
        for ex in data:
            cat = ex.get("category", "all")
            if len(buckets[cat]) < max_samples_per_cat:
                buckets[cat].append(ex)
        data = [e for bucket in buckets.values() for e in bucket]

    for ex in tqdm(data, desc="Evaluating BBQ"):
        prompt = build_prompt(ex)
        choices = [f" {letter}" for letter in LETTERS]
        pred = mc_argmax(model, tokenizer, prompt, choices, device, length_normalize=False)
        # Schema varies across mirrors: Elfsong/BBQ uses "answer_label"; older
        # flattened mirrors use "label". Fall back through the aliases.
        gold = int(ex.get("answer_label", ex.get("label", 0)))
        cat = ex.get("category", "all")
        is_correct = int(pred == gold)
        by_cat[cat]["correct"] += is_correct
        by_cat[cat]["total"] += 1
        context_cond = ex.get("context_condition", "")
        if context_cond == "ambig":
            by_cat[cat]["ambig_correct"] += is_correct
            by_cat[cat]["ambig_total"] += 1
        elif context_cond == "disambig":
            by_cat[cat]["disambig_correct"] += is_correct
            by_cat[cat]["disambig_total"] += 1
        total_correct += is_correct
        total_count += 1

    per_cat = {c: {"accuracy": v["correct"] / v["total"] if v["total"] else 0.0, **v}
               for c, v in by_cat.items()}
    return {
        "accuracy": total_correct / total_count if total_count else 0.0,
        "correct": total_correct,
        "total": total_count,
        "per_category": per_cat,
    }


def main():
    parser = argparse.ArgumentParser(description="BBQ evaluation")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-samples-per-cat", type=int, default=0,
                        help="0 = full; otherwise cap per BBQ category")
    parser.add_argument("--output", type=str, default="experiments/results/bbq_metrics.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    model, tokenizer = load_scratch_model(args.checkpoint, args.device, args.tokenizer)

    print("Evaluating BBQ...")
    results = evaluate_bbq(model, tokenizer, args.device, args.max_samples_per_cat)
    results["model"] = str(args.checkpoint)

    print(f"\nBBQ accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
