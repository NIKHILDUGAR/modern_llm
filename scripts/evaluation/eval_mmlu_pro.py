#!/usr/bin/env python3
"""MMLU-Pro (Wang et al., 2024) 5-shot multiple-choice evaluation.

Differences from MMLU:
    - 10 answer choices (A-J) rather than 4.
    - Harder, professionally curated questions with per-category dev/shot pool.

Protocol is identical to MMLU's: 5-shot CoT-free letter prediction via
next-token log-prob over " A"..." J". Accuracy reported overall and per category.

Usage:
    python scripts/evaluation/eval_mmlu_pro.py \\
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
LETTERS = [chr(ord("A") + i) for i in range(10)]


def format_example(ex: dict, include_answer: bool) -> str:
    prompt = ex["question"].strip()
    for letter, choice in zip(LETTERS, ex["options"]):
        prompt += f"\n{letter}. {choice}"
    prompt += "\nAnswer:"
    if include_answer:
        # MMLU-Pro provides "answer" as the letter string and "answer_index" as int.
        idx = ex.get("answer_index", LETTERS.index(ex["answer"]))
        prompt += f" {LETTERS[idx]}\n\n"
    return prompt


def build_prompt(dev_examples, question_ex, category: str, n_shot: int) -> str:
    header = f"The following are multiple choice questions (with answers) about {category}.\n\n"
    shots = "".join(format_example(dev_examples[i], include_answer=True)
                    for i in range(min(n_shot, len(dev_examples))))
    return header + shots + format_example(question_ex, include_answer=False)


def evaluate_mmlu_pro(model, tokenizer, device: str, n_shot: int, max_samples: int) -> dict:
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    test = ds["test"]
    # validation split serves as the few-shot pool (CoT exemplars stripped)
    shot_pool = ds["validation"] if "validation" in ds else ds["test"]
    pool_by_cat = defaultdict(list)
    for ex in shot_pool:
        pool_by_cat[ex["category"]].append(ex)

    if max_samples and max_samples > 0:
        test = test.select(range(min(max_samples, len(test))))

    by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total_count = 0

    for ex in tqdm(test, desc="Evaluating MMLU-Pro"):
        cat = ex["category"]
        num_opts = len(ex["options"])
        prompt = build_prompt(pool_by_cat.get(cat, []), ex, cat, n_shot)
        choices = [f" {letter}" for letter in LETTERS[:num_opts]]
        pred_idx = mc_argmax(model, tokenizer, prompt, choices, device, length_normalize=False)
        gold_idx = ex.get("answer_index", LETTERS.index(ex["answer"]))
        is_correct = int(pred_idx == gold_idx)
        by_cat[cat]["correct"] += is_correct
        by_cat[cat]["total"] += 1
        total_correct += is_correct
        total_count += 1

    per_cat = {
        c: {"accuracy": v["correct"] / v["total"] if v["total"] else 0.0, **v}
        for c, v in by_cat.items()
    }
    return {
        "accuracy": total_correct / total_count if total_count else 0.0,
        "correct": total_correct,
        "total": total_count,
        "per_category": per_cat,
        "n_shot": n_shot,
    }


def main():
    parser = argparse.ArgumentParser(description="MMLU-Pro 5-shot evaluation")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--n-shot", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = full test set")
    parser.add_argument("--output", type=str, default="experiments/results/mmlu_pro_metrics.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    model, tokenizer = load_scratch_model(args.checkpoint, args.device, args.tokenizer)

    print(f"Evaluating MMLU-Pro ({args.n_shot}-shot)...")
    results = evaluate_mmlu_pro(model, tokenizer, args.device, args.n_shot, args.max_samples)
    results["model"] = str(args.checkpoint)

    print(f"\nMMLU-Pro Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
