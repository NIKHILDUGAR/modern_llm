#!/usr/bin/env python3
"""CommonsenseQA (Talmor et al., 2019) evaluation.

Protocol:
    5-way multiple choice (A-E). We score " A" ... " E" completions by
    next-token log-prob after a question+choices prompt. Evaluated on the
    validation split (test labels are private).

Usage:
    python scripts/evaluation/eval_commonsenseqa.py \\
        --checkpoint experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _eval_common import DEFAULT_TOKENIZER, load_scratch_model, mc_argmax

DEFAULT_CKPT = "experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt"


def build_prompt(ex: dict) -> str:
    prompt = f"Question: {ex['question']}"
    for letter, text in zip(ex["choices"]["label"], ex["choices"]["text"]):
        prompt += f"\n{letter}. {text}"
    prompt += "\nAnswer:"
    return prompt


def evaluate_csqa(model, tokenizer, device: str, max_samples: int) -> dict:
    ds = load_dataset("tau/commonsense_qa", split="validation")
    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0
    for ex in tqdm(ds, desc="Evaluating CommonsenseQA"):
        labels = ex["choices"]["label"]
        prompt = build_prompt(ex)
        choices = [f" {lbl}" for lbl in labels]
        pred_idx = mc_argmax(model, tokenizer, prompt, choices, device, length_normalize=False)
        gold_idx = labels.index(ex["answerKey"])
        correct += int(pred_idx == gold_idx)
        total += 1

    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="CommonsenseQA evaluation")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", type=str, default="experiments/results/commonsenseqa_metrics.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    model, tokenizer = load_scratch_model(args.checkpoint, args.device, args.tokenizer)

    print("Evaluating CommonsenseQA...")
    results = evaluate_csqa(model, tokenizer, args.device, args.max_samples)
    results["model"] = str(args.checkpoint)

    print(f"\nCommonsenseQA accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
