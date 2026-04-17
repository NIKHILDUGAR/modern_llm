#!/usr/bin/env python3
"""ANLI (Adversarial NLI, Nie et al., 2020) evaluation.

ANLI has three rounds of increasing difficulty (R1, R2, R3). Standard practice
is to evaluate each round's test split separately and report per-round accuracy.

Protocol:
    3-way entailment classification via log-likelihood of the verbalized label
    words " True"/" Neither"/" False" (lm-eval-harness template), conditioned on
    a premise+hypothesis prompt.

Usage:
    python scripts/evaluation/eval_anli.py \\
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
# ANLI label mapping: 0=entailment, 1=neutral, 2=contradiction
LABEL_WORDS = [" True", " Neither", " False"]


def build_prompt(premise: str, hypothesis: str) -> str:
    return f"{premise}\nQuestion: {hypothesis} True, False, or Neither?\nAnswer:"


def evaluate_anli_round(model, tokenizer, device: str, split: str, max_samples: int) -> dict:
    ds = load_dataset("facebook/anli", split=split)
    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))
    correct = 0
    total = 0
    for ex in tqdm(ds, desc=f"Evaluating ANLI {split}"):
        prompt = build_prompt(ex["premise"], ex["hypothesis"])
        pred = mc_argmax(model, tokenizer, prompt, LABEL_WORDS, device, length_normalize=False)
        correct += int(pred == ex["label"])
        total += 1
    return {"accuracy": correct / total if total else 0.0, "correct": correct, "total": total}


def main():
    parser = argparse.ArgumentParser(description="ANLI evaluation (R1/R2/R3)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = full; cap applied per round")
    parser.add_argument("--output", type=str, default="experiments/results/anli_metrics.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    model, tokenizer = load_scratch_model(args.checkpoint, args.device, args.tokenizer)

    rounds = {}
    for split in ["test_r1", "test_r2", "test_r3"]:
        print(f"\n--- ANLI {split} ---")
        rounds[split] = evaluate_anli_round(model, tokenizer, args.device, split, args.max_samples)
        print(f"{split} accuracy: {rounds[split]['accuracy']:.2%}")

    total_correct = sum(r["correct"] for r in rounds.values())
    total_count = sum(r["total"] for r in rounds.values())
    results = {
        "accuracy": total_correct / total_count if total_count else 0.0,
        "correct": total_correct,
        "total": total_count,
        "per_round": rounds,
        "model": str(args.checkpoint),
    }

    print(f"\nANLI overall accuracy: {results['accuracy']:.2%}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
