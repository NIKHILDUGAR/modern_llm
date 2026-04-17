#!/usr/bin/env python3
"""HellaSwag (Zellers et al., 2019) zero-shot sentence-completion evaluation.

Protocol:
    Concatenate "activity_label: ctx_a ctx_b.capitalize()" as the context and
    score each of the four endings by length-normalized log-likelihood (the
    standard lm-eval setup, since endings can vary widely in length). Argmax
    over length-normalized scores is the prediction.

Usage:
    python scripts/evaluation/eval_hellaswag.py \\
        --checkpoint experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _eval_common import DEFAULT_TOKENIZER, load_scratch_model, mc_argmax

DEFAULT_CKPT = "experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt"


def preprocess(text: str) -> str:
    # Standard HellaSwag cleanup used by lm-eval-harness
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def evaluate_hellaswag(model, tokenizer, device: str, max_samples: int) -> dict:
    ds = load_dataset("Rowan/hellaswag", split="validation")
    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0
    for ex in tqdm(ds, desc="Evaluating HellaSwag"):
        ctx = preprocess(ex["activity_label"] + ": " + ex["ctx_a"] + " " + ex["ctx_b"].capitalize())
        endings = [" " + preprocess(e) for e in ex["endings"]]
        pred = mc_argmax(model, tokenizer, ctx, endings, device, length_normalize=True)
        gold = int(ex["label"])
        correct += int(pred == gold)
        total += 1

    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="HellaSwag evaluation")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = full validation set")
    parser.add_argument("--output", type=str, default="experiments/results/hellaswag_metrics.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    model, tokenizer = load_scratch_model(args.checkpoint, args.device, args.tokenizer)

    print("Evaluating HellaSwag (zero-shot, length-normalized)...")
    results = evaluate_hellaswag(model, tokenizer, args.device, args.max_samples)
    results["model"] = str(args.checkpoint)

    print(f"\nHellaSwag Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
