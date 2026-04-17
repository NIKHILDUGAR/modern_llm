#!/usr/bin/env python3
"""GPQA (Rein et al., 2023) evaluation.

GPQA is a graduate-level science QA dataset with 4-way multiple choice. We use
the publicly released "gpqa_main" and "gpqa_diamond" configs. Protocol follows
the original paper: 0-shot letter prediction by log-likelihood over " A".." D".
Choices are shuffled deterministically per-example from the dataset's
(correct_answer, incorrect_answer_{1,2,3}) fields.

Usage:
    python scripts/evaluation/eval_gpqa.py \\
        --checkpoint experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _eval_common import DEFAULT_TOKENIZER, load_scratch_model, mc_argmax

DEFAULT_CKPT = "experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt"
LETTERS = ["A", "B", "C", "D"]


def deterministic_shuffle(example) -> tuple[list[str], int]:
    """Deterministic shuffle keyed on the question — reproducible across runs."""
    choices = [
        example["Correct Answer"],
        example["Incorrect Answer 1"],
        example["Incorrect Answer 2"],
        example["Incorrect Answer 3"],
    ]
    seed = int(hashlib.md5(example["Question"].encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    idx = list(range(4))
    rng.shuffle(idx)
    shuffled = [choices[i] for i in idx]
    gold = idx.index(0)
    return shuffled, gold


def build_prompt(question: str, choices: list[str]) -> str:
    prompt = f"Question: {question.strip()}"
    for letter, choice in zip(LETTERS, choices):
        prompt += f"\n{letter}. {choice}"
    prompt += "\nAnswer:"
    return prompt


def evaluate_gpqa(model, tokenizer, device: str, config_name: str, max_samples: int) -> dict:
    ds = load_dataset("Idavidrein/gpqa", config_name, split="train")
    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    correct = 0
    total = 0
    for ex in tqdm(ds, desc=f"Evaluating GPQA ({config_name})"):
        shuffled_choices, gold_idx = deterministic_shuffle(ex)
        prompt = build_prompt(ex["Question"], shuffled_choices)
        choices = [f" {letter}" for letter in LETTERS]
        pred = mc_argmax(model, tokenizer, prompt, choices, device, length_normalize=False)
        correct += int(pred == gold_idx)
        total += 1

    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
    }


def main():
    parser = argparse.ArgumentParser(description="GPQA evaluation")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--config", type=str, default="gpqa_main",
                        choices=["gpqa_main", "gpqa_diamond", "gpqa_extended"])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", type=str, default="experiments/results/gpqa_metrics.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    model, tokenizer = load_scratch_model(args.checkpoint, args.device, args.tokenizer)

    print(f"Evaluating GPQA ({args.config})...")
    try:
        results = evaluate_gpqa(model, tokenizer, args.device, args.config, args.max_samples)
    except Exception as e:
        # Idavidrein/gpqa is gated — if access is missing, make the failure
        # explicit rather than crashing silently.
        msg = (
            f"Could not load Idavidrein/gpqa ({args.config}). "
            f"GPQA is gated on HuggingFace — accept the license and run "
            f"`huggingface-cli login` first. Underlying error: {e}"
        )
        print(msg, file=sys.stderr)
        results = {"accuracy": None, "correct": 0, "total": 0, "error": str(e)}

    results["model"] = str(args.checkpoint)
    results["config"] = args.config

    if results.get("accuracy") is not None:
        print(f"\nGPQA ({args.config}) accuracy: {results['accuracy']:.2%}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
