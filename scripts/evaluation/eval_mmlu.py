#!/usr/bin/env python3
"""MMLU (Hendrycks et al., 2021) 5-shot multiple-choice evaluation.

Protocol:
    For each question, construct a 5-shot prompt from the dev split of the same
    subject, append the question with the letter-choice template, and score the
    four completions " A" / " B" / " C" / " D" by next-token log-prob. The model
    picks the argmax. Accuracy is reported overall and per-subject.

Usage:
    python scripts/evaluation/eval_mmlu.py \\
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
LETTERS = ["A", "B", "C", "D"]


def format_example(ex: dict, include_answer: bool) -> str:
    prompt = ex["question"].strip()
    for letter, choice in zip(LETTERS, ex["choices"]):
        prompt += f"\n{letter}. {choice}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {LETTERS[ex['answer']]}\n\n"
    return prompt


def build_prompt(dev_examples, question_ex, subject: str, n_shot: int) -> str:
    header = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    shots = "".join(format_example(dev_examples[i], include_answer=True) for i in range(min(n_shot, len(dev_examples))))
    return header + shots + format_example(question_ex, include_answer=False)


def evaluate_mmlu(model, tokenizer, device: str, n_shot: int, max_samples_per_subject: int) -> dict:
    # "all" config gives every subject in one stream
    test = load_dataset("cais/mmlu", "all", split="test")
    dev = load_dataset("cais/mmlu", "all", split="dev")
    dev_by_subject = defaultdict(list)
    for ex in dev:
        dev_by_subject[ex["subject"]].append(ex)

    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total_count = 0

    # optional subsample per subject
    if max_samples_per_subject > 0:
        buckets = defaultdict(list)
        for ex in test:
            if len(buckets[ex["subject"]]) < max_samples_per_subject:
                buckets[ex["subject"]].append(ex)
        examples = [e for bucket in buckets.values() for e in bucket]
    else:
        examples = list(test)

    for ex in tqdm(examples, desc="Evaluating MMLU"):
        subject = ex["subject"]
        prompt = build_prompt(dev_by_subject[subject], ex, subject, n_shot)
        choices = [f" {letter}" for letter in LETTERS]
        pred = mc_argmax(model, tokenizer, prompt, choices, device, length_normalize=False)
        is_correct = int(pred == ex["answer"])
        by_subject[subject]["correct"] += is_correct
        by_subject[subject]["total"] += 1
        total_correct += is_correct
        total_count += 1

    per_subject = {
        s: {"accuracy": v["correct"] / v["total"] if v["total"] else 0.0, **v}
        for s, v in by_subject.items()
    }
    return {
        "accuracy": total_correct / total_count if total_count else 0.0,
        "correct": total_correct,
        "total": total_count,
        "per_subject": per_subject,
        "n_shot": n_shot,
    }


def main():
    parser = argparse.ArgumentParser(description="MMLU 5-shot evaluation")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--n-shot", type=int, default=5)
    parser.add_argument("--max-samples-per-subject", type=int, default=0,
                        help="0 = run full test set; otherwise cap per subject")
    parser.add_argument("--output", type=str, default="experiments/results/mmlu_metrics.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    model, tokenizer = load_scratch_model(args.checkpoint, args.device, args.tokenizer)

    print(f"Evaluating MMLU ({args.n_shot}-shot)...")
    results = evaluate_mmlu(model, tokenizer, args.device, args.n_shot, args.max_samples_per_subject)
    results["model"] = str(args.checkpoint)

    print(f"\nMMLU Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
