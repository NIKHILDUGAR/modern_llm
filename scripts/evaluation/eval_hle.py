#!/usr/bin/env python3
"""HLE (Humanity's Last Exam, CAIS 2025) evaluation.

HLE is a frontier-closed-book benchmark with a mix of multiple-choice and
short-answer items. It is hosted (gated) at `cais/hle` on HuggingFace. The
dataset fields we rely on:
    - question: str
    - answer: str (for exact-match items)
    - answer_type: "multipleChoice" or "exactMatch"
    - options / choices: list[str] (for MC)
    - correct_answer_letter: str (for MC)

Protocol:
    - MC items: standard letter-prediction log-likelihood (lm-eval style).
    - Short-answer items: greedy generation with a short answer template and
      exact-match scoring after basic normalization (strip punctuation, lower).

This is a lightweight local scorer — the official HLE pipeline uses an
LLM-judge for open-ended answers. For a small scratch model this is a sane
conservative estimate; we surface `exact_match` separately from `mc_accuracy`.

Usage:
    python scripts/evaluation/eval_hle.py \\
        --checkpoint experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _eval_common import (
    DEFAULT_TOKENIZER,
    greedy_generate,
    load_scratch_model,
    mc_argmax,
)

DEFAULT_CKPT = "experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt"
LETTERS = [chr(ord("A") + i) for i in range(26)]


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return s.strip()


def build_mc_prompt(question: str, options: list[str]) -> str:
    prompt = f"Question: {question.strip()}"
    for letter, opt in zip(LETTERS, options):
        prompt += f"\n{letter}. {opt}"
    prompt += "\nAnswer:"
    return prompt


def build_short_answer_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"


def evaluate_hle(model, tokenizer, device: str, max_samples: int) -> dict:
    # cais/hle is the canonical source (gated); try a couple of fallbacks.
    last_err = None
    for name in ("cais/hle", "cais/HLE", "cais/humanitys-last-exam"):
        try:
            ds = load_dataset(name, split="test")
            break
        except Exception as e:
            last_err = e
            ds = None
    if ds is None:
        msg = ("Could not load Humanity's Last Exam dataset. "
               "HLE is gated at `cais/hle` — accept the license and login via "
               "`huggingface-cli login`. Underlying error: " + str(last_err))
        print(msg, file=sys.stderr)
        return {"error": str(last_err), "mc_accuracy": None, "exact_match": None,
                "mc_total": 0, "short_total": 0}

    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    mc_correct = 0
    mc_total = 0
    short_correct = 0
    short_total = 0

    for ex in tqdm(ds, desc="Evaluating HLE"):
        # Field-name tolerance — HLE has shifted schemas in community mirrors.
        question = ex.get("question") or ex.get("Question") or ex.get("prompt", "")
        answer_type = (ex.get("answer_type") or ex.get("type") or "").lower()
        options = ex.get("options") or ex.get("choices") or ex.get("answer_choices")
        gold_answer = ex.get("answer") or ex.get("correct_answer") or ""

        if options and isinstance(options, (list, tuple)) and len(options) >= 2:
            prompt = build_mc_prompt(question, list(options))
            choices = [f" {LETTERS[i]}" for i in range(len(options))]
            pred_idx = mc_argmax(model, tokenizer, prompt, choices, device, length_normalize=False)
            # resolve gold letter
            gold_letter = (ex.get("correct_answer_letter") or ex.get("answer_letter") or "").strip().upper()
            if gold_letter and gold_letter in LETTERS:
                gold_idx = LETTERS.index(gold_letter)
            else:
                # Fall back to matching by text
                gold_text = str(gold_answer).strip()
                try:
                    gold_idx = list(options).index(gold_text)
                except ValueError:
                    gold_idx = -1
            if gold_idx >= 0:
                mc_correct += int(pred_idx == gold_idx)
                mc_total += 1
        else:
            if not gold_answer:
                continue
            prompt = build_short_answer_prompt(question)
            pred = greedy_generate(model, tokenizer, prompt, device,
                                   max_new_tokens=64, stop_strings=["\n", "Question:"])
            if normalize(pred) == normalize(str(gold_answer)) or \
               normalize(str(gold_answer)) in normalize(pred):
                short_correct += 1
            short_total += 1

    return {
        "mc_accuracy": (mc_correct / mc_total) if mc_total else None,
        "mc_correct": mc_correct,
        "mc_total": mc_total,
        "exact_match": (short_correct / short_total) if short_total else None,
        "short_correct": short_correct,
        "short_total": short_total,
    }


def main():
    parser = argparse.ArgumentParser(description="HLE (Humanity's Last Exam) evaluation")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", type=str, default="experiments/results/hle_metrics.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    model, tokenizer = load_scratch_model(args.checkpoint, args.device, args.tokenizer)

    print("Evaluating HLE...")
    results = evaluate_hle(model, tokenizer, args.device, args.max_samples)
    results["model"] = str(args.checkpoint)

    if results.get("mc_accuracy") is not None:
        print(f"\nHLE MC accuracy: {results['mc_accuracy']:.2%} ({results['mc_correct']}/{results['mc_total']})")
    if results.get("exact_match") is not None:
        print(f"HLE exact match: {results['exact_match']:.2%} ({results['short_correct']}/{results['short_total']})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
