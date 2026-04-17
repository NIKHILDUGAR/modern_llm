#!/usr/bin/env python3
"""CoQA (Reddy et al., 2019) conversational question answering.

Protocol:
    For each document, iterate through the conversation turn by turn, prefixing
    prior Q/A pairs. Generate the answer greedily and score with the SQuAD-style
    F1 (macro-averaged across turns).

Usage:
    python scripts/evaluation/eval_coqa.py \\
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
    """SQuAD-style normalizer (articles/punct/lower/whitespace)."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    p = overlap / len(pred_tokens)
    r = overlap / len(gold_tokens)
    return 2 * p * r / (p + r)


def em_score(pred: str, gold: str) -> int:
    return int(normalize_answer(pred) == normalize_answer(gold))


def evaluate_coqa(model, tokenizer, device: str, max_docs: int) -> dict:
    ds = load_dataset("stanfordnlp/coqa", split="validation")
    if max_docs and max_docs > 0:
        ds = ds.select(range(min(max_docs, len(ds))))

    total_f1 = 0.0
    total_em = 0.0
    total_turns = 0

    for ex in tqdm(ds, desc="Evaluating CoQA"):
        story = ex["story"]
        questions = ex["questions"]
        answers = ex["answers"]["input_text"]
        history = ""
        for q, gold in zip(questions, answers):
            prompt = (
                f"Passage: {story}\n"
                f"{history}"
                f"Q: {q}\nA:"
            )
            pred = greedy_generate(model, tokenizer, prompt, device,
                                   max_new_tokens=32, stop_strings=["\n", "Q:", "Passage:"])
            pred = pred.strip()
            total_f1 += f1_score(pred, gold)
            total_em += em_score(pred, gold)
            total_turns += 1
            history += f"Q: {q}\nA: {gold}\n"

    return {
        "f1": total_f1 / total_turns if total_turns else 0.0,
        "exact_match": total_em / total_turns if total_turns else 0.0,
        "total_turns": total_turns,
    }


def main():
    parser = argparse.ArgumentParser(description="CoQA evaluation")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-docs", type=int, default=50,
                        help="Number of conversations to evaluate (each has many turns)")
    parser.add_argument("--output", type=str, default="experiments/results/coqa_metrics.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    model, tokenizer = load_scratch_model(args.checkpoint, args.device, args.tokenizer)

    print("Evaluating CoQA...")
    results = evaluate_coqa(model, tokenizer, args.device, args.max_docs)
    results["model"] = str(args.checkpoint)

    print(f"\nCoQA F1: {results['f1']:.2%}  EM: {results['exact_match']:.2%}  turns={results['total_turns']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
