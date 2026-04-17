#!/usr/bin/env python3
"""MixEval (Easy split) evaluation (Ni et al., 2024).

MixEval blends items from many existing benchmarks (MMLU, TriviaQA, BoolQ,
HellaSwag, DROP, GSM8K, etc.) into two splits — Hard and Easy. The original
evaluation uses an LLM-judge for free-form answers, which is not appropriate
for a small scratch model; we keep things local:

    - multiple-choice items  -> letter log-likelihood (lm-eval style)
    - free-form items        -> greedy generation + SQuAD-style F1 / exact match

The dataset is at `MixEval/MixEval`. We filter to the "easy" / "free-form" +
"multiple-choice" splits covered by mixeval_easy. Field names vary across
dataset versions, so we detect them defensively.

Usage:
    python scripts/evaluation/eval_mixeval_easy.py \\
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
from _eval_common import (
    DEFAULT_TOKENIZER,
    greedy_generate,
    load_scratch_model,
    mc_argmax,
)

DEFAULT_CKPT = "experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt"
LETTERS = [chr(ord("A") + i) for i in range(26)]


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def f1_score(pred: str, gold: str) -> float:
    p_toks = normalize(pred).split()
    g_toks = normalize(gold).split()
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    p = overlap / len(p_toks)
    r = overlap / len(g_toks)
    return 2 * p * r / (p + r)


def _pull(ex, *names, default=None):
    for n in names:
        if n in ex and ex[n] is not None:
            return ex[n]
    return default


def _load_mixeval_easy():
    """Try a few repo names / configs; MixEval has shifted schemas over time."""
    candidates = [
        ("MixEval/MixEval", "MixEval_free-form", "train"),
        ("MixEval/MixEval", "MixEval_multi-choice", "train"),
        ("MixEval/MixEval", "free-form", "train"),
        ("MixEval/MixEval", "multiple-choice", "train"),
        ("MixEval/MixEval", None, "train"),
    ]
    datasets_loaded = []
    last_err = None
    for repo, cfg, split in candidates:
        try:
            if cfg:
                ds = load_dataset(repo, cfg, split=split)
            else:
                ds = load_dataset(repo, split=split)
            datasets_loaded.append((cfg or "default", ds))
        except Exception as e:
            last_err = e
            continue
    if not datasets_loaded:
        raise RuntimeError(f"Could not load MixEval_Easy from any known repo/config: {last_err}")
    return datasets_loaded


def evaluate_mixeval_easy(model, tokenizer, device: str, max_samples: int) -> dict:
    try:
        subsets = _load_mixeval_easy()
    except Exception as e:
        msg = ("Could not load MixEval (easy split). The dataset may be gated. "
               "Try accepting the license at https://huggingface.co/datasets/MixEval/MixEval "
               f"and running `huggingface-cli login`. Underlying error: {e}")
        print(msg, file=sys.stderr)
        return {"error": str(e), "mc_accuracy": None, "freeform_f1": None,
                "mc_total": 0, "freeform_total": 0}

    mc_correct = 0
    mc_total = 0
    ff_f1 = 0.0
    ff_em = 0.0
    ff_total = 0

    for subset_name, ds in subsets:
        is_easy = lambda e: str(_pull(e, "split", "difficulty", "benchmark_split", default="")).lower().find("easy") >= 0
        ds_easy = ds.filter(is_easy) if any(k in ds.column_names for k in ("split", "difficulty", "benchmark_split")) else ds
        if len(ds_easy) == 0:
            ds_easy = ds  # fall back — some mirrors don't label "easy" explicitly
        if max_samples and max_samples > 0:
            ds_easy = ds_easy.select(range(min(max_samples, len(ds_easy))))

        for ex in tqdm(ds_easy, desc=f"MixEval {subset_name}"):
            question = _pull(ex, "question", "prompt", "problem", default="")
            options = _pull(ex, "options", "choices", "answer_choices")
            gold = _pull(ex, "answer", "target", "correct_answer", "reference_answer", default="")
            gold_letter = _pull(ex, "correct_answer_letter", "answer_letter", default="")

            if options and isinstance(options, (list, tuple)) and len(options) >= 2:
                prompt = f"Question: {question.strip()}"
                for letter, opt in zip(LETTERS, options):
                    prompt += f"\n{letter}. {opt}"
                prompt += "\nAnswer:"
                choices = [f" {LETTERS[i]}" for i in range(len(options))]
                pred_idx = mc_argmax(model, tokenizer, prompt, choices, device, length_normalize=False)
                if gold_letter and str(gold_letter).strip().upper() in LETTERS:
                    gold_idx = LETTERS.index(str(gold_letter).strip().upper())
                else:
                    try:
                        gold_idx = list(options).index(str(gold).strip())
                    except ValueError:
                        gold_idx = -1
                if gold_idx >= 0:
                    mc_correct += int(pred_idx == gold_idx)
                    mc_total += 1
            else:
                prompt = f"Question: {question.strip()}\nAnswer:"
                pred = greedy_generate(model, tokenizer, prompt, device,
                                       max_new_tokens=64, stop_strings=["\n", "Question:"])
                pred = pred.strip()
                golds = gold if isinstance(gold, list) else [gold]
                best_f1 = max((f1_score(pred, g) for g in golds if g), default=0.0)
                best_em = max((int(normalize(pred) == normalize(g)) for g in golds if g), default=0)
                ff_f1 += best_f1
                ff_em += best_em
                ff_total += 1

    return {
        "mc_accuracy": (mc_correct / mc_total) if mc_total else None,
        "mc_correct": mc_correct,
        "mc_total": mc_total,
        "freeform_f1": (ff_f1 / ff_total) if ff_total else None,
        "freeform_em": (ff_em / ff_total) if ff_total else None,
        "freeform_total": ff_total,
    }


def main():
    parser = argparse.ArgumentParser(description="MixEval Easy evaluation")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Cap per subset (MC and free-form are split into subsets)")
    parser.add_argument("--output", type=str, default="experiments/results/mixeval_easy_metrics.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    model, tokenizer = load_scratch_model(args.checkpoint, args.device, args.tokenizer)

    print("Evaluating MixEval_Easy...")
    results = evaluate_mixeval_easy(model, tokenizer, args.device, args.max_samples)
    results["model"] = str(args.checkpoint)

    if results.get("mc_accuracy") is not None:
        print(f"\nMixEval MC accuracy : {results['mc_accuracy']:.2%} ({results['mc_correct']}/{results['mc_total']})")
    if results.get("freeform_f1") is not None:
        print(f"MixEval free-form F1: {results['freeform_f1']:.2%} (EM {results['freeform_em']:.2%}) "
              f"over {results['freeform_total']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
