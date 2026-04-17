#!/usr/bin/env python3
"""IFBench test-set evaluation (Ivison et al., 2024 — AI2).

IFBench is an extension of IFEval with a broader constraint taxonomy, hosted
by Allen AI. The test split lives at `allenai/IFBench_test` on HuggingFace.
The evaluation protocol is identical to IFEval: each prompt carries a list of
machine-verifiable constraints; we report prompt-level and instruction-level
accuracy.

We reuse the constraint-verifier registry from `eval_ifeval.py`. Any
instruction ids that IFBench adds beyond IFEval will be logged under
`unverified_constraint_counts` so the metric denominator stays honest — they
are counted as failures, never silent passes.

NOTE: `allenai/IFBench_test` is publicly gated as of writing. If loading fails,
we emit a clear error with remediation instructions.

Usage:
    python scripts/evaluation/eval_ifbench_test.py \\
        --checkpoint experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _eval_common import DEFAULT_TOKENIZER, greedy_generate, load_scratch_model
# Reuse verifier registry from IFEval (pure functions, no state)
from eval_ifeval import check_response

DEFAULT_CKPT = "experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt"


def _pull_field(ex: dict, *names, default=None):
    for n in names:
        if n in ex and ex[n] is not None:
            return ex[n]
    return default


def evaluate_ifbench(model, tokenizer, device: str, max_samples: int) -> dict:
    last_err = None
    ds = None
    for repo in ("allenai/IFBench_test", "allenai/IFBench", "allenai/ifbench_test"):
        try:
            ds = load_dataset(repo, split="test")
            break
        except Exception as e:
            last_err = e
            try:
                ds = load_dataset(repo, split="train")
                break
            except Exception as e2:
                last_err = e2
                continue
    if ds is None:
        msg = ("Could not load IFBench_test — the AI2 repo may be gated or "
               "renamed. Try `huggingface-cli login` after accepting the "
               "license at https://huggingface.co/datasets/allenai/IFBench_test. "
               f"Underlying error: {last_err}")
        print(msg, file=sys.stderr)
        return {"error": str(last_err),
                "prompt_accuracy": None, "instruction_accuracy": None,
                "prompts_total": 0, "instructions_total": 0}

    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    n_prompts = 0
    n_prompts_pass = 0
    n_instructions = 0
    n_instructions_pass = 0
    unverified = Counter()

    for ex in tqdm(ds, desc="Evaluating IFBench_test"):
        prompt = _pull_field(ex, "prompt", "instruction", "input", default="")
        inst_ids = _pull_field(ex, "instruction_id_list", "constraints", default=[]) or []
        kwargs_list = _pull_field(ex, "kwargs", "constraint_kwargs", default=None)
        if kwargs_list is None:
            kwargs_list = [{} for _ in inst_ids]

        response = greedy_generate(model, tokenizer, prompt, device,
                                   max_new_tokens=256)

        prompt_pass = True
        for iid, kw in zip(inst_ids, kwargs_list):
            kw = kw or {}
            result = check_response(response, iid, kw)
            if result is None:
                unverified[iid] += 1
                prompt_pass = False
                continue
            n_instructions += 1
            if result:
                n_instructions_pass += 1
            else:
                prompt_pass = False
        n_prompts += 1
        if prompt_pass:
            n_prompts_pass += 1

    return {
        "prompt_accuracy": n_prompts_pass / n_prompts if n_prompts else 0.0,
        "instruction_accuracy": n_instructions_pass / n_instructions if n_instructions else 0.0,
        "prompts_total": n_prompts,
        "prompts_pass": n_prompts_pass,
        "instructions_total": n_instructions,
        "instructions_pass": n_instructions_pass,
        "unverified_constraint_counts": dict(unverified),
    }


def main():
    parser = argparse.ArgumentParser(description="IFBench_test instruction-following evaluation")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", type=str, default="experiments/results/ifbench_test_metrics.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    model, tokenizer = load_scratch_model(args.checkpoint, args.device, args.tokenizer)

    print("Evaluating IFBench_test...")
    results = evaluate_ifbench(model, tokenizer, args.device, args.max_samples)
    results["model"] = str(args.checkpoint)

    if results.get("prompt_accuracy") is not None:
        print(f"\nIFBench prompt accuracy     : {results['prompt_accuracy']:.2%}")
        print(f"IFBench instruction accuracy: {results['instruction_accuracy']:.2%}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
