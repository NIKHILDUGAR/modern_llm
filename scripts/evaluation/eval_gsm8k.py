#!/usr/bin/env python3
"""GSM8K evaluation with verifier reranking and error analysis.

Evaluates math problem solving with optional verifier-based reranking.
Provides error taxonomy and false positive/negative analysis.

Usage:
    python scripts/evaluation/eval_gsm8k.py --checkpoint path/to/model.pt
    python scripts/evaluation/eval_gsm8k.py --checkpoint model.pt --verifier verifier.pt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

# Add local eval helpers and src to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from _eval_common import DEFAULT_TOKENIZER, load_scratch_model


def extract_answer(text: str) -> Optional[str]:
    """Extract final numeric answer from model output."""
    # Look for #### marker first (GSM8K format)
    match = re.search(r"####\s*([\d,.-]+)", text)
    if match:
        return match.group(1).replace(",", "")
    
    # Look for "the answer is X" pattern
    match = re.search(r"answer is\s*([\d,.-]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).replace(",", "")
    
    # Fall back to last number in text
    numbers = re.findall(r"[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer is None:
        return ""
    # Remove commas, strip whitespace
    return answer.replace(",", "").strip()


def classify_error(question: str, pred: str, gold: str, model_output: str) -> str:
    """Classify the type of error made.
    
    Categories:
    - extraction: Answer present but not extracted correctly
    - arithmetic: Calculation error in chain of thought
    - reasoning: Wrong approach/logic
    """
    pred_norm = normalize_answer(pred) if pred else ""
    gold_norm = normalize_answer(gold)
    
    # If gold answer appears in output but wasn't extracted
    if gold_norm in model_output:
        return "extraction"
    
    # Check for numbers close to correct (arithmetic errors)
    try:
        pred_num = float(pred_norm) if pred_norm else 0
        gold_num = float(gold_norm)
        # Within 10% or off by small factor
        if abs(pred_num - gold_num) < abs(gold_num) * 0.2:
            return "arithmetic"
    except ValueError:
        pass
    
    return "reasoning"


def load_verifier(verifier_path: str, device: str):
    """Load verifier model."""
    from modern_llm.alignment.verifier import Verifier
    from modern_llm.config.model_config import ModernLLMConfig

    checkpoint = torch.load(verifier_path, map_location=device, weights_only=False)
    
    if "config" in checkpoint:
        config = ModernLLMConfig(**checkpoint["config"])
    else:
        config = ModernLLMConfig()

    verifier = Verifier(config)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    verifier.load_state_dict(state_dict, strict=False)
    verifier.to(device)
    verifier.eval()

    return verifier


def generate_solutions(
    model,
    tokenizer,
    question: str,
    device: str,
    n_samples: int = 1,
    max_length: int = 256,
) -> list[str]:
    """Generate solutions for a math question."""
    prompt = f"Question: {question}\n\nLet me solve this step by step:\n"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    solutions = []
    with torch.no_grad():
        for _ in range(n_samples):
            # Simple greedy decode for scratch model
            generated = inputs["input_ids"].clone()
            for _ in range(max_length):
                outputs = model(generated)
                logits = outputs["logits"]
                next_token = logits[0, -1, :].argmax()
                generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
                if next_token == tokenizer.eos_token_id:
                    break
            
            output = tokenizer.decode(generated[0], skip_special_tokens=True)
            solutions.append(output[len(prompt):])

    return solutions


def evaluate_gsm8k(
    model,
    tokenizer,
    device: str,
    verifier=None,
    max_samples: int = 100,
    n_samples_per_q: int = 1,
) -> dict:
    """Evaluate on GSM8K with optional verifier reranking."""
    dataset = load_dataset("gsm8k", "main", split="test")

    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    results_no_verifier = {"correct": 0, "total": 0}
    results_with_verifier = {"correct": 0, "total": 0}
    
    error_counts = {"extraction": 0, "arithmetic": 0, "reasoning": 0}
    examples = {"false_positives": [], "false_negatives": [], "errors": []}

    for example in tqdm(dataset, desc="Evaluating GSM8K"):
        question = example["question"]
        gold_answer = example["answer"].split("####")[-1].strip()

        # Generate solutions
        solutions = generate_solutions(model, tokenizer, question, device, n_samples_per_q)
        
        # Without verifier: just use first solution
        pred_answer = extract_answer(solutions[0])
        is_correct_no_v = normalize_answer(pred_answer) == normalize_answer(gold_answer)
        
        results_no_verifier["total"] += 1
        if is_correct_no_v:
            results_no_verifier["correct"] += 1
        else:
            error_type = classify_error(question, pred_answer, gold_answer, solutions[0])
            error_counts[error_type] += 1
            if len(examples["errors"]) < 5:
                examples["errors"].append({
                    "question": question[:100],
                    "gold": gold_answer,
                    "pred": pred_answer,
                    "error_type": error_type,
                })

        # With verifier (if available)
        if verifier and n_samples_per_q > 1:
            scores = []
            for sol in solutions:
                score = verifier.score(
                    tokenizer(question + sol, return_tensors="pt")["input_ids"].to(device)
                )
                scores.append((score, sol))
            
            best_sol = max(scores, key=lambda x: x[0])[1]
            best_pred = extract_answer(best_sol)
            is_correct_with_v = normalize_answer(best_pred) == normalize_answer(gold_answer)
            
            results_with_verifier["total"] += 1
            if is_correct_with_v:
                results_with_verifier["correct"] += 1
            
            # Track false positives/negatives
            if is_correct_with_v and not is_correct_no_v:
                if len(examples["false_negatives"]) < 3:
                    examples["false_negatives"].append({
                        "question": question[:100],
                        "gold": gold_answer,
                        "note": "Verifier correctly rescued this"
                    })
            elif not is_correct_with_v and is_correct_no_v:
                if len(examples["false_positives"]) < 3:
                    examples["false_positives"].append({
                        "question": question[:100],
                        "gold": gold_answer,
                        "note": "Verifier wrongly rejected correct answer"
                    })

    # Compute metrics
    em_no_v = results_no_verifier["correct"] / results_no_verifier["total"]
    
    output = {
        "exact_match_no_verifier": em_no_v,
        "correct_no_verifier": results_no_verifier["correct"],
        "total": results_no_verifier["total"],
        "error_taxonomy": error_counts,
        "examples": examples,
    }
    
    if verifier and results_with_verifier["total"] > 0:
        em_with_v = results_with_verifier["correct"] / results_with_verifier["total"]
        output["exact_match_with_verifier"] = em_with_v
        output["correct_with_verifier"] = results_with_verifier["correct"]
        output["verifier_improvement"] = em_with_v - em_no_v

    return output


def main():
    parser = argparse.ArgumentParser(description="GSM8K evaluation with verifier")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--verifier", type=str, help="Verifier checkpoint (optional)")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--n-samples", type=int, default=1, help="Solutions per question")
    parser.add_argument("--output", type=str, default="experiments/results/gsm8k_results.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    model, tokenizer = load_scratch_model(args.checkpoint, args.device, args.tokenizer)

    verifier = None
    if args.verifier:
        print(f"Loading verifier: {args.verifier}")
        verifier = load_verifier(args.verifier, args.device)

    print(f"Evaluating on GSM8K (max all samples)...")
    results = evaluate_gsm8k(
        model, tokenizer, args.device,
        verifier=verifier,
        max_samples=args.max_samples,
        n_samples_per_q=args.n_samples,
    )

    results["model"] = str(args.checkpoint)
    results["has_verifier"] = args.verifier is not None

    print(f"\n{'='*50}")
    print("GSM8K Results")
    print(f"{'='*50}")
    print(f"Exact Match (no verifier): {results['exact_match_no_verifier']:.2%}")
    if "exact_match_with_verifier" in results:
        print(f"Exact Match (with verifier): {results['exact_match_with_verifier']:.2%}")
        print(f"Verifier improvement: {results['verifier_improvement']:+.2%}")
    print(f"\nError Taxonomy:")
    for error_type, count in results["error_taxonomy"].items():
        print(f"  {error_type}: {count}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
