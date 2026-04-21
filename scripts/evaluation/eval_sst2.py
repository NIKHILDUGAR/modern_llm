#!/usr/bin/env python3
"""SST-2 few-shot sentiment classification evaluation.

Evaluates models on binary sentiment (positive/negative) using few-shot prompting.
Designed for GPT-2 scale models where simple prompting works reasonably well.

Usage:
    python scripts/evaluation/eval_sst2.py --checkpoint path/to/model.pt
    python scripts/evaluation/eval_sst2.py --hf-model gpt2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# Few-shot prompt template - using question format (70% accuracy vs 50% for simple)
FEW_SHOT_PROMPT = """Is this review positive or negative?

Review: "I love this movie, it's fantastic!"
Answer: positive

Review: "This was terrible and boring."
Answer: negative

Review: "A wonderful experience from start to finish."
Answer: positive

Review: "{text}"
Answer:"""


def load_scratch_model(checkpoint_path: str, device: str):
    """Load our scratch-trained model from checkpoint."""
    from modern_llm.config.model_config import ModernLLMConfig
    from modern_llm.models.transformer import ModernDecoderLM

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config from checkpoint, normalizing any key name variations
    if "config" in checkpoint:
        cfg = checkpoint["config"].copy()
        # Handle key name variations between different config versions
        if "num_layers" in cfg and "n_layers" not in cfg:
            cfg["n_layers"] = cfg.pop("num_layers")
        if "max_position_embeddings" in cfg and "max_seq_len" not in cfg:
            cfg["max_seq_len"] = cfg.pop("max_position_embeddings")
        # Remove any keys that ModernLLMConfig doesn't accept
        valid_keys = {
            "vocab_size", "d_model", "n_layers", "n_heads", "ffn_hidden_size",
            "max_seq_len", "rmsnorm_eps", "dropout", "initializer_range",
            "rope_theta", "rope_scaling", "use_rope", "use_attention_sinks",
            "num_attention_sinks", "use_swiglu", "swiglu_multiplier", "use_gqa",
            "gqa_groups", "use_moe", "moe_config", "tie_embeddings",
        }
        cfg = {k: v for k, v in cfg.items() if k in valid_keys}
        config = ModernLLMConfig(**cfg)
    else:
        config = ModernLLMConfig()

    model = ModernDecoderLM(config)
    
    # Handle different checkpoint formats
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_hf_model(model_name: str, device: str):
    """Load a HuggingFace model for baseline comparison."""
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def predict_sentiment(
    model,
    tokenizer,
    text: str,
    device: str,
    is_hf_model: bool = True,
) -> str:
    """Predict sentiment for a single example using next-token prediction."""
    prompt = FEW_SHOT_PROMPT.format(text=text)
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        if is_hf_model:
            outputs = model(**inputs)
            logits = outputs.logits
        else:
            outputs = model(inputs["input_ids"])
            logits = outputs["logits"]

    # Get logits for the last token position
    next_logits = logits[0, -1, :]

    # Get probabilities for "positive" vs "negative" tokens
    pos_tokens = tokenizer.encode(" positive", add_special_tokens=False)
    neg_tokens = tokenizer.encode(" negative", add_special_tokens=False)

    pos_prob = next_logits[pos_tokens[0]].item() if pos_tokens else -float("inf")
    neg_prob = next_logits[neg_tokens[0]].item() if neg_tokens else -float("inf")

    return "positive" if pos_prob > neg_prob else "negative"


def evaluate_sst2(
    model,
    tokenizer,
    device: str,
    max_samples: int = 500,
    is_hf_model: bool = True,
) -> dict:
    """Evaluate on SST-2 validation set."""
    dataset = load_dataset("glue", "sst2", split="validation")

    #if max_samples and len(dataset) > max_samples:
    #    dataset = dataset.select(range(max_samples))

    correct = 0
    total = 0
    predictions = []

    for example in tqdm(dataset, desc="Evaluating SST-2"):
        text = example["sentence"]
        label = "positive" if example["label"] == 1 else "negative"

        pred = predict_sentiment(model, tokenizer, text, device, is_hf_model)
        predictions.append({"text": text, "label": label, "pred": pred})

        if pred == label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": predictions[:10],  # Save first 10 for inspection
    }


def main():
    parser = argparse.ArgumentParser(description="SST-2 few-shot evaluation")
    parser.add_argument("--checkpoint", type=str, help="Path to scratch model checkpoint")
    parser.add_argument("--hf-model", type=str, help="HuggingFace model name (e.g., gpt2)")
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples to evaluate")
    parser.add_argument("--output", type=str, default="experiments/results/sst2_results.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.checkpoint and not args.hf_model:
        parser.error("Either --checkpoint or --hf-model must be specified")

    # Load model
    if args.hf_model:
        print(f"Loading HF model: {args.hf_model}")
        model, tokenizer = load_hf_model(args.hf_model, args.device)
        model_name = args.hf_model
        is_hf = True
    else:
        print(f"Loading scratch model: {args.checkpoint}")
        model, tokenizer = load_scratch_model(args.checkpoint, args.device)
        model_name = Path(args.checkpoint).stem
        is_hf = False

    # Evaluate
    print(f"Evaluating on SST-2 (max all samples)...")
    results = evaluate_sst2(model, tokenizer, args.device, args.max_samples, is_hf)

    # Add metadata
    results["model"] = model_name
    results["is_hf_baseline"] = is_hf

    print(f"\nSST-2 Accuracy: {results['accuracy']:.2%} ({results['correct']}/{results['total']})")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

