"""Evaluate all scratch LM checkpoints and write metrics tables.

Loads checkpoints from experiments/runs/, runs validation, and writes
perplexity/loss CSV tables for inclusion in the final report.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from modern_llm.config import ModernLLMConfig, TrainingConfig
from modern_llm.data import LanguageModelingDatasetConfig, load_causal_lm_dataset
from modern_llm.models import ModernDecoderLM


def _load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[ModernDecoderLM, int]:
    """Load model weights and training step from checkpoint.

    Pre:
        - checkpoint_path points to a valid .pt file saved by Trainer.
    Post:
        - returns (model, global_step).
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state" not in state:
        raise ValueError(f"Checkpoint {checkpoint_path} missing 'model_state' key.")

    config_dict = state.get("config")
    if not config_dict:
        raise ValueError(f"Checkpoint {checkpoint_path} missing 'config' key; cannot rebuild model.")

    model_config = ModernLLMConfig(**config_dict)
    model = ModernDecoderLM(model_config)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()

    global_step = state.get("step", 0)
    return model, global_step


def _evaluate_model(
    model: ModernDecoderLM,
    eval_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute validation loss and perplexity.

    Math:
        loss = (1/N) Σ CrossEntropy(logits, labels)
        perplexity = exp(loss)

    Pre:
        - model is in eval mode and on device.
    Post:
        - returns dict with 'loss' and 'perplexity'.
    """
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )
            loss = outputs.get("loss")
            if loss is None:
                raise ValueError("Model did not return a loss during evaluation.")
            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    perplexity = math.exp(avg_loss)
    return {"loss": avg_loss, "perplexity": perplexity}


def main() -> None:
    """Evaluate all scratch LM checkpoints and write results to CSV."""

    parser = argparse.ArgumentParser(description="Evaluate scratch LM checkpoints and write metrics tables.")
    parser.add_argument("--runs_dir", type=str, default="experiments/runs")
    parser.add_argument("--output_csv", type=str, default="experiments/lm_checkpoint_metrics.csv")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--tokenizer_name", type=str, default="Xenova/text-embedding-ada-002")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_proc", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_len

    eval_dataset = load_causal_lm_dataset(
        LanguageModelingDatasetConfig(
            dataset_name=args.dataset_name,
            dataset_config_name=args.dataset_config_name,
            split="validation",
            max_length=args.max_seq_len,
            num_proc=args.num_proc,
        ),
        tokenizer,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    checkpoint_paths = sorted(runs_dir.rglob("*.pt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {runs_dir}")

    results: List[Dict[str, any]] = []
    for ckpt_path in checkpoint_paths:
        run_name = ckpt_path.parent.name
        ckpt_name = ckpt_path.stem
        print(f"Evaluating {run_name}/{ckpt_name}...")

        try:
            model, global_step = _load_checkpoint(ckpt_path, device)
            metrics = _evaluate_model(model, eval_loader, device)
            results.append({
                "run_name": run_name,
                "checkpoint": ckpt_name,
                "global_step": global_step,
                "val_loss": metrics["loss"],
                "val_perplexity": metrics["perplexity"],
            })
            print(f"  Step {global_step}: loss={metrics['loss']:.4f}, ppl={metrics['perplexity']:.2f}")
        except Exception as exc:
            print(f"  Skipping {ckpt_path}: {exc}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    print(f"\nWrote {len(results)} checkpoint metrics to {output_path}")


if __name__ == "__main__":
    main()



