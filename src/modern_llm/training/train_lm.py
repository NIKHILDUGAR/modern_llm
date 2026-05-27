"""Language modeling training entrypoint (causal LM on WikiText-2/TinyStories)."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from modern_llm.config import ModernLLMConfig, TrainingConfig
from modern_llm.data import LanguageModelingDatasetConfig, load_causal_lm_dataset, load_multi_dataset
from modern_llm.data.lm_datasets import (
    load_packed_pretrain_dataset,
    load_packed_pretrain_train_eval_split,
    make_lm_dataloader,
)
from modern_llm.models import ModernDecoderLM
from modern_llm.quantization import prepare_model_for_quantization
from modern_llm.training.distributed import is_main_process, scale_grad_accum_for_world_size, world_size
from modern_llm.training.trainer_base import Trainer
from modern_llm.utils.paths import apply_env_defaults

# Make sure HF cache env defaults are set as soon as this module is imported.
apply_env_defaults()


def _sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Tensor:
    """Sample the next token id from logits with optional temperature and top-k truncation.

    Pre:
        - logits: shape (vocab_size,) on a single device.
        - temperature > 0.
    Post:
        - returns a scalar tensor containing the sampled token id.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, received {temperature}")

    logits = logits / temperature
    if top_k is not None and top_k > 0 and top_k < logits.size(-1):
        values, indices = torch.topk(logits, top_k)
        min_topk = values[..., -1]
        logits = torch.where(logits < min_topk, torch.full_like(logits, float("-inf")), logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_text(
    model: ModernDecoderLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
) -> str:
    """Generate text from a trained `ModernDecoderLM` given a prompt.

    Pre:
        - `max_new_tokens` > 0.
        - `prompt` is a non-empty string whose tokenized length is < model.config.max_seq_len.
    Post:
        - returns the decoded prompt + continuation string.
    """
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, received {max_new_tokens}")
    if not prompt:
        raise ValueError("prompt must be a non-empty string")

    model.eval()
    device = next(model.parameters()).device

    encoded = tokenizer.encode(prompt, return_tensors="pt")
    if encoded.dim() != 2 or encoded.size(0) != 1:
        raise ValueError("tokenizer.encode must return tensors of shape (1, seq_len)")
    input_ids = encoded.to(device)

    max_seq_len = model.config.max_seq_len
    if input_ids.size(1) >= max_seq_len:
        raise ValueError(
            f"Prompt length {input_ids.size(1)} exceeds or equals model max_seq_len {max_seq_len}"
        )

    available_tokens = max_seq_len - input_ids.size(1)
    steps = min(max_new_tokens, available_tokens)

    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        for _ in range(steps):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            if logits is None:
                raise ValueError("Model did not return logits during generation.")
            next_token_logits = logits[:, -1, :].squeeze(0)
            next_token = _sample_next_token(next_token_logits, temperature=temperature, top_k=top_k)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            attention_mask = torch.ones_like(input_ids, device=device)

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
import torch.nn as nn

def print_model_parameters(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Percentage Trainable: {100 * trainable_params / total_params:.4f}%\n")
    
    # Optional: Print trainable layers specifically
    print("Trainable Layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"- {name}: {param.shape}")

# Usage:

def run_training(
    model_config: ModernLLMConfig,
    train_config: TrainingConfig,
    dataset_names: Optional[list] = None,
    tokenizer_name: str ="Xenova/text-embedding-ada-002",
    packed_shards_dir: Optional[str] = None,
    packed_eval_windows: int = 256,
) -> Path:
    """Run pretraining and return path to final checkpoint.

    Pre: model_config and train_config are valid.
    Post: Returns path to final checkpoint file.

    Args:
        model_config: Model architecture config
        train_config: Training hyperparameters
        dataset_names: List of dataset names to train on (HF streaming path).
            Ignored when `packed_shards_dir` is provided.
        tokenizer_name: Tokenizer to use.
        packed_shards_dir: If set, read packed uint32 shards produced by
            scripts/data/tokenize_pretrain.py from this directory instead of
            streaming raw HF datasets. Expected layout: <dir>/index.json +
            <dir>/shard_*.bin.
    """
    # Default to WikiText-2 if no datasets specified
    if dataset_names is None:
        dataset_names = ["wikitext-2-raw-v1"]
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = model_config.max_seq_len

    # Update vocab size from tokenizer while preserving every other field
    # (QK-norm, z_loss_coef, scale_embeddings, rope_theta, ...) — rebuilding
    # field-by-field silently dropped those in earlier versions.
    from dataclasses import replace as _dc_replace
    model_config = _dc_replace(model_config, vocab_size=tokenizer.vocab_size)

    # Load training data (single or multiple datasets)
    from modern_llm.data.lm_datasets import DATASET_REGISTRY

    if packed_shards_dir:
        if is_main_process():
            print(f"Loading packed pretrain shards from: {packed_shards_dir}")
        # Hold out the last `packed_eval_windows` windows as eval so the
        # eval distribution matches training (same tokenizer, same packing,
        # no padding, no attention-mask-zero rows that produce NaN softmax).
        train_dataset, eval_dataset = load_packed_pretrain_train_eval_split(
            packed_shards_dir,
            seq_len=model_config.max_seq_len,
            eval_windows=packed_eval_windows,
        )
    elif len(dataset_names) == 1 and dataset_names[0] in ["wikitext-2-raw-v1", "wikitext-103-raw-v1"]:
        # Single wikitext dataset - use original loader for validation split
        hf_name, hf_config, _ = DATASET_REGISTRY[dataset_names[0]]
        train_dataset = load_causal_lm_dataset(
            LanguageModelingDatasetConfig(
                dataset_name=hf_name,
                dataset_config_name=hf_config,
                split="train",
                max_length=model_config.max_seq_len,
            ),
            tokenizer,
        )
        eval_dataset = load_causal_lm_dataset(
            LanguageModelingDatasetConfig(
                dataset_name=hf_name,
                dataset_config_name=hf_config,
                split="validation",
                max_length=model_config.max_seq_len,
            ),
            tokenizer,
        )
    else:
        # Multiple datasets - concatenate them
        print(f"Loading {len(dataset_names)} datasets: {dataset_names}")
        train_dataset = load_multi_dataset(
            dataset_names,
            tokenizer,
            split="train",
            max_length=model_config.max_seq_len,
        )
        # For eval, just use WikiText-2 validation (standard benchmark)
        hf_name, hf_config, _ = DATASET_REGISTRY["wikitext-2-raw-v1"]
        eval_dataset = load_causal_lm_dataset(
            LanguageModelingDatasetConfig(
                dataset_name=hf_name,
                dataset_config_name=hf_config,
                split="validation",
                max_length=model_config.max_seq_len,
            ),
            tokenizer,
        )

    # Use the DDP-aware loader builder. Under WORLD_SIZE>1 each rank gets a
    # disjoint slice via DistributedSampler; otherwise behaves like a normal
    # shuffled DataLoader.
    train_loader = make_lm_dataloader(
        train_dataset,
        micro_batch_size=train_config.micro_batch_size,
        shuffle=True,
        num_workers=16,
        seed=train_config.seed or 42,
    )
    eval_loader = make_lm_dataloader(
        eval_dataset,
        micro_batch_size=train_config.micro_batch_size,
        shuffle=False,
        num_workers=4,
    )

    model = ModernDecoderLM(model_config)
    if train_config.quantization is not None and train_config.quantization.enabled:
        train_config.compile_model = False
        summary = prepare_model_for_quantization(model, train_config.quantization)
        if is_main_process():
            print(
                f"Quantization enabled: mode={summary.mode} "
                f"replaced_modules={len(summary.replaced_modules)}"
            )
    if is_main_process():
        print("=================")
        print_model_parameters(model)
        print(
            f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters "
            f"(world_size={world_size()})"
        )
        print("=================")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    warm = max(train_config.warmup_steps, 1)
    total = max(train_config.max_steps, warm + 1)
    min_ratio = train_config.min_lr_ratio

    # Optional env override that specifies a step where the LR should reach
    # the cosine minimum (plateau). When set, scale the cosine "progress"
    # so the scheduler reaches the minimum at that step instead of at `total`.
    plateau_env = 0
    try:
        plateau_env = int(os.environ.get("PRETRAIN_LR_PLATEAU_STEP", "0") or 0)
    except Exception:
        plateau_env = 0

    if plateau_env and plateau_env > warm:
        plateau = plateau_env
        scale = max(1.0, (total - warm) / max(1, (plateau - warm)))
        if is_main_process():
            print(f"[train_lm] PRETRAIN_LR_PLATEAU_STEP={plateau} -> scaling lr progress by {scale:.6f}")
    else:
        # Preserve previous heuristic multiplier when no plateau override provided.
        scale = 1.3

    def lr_lambda(step: int) -> float:
        if step < train_config.warmup_steps:
            return float(step + 1) / float(warm)
        progress = (step - warm) / max(total - warm, 1)
        progress = min(max(progress * scale, 0.0), 1.0)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        config=train_config,
        lr_scheduler=scheduler,
    )
    trainer.train()

    final_checkpoint = train_config.output_dir / f"{train_config.run_name}_final.pt"
    return final_checkpoint


def main() -> None:
    """Train the from-scratch decoder LM on WikiText-2 or TinyStories."""

    parser = argparse.ArgumentParser(description="Train Modern Decoder LM on a causal LM corpus.")
    parser.add_argument("--run_name", type=str, default="scratch-lm")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--tokenizer_name", type=str, default="Xenova/text-embedding-ada-002")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--ffn_hidden_size", type=int, default=2048)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--output_dir", type=str, default="experiments/runs")
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument(
        "--gen_prompt",
        type=str,
        default="The meaning of life is",
        help="Prompt used for post-training text generation.",
    )
    parser.add_argument(
        "--gen_max_new_tokens",
        type=int,
        default=64,
        help="Number of new tokens to sample after training (set to 0 to disable).",
    )
    parser.add_argument(
        "--gen_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for post-training generation.",
    )
    parser.add_argument(
        "--gen_top_k",
        type=int,
        default=50,
        help="Top-k truncation for sampling (<=0 disables top-k).",
    )
    args = parser.parse_args()

    if args.batch_size % args.micro_batch_size != 0:
        raise ValueError("batch_size must be divisible by micro_batch_size for gradient accumulation.")

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_len

    train_dataset = load_causal_lm_dataset(
        LanguageModelingDatasetConfig(
            dataset_name=args.dataset_name,
            dataset_config_name=args.dataset_config_name,
            split="train",
            max_length=args.max_seq_len,
            num_proc=args.num_proc,
        ),
        tokenizer,
    )
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

    train_loader = make_lm_dataloader(
        train_dataset,
        micro_batch_size=args.micro_batch_size,
        shuffle=True,
        num_workers=0,
    )
    eval_loader = make_lm_dataloader(
        eval_dataset,
        micro_batch_size=args.micro_batch_size,
        shuffle=False,
        num_workers=0,
    )

    model_config = ModernLLMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        max_seq_len=args.max_seq_len,
        rope_theta=args.rope_theta,
        dropout=args.dropout,
    )
    model = ModernDecoderLM(model_config)

    training_config = TrainingConfig(
        run_name=args.run_name,
        dataset_name=args.dataset_name,
        tokenizer_name=args.tokenizer_name,
        output_dir=output_dir,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=scale_grad_accum_for_world_size(
            args.batch_size,
            args.micro_batch_size,
        ),
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        eval_every=args.eval_every,
        save_every=args.save_every,
        log_every=args.log_every,
        mixed_precision=args.mixed_precision,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)

    scheduler = None
    if training_config.warmup_steps > 0:
        def lr_lambda(step: int) -> float:
            if step < training_config.warmup_steps:
                return float(step + 1) / float(training_config.warmup_steps)
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        config=training_config,
        lr_scheduler=scheduler,
    )
    trainer.train()

    if args.gen_max_new_tokens > 0:
        top_k = args.gen_top_k if args.gen_top_k > 0 else None
        sample = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.gen_prompt,
            max_new_tokens=args.gen_max_new_tokens,
            temperature=args.gen_temperature,
            top_k=top_k,
        )
        separator = "=" * 80
        print(separator)
        print("Post-training sample generation:")
        print(sample)
        print(separator)
