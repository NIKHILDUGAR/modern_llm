"""Supervised fine-tuning stage (instruction tuning per Ouyang et al., 2022).

Takes a pretrained checkpoint and finetunes on instruction-following data
using the standard cross-entropy loss with response-only masking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer

from modern_llm.config import ModernLLMConfig, PipelineConfig, TrainingConfig
from modern_llm.data.instruction_datasets import (
    InstructionDatasetConfig,
    create_instruction_dataloader,
    load_instruction_dataset,
)
from modern_llm.models.transformer import ModernDecoderLM
from modern_llm.training.distributed import get_device, init_distributed, is_main_process
from modern_llm.training.trainer_base import Trainer
from modern_llm.utils.checkpointing import load_checkpoint
from modern_llm.utils.paths import apply_env_defaults

apply_env_defaults()


def load_pretrained_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ModernDecoderLM, ModernLLMConfig]:
    """Load pretrained model from checkpoint.

    Pre: checkpoint_path exists and contains model_state and config.
    Post: Returns model and config, model is on specified device.
    """
    ckpt = load_checkpoint(checkpoint_path)

    if "config" not in ckpt or ckpt["config"] is None:
        raise ValueError(f"Checkpoint {checkpoint_path} missing 'config' key")

    config = ModernLLMConfig(**ckpt["config"])
    model = ModernDecoderLM(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    return model, config


def run_sft(
    pretrain_checkpoint: Path,
    train_config: TrainingConfig,
    dataset_config: InstructionDatasetConfig,
    tokenizer_name: str="Xenova/text-embedding-ada-002",
    eval_split: Optional[str] = None,
    train_dataset: Optional[torch.utils.data.Dataset] = None,
) -> Path:
    """Run supervised fine-tuning on pretrained model.

    Pre:
        - pretrain_checkpoint exists with valid model state
        - dataset specified in dataset_config is accessible,
          OR a pre-built `train_dataset` is supplied (e.g. an
          SFTMixtureDataset from build_sft_mixture).
    Post:
        - Returns path to final SFT checkpoint

    When `train_dataset` is provided, `dataset_config.dataset_name` is
    ignored for training data loading (it is still used for the eval
    split if `eval_split` is provided).
    """
    # Initialize DDP early so get_device() returns the correct per-rank device.
    init_distributed()
    device = get_device()

    # Load model
    if is_main_process():
        print(f"Loading pretrained model from {pretrain_checkpoint}")
    model, model_config = load_pretrained_model(pretrain_checkpoint, device)
    if is_main_process():
        print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    if train_dataset is None:
        print(f"Loading instruction dataset: {dataset_config.dataset_name}")
        train_dataset = load_instruction_dataset(dataset_config, tokenizer)
    else:
        print(f"Using pre-built train_dataset ({type(train_dataset).__name__})")
    print(f"Training examples: {len(train_dataset)}")

    train_dataloader = create_instruction_dataloader(
        train_dataset,
        batch_size=train_config.micro_batch_size,
        shuffle=True,
    )

    eval_dataloader = None
    if eval_split:
        eval_config = InstructionDatasetConfig(
            dataset_name=dataset_config.dataset_name,
            max_length=dataset_config.max_length,
            split=eval_split,
            num_examples=min(500, dataset_config.num_examples or 500),
        )
        eval_dataset = load_instruction_dataset(eval_config, tokenizer)
        eval_dataloader = create_instruction_dataloader(
            eval_dataset,
            batch_size=train_config.micro_batch_size,
            shuffle=False,
        )
        print(f"Eval examples: {len(eval_dataset)}")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=train_config.max_steps)

    # Train
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        config=train_config,
        eval_dataloader=eval_dataloader,
        lr_scheduler=scheduler,
    )

    print(f"Starting SFT for {train_config.max_steps} steps")
    trainer.train()

    # Return final checkpoint path
    final_ckpt = train_config.output_dir / f"{train_config.run_name}_final.pt"
    print(f"SFT complete. Final checkpoint: {final_ckpt}")
    return final_ckpt


def main() -> None:
    """CLI entrypoint for SFT training."""
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning")
    parser.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        required=True,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Pipeline config preset (local, gpu, etc.) or path to JSON",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tatsu-lab/alpaca",
        help="Instruction dataset name on HF Hub",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Effective batch size",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=2,
        help="Micro batch size per GPU",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs"),
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="sft",
        help="Run name for logging and checkpoints",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default=None,
        help="Eval split name (e.g., 'test')",
    )

    args = parser.parse_args()

    # Build configs
    if args.config:
        if Path(args.config).exists():
            pipeline_config = PipelineConfig.load(args.config)
            train_config = pipeline_config.get_sft_config()
            dataset_name = pipeline_config.sft_dataset
            max_length = pipeline_config.max_seq_len
        else:
            from modern_llm.config import get_pipeline_preset
            pipeline_config = get_pipeline_preset(args.config)
            train_config = pipeline_config.get_sft_config()
            dataset_name = pipeline_config.sft_dataset
            max_length = pipeline_config.max_seq_len
    else:
        train_config = TrainingConfig(
            run_name=args.run_name,
            dataset_name=args.dataset,
            tokenizer_name="Xenova/text-embedding-ada-002",
            output_dir=args.output_dir / args.run_name,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.batch_size // args.micro_batch_size,
            learning_rate=args.lr,
            max_steps=args.max_steps,
            warmup_steps=100,
            weight_decay=0.01,
        )
        dataset_name = args.dataset
        max_length = args.max_length

    dataset_config = InstructionDatasetConfig(
        dataset_name=dataset_name,
        max_length=max_length,
        split="train",
    )

    run_sft(
        pretrain_checkpoint=args.pretrain_checkpoint,
        train_config=train_config,
        dataset_config=dataset_config,
        eval_split=args.eval_split,
    )


if __name__ == "__main__":
    main()
