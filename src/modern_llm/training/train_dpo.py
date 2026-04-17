"""Direct Preference Optimization stage (Rafailov et al., 2023).

Takes an SFT checkpoint and further aligns it using pairwise preference data.
The model learns to prefer chosen responses over rejected ones.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn, Tensor
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from modern_llm.alignment.dpo_loss import dpo_loss
from modern_llm.config import ModernLLMConfig, PipelineConfig, TrainingConfig
from modern_llm.data.preference_datasets import PreferenceDatasetConfig, load_preference_dataset
from modern_llm.models.transformer import ModernDecoderLM
from modern_llm.training.distributed import (
    barrier,
    get_device,
    init_distributed,
    is_main_process,
    maybe_distributed_sampler,
    seed_everything,
    unwrap_model,
    wrap_ddp,
)
from modern_llm.utils.checkpointing import load_checkpoint, save_checkpoint
from modern_llm.utils.logging_utils import create_logger
from modern_llm.utils.paths import apply_env_defaults

apply_env_defaults()


@dataclass
class DPOConfig:
    """DPO-specific hyperparameters."""

    beta: float = 0.1  # Temperature parameter
    max_length: int = 512  # Max tokens per response
    label_smoothing: float = 0.0


class PreferenceDataset(Dataset):
    """Dataset that tokenizes preference pairs for DPO training."""

    def __init__(
        self,
        config: PreferenceDatasetConfig,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        num_examples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        raw_dataset = load_preference_dataset(config)
        if num_examples:
            raw_dataset = raw_dataset.select(range(min(num_examples, len(raw_dataset))))

        self.examples = []
        for item in raw_dataset:
            processed = self._process_item(item, config)
            if processed:
                self.examples.append(processed)

    def _process_item(self, item: dict, config: PreferenceDatasetConfig) -> Optional[dict]:
        """Tokenize a single preference pair."""
        # Try to get prompt from config field, or fall back to "prompt" key if it exists
        if config.prompt_field and config.prompt_field in item:
            prompt = item[config.prompt_field]
        elif "prompt" in item:
            prompt = item["prompt"]
        else:
            prompt = ""
        
        # Get chosen/rejected (may be processed from original dataset format)
        chosen = item.get("chosen", item.get(config.chosen_field, ""))
        rejected = item.get("rejected", item.get(config.rejected_field, ""))

        # Handle nested format (some datasets have {"content": ...})
        if isinstance(chosen, list):
            chosen = " ".join(c.get("content", str(c)) for c in chosen)
        if isinstance(rejected, list):
            rejected = " ".join(r.get("content", str(r)) for r in rejected)

        # Combine prompt with responses
        chosen_text = f"{prompt}\n{chosen}" if prompt else chosen
        rejected_text = f"{prompt}\n{rejected}" if prompt else rejected

        # Tokenize both
        chosen_tokens = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        rejected_tokens = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_tokens["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_tokens["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_tokens["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_tokens["attention_mask"].squeeze(0),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def collate_preference_batch(batch: list[dict]) -> dict:
    """Collate function for preference pairs."""
    return {
        "chosen_input_ids": torch.stack([x["chosen_input_ids"] for x in batch]),
        "chosen_attention_mask": torch.stack([x["chosen_attention_mask"] for x in batch]),
        "rejected_input_ids": torch.stack([x["rejected_input_ids"] for x in batch]),
        "rejected_attention_mask": torch.stack([x["rejected_attention_mask"] for x in batch]),
    }


def compute_sequence_logprobs(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
) -> Tensor:
    """Compute log probabilities for each sequence.

    Pre: input_ids and attention_mask are (B, L) tensors.
    Post: Returns (B,) tensor of summed log probabilities.
    """
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs["logits"] if isinstance(outputs, dict) else outputs

    # Shift for causal LM: predict next token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    # Compute log probabilities
    log_probs = torch.log_softmax(shift_logits, dim=-1)

    # Gather log probs for actual tokens
    token_log_probs = torch.gather(
        log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Mask padding and sum
    token_log_probs = token_log_probs * shift_mask.float()
    sequence_log_probs = token_log_probs.sum(dim=-1)

    return sequence_log_probs


class DPOTrainer:
    """Trainer specifically for DPO alignment."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        config: TrainingConfig,
        dpo_config: DPOConfig,
        eval_dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        # Bring up DDP first so device + seed are correct.
        init_distributed()
        if config.seed is not None:
            seed_everything(config.seed)

        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.config = config
        self.dpo_config = dpo_config
        self.eval_dataloader = eval_dataloader
        self.lr_scheduler = lr_scheduler

        self.device = get_device()
        model.to(self.device)
        # NOTE: wrap_ddp returns the bare model when WORLD_SIZE<=1.
        self.model = wrap_ddp(model)

        self.logger = create_logger(f"dpo.{config.run_name}")
        self.use_amp = config.mixed_precision in {"fp16", "bf16"} and self.device.type == "cuda"
        self.scaler = GradScaler() if config.mixed_precision == "fp16" else None

        self.global_step = 0
        self.micro_step = 0

    def train(self) -> None:
        """Run DPO training loop."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        accumulation_steps = self.config.gradient_accumulation_steps
        max_steps = self.config.max_steps

        epoch = 0
        with tqdm(
            total=max_steps,
            desc="DPO Training",
            unit="step",
            disable=not is_main_process(),
        ) as pbar:
            while self.global_step < max_steps:
                # Reshuffle DistributedSampler each epoch.
                sampler = getattr(self.train_dataloader, "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

                for batch in self.train_dataloader:
                    prev_step = self.global_step
                    loss, metrics = self._training_step(batch, accumulation_steps)

                    if self.global_step > prev_step:
                        pbar.update(self.global_step - prev_step)
                        pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{metrics['accuracy']:.2%}")

                    if self.global_step >= max_steps:
                        break

                    if (
                        self.config.log_every > 0
                        and self.global_step % self.config.log_every == 0
                        and is_main_process()
                    ):
                        self.logger.info(
                            "step=%d loss=%.4f accuracy=%.2f%% lr=%.3e",
                            self.global_step,
                            loss,
                            metrics["accuracy"] * 100,
                            self.optimizer.param_groups[0]["lr"],
                        )

                    if self.config.save_every > 0 and self.global_step % self.config.save_every == 0:
                        self._save_checkpoint()

                if self.global_step >= max_steps:
                    break
                epoch += 1

        self._save_checkpoint(suffix="final")

    def _training_step(self, batch: dict, accumulation_steps: int) -> tuple[float, dict]:
        """Execute one DPO training step."""
        batch = self._move_to_device(batch)

        autocast_dtype = torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16
        with autocast(device_type="cuda", dtype=autocast_dtype, enabled=self.use_amp):
            # Forward pass for chosen and rejected
            self.model.eval()  # No dropout for log prob computation
            with torch.no_grad():
                chosen_logprobs = compute_sequence_logprobs(
                    self.model,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                )
                rejected_logprobs = compute_sequence_logprobs(
                    self.model,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                )

            self.model.train()

            # Recompute with gradients for the chosen path
            outputs = self.model(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"],
            )
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs

            # Compute chosen log probs with gradients
            shift_logits = logits[:, :-1, :]
            shift_labels = batch["chosen_input_ids"][:, 1:]
            shift_mask = batch["chosen_attention_mask"][:, 1:]

            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_log_probs = torch.gather(
                log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            token_log_probs = token_log_probs * shift_mask.float()
            chosen_logprobs_grad = token_log_probs.sum(dim=-1)

            # DPO loss
            loss = dpo_loss(
                chosen_logprobs_grad,
                rejected_logprobs.detach(),
                beta=self.dpo_config.beta,
            )
            loss = loss / accumulation_steps

        # Backward
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self.micro_step += 1
        step_completed = self.micro_step % accumulation_steps == 0

        if step_completed:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            if self.config.max_grad_norm > 0:
                clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

        # Metrics
        with torch.no_grad():
            accuracy = (chosen_logprobs > rejected_logprobs).float().mean().item()

        return float(loss.item() * accumulation_steps), {"accuracy": accuracy}

    def _move_to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()}

    def _save_checkpoint(self, suffix: Optional[str] = None) -> None:
        # Rank 0 writes; all other ranks barrier so a subsequent load sees a
        # complete file.
        if not is_main_process():
            barrier()
            return

        tag = suffix or f"step{self.global_step}"
        path = self.config.output_dir / f"{self.config.run_name}_{tag}.pt"

        # Always save the unwrapped state_dict (no `module.` / `_orig_mod.` prefix).
        bare_model = unwrap_model(self.model)
        config_dict = None
        if hasattr(bare_model, "config"):
            config_dict = {k: v for k, v in bare_model.config.__dict__.items() if not k.startswith("_")}

        save_checkpoint(
            path,
            model_state=bare_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            step=self.global_step,
            run_name=self.config.run_name,
            config=config_dict,
        )
        self.logger.info(f"Saved checkpoint: {path}")
        barrier()


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ModernDecoderLM, ModernLLMConfig]:
    """Load model from SFT checkpoint."""
    ckpt = load_checkpoint(checkpoint_path)

    if "config" not in ckpt or ckpt["config"] is None:
        raise ValueError(f"Checkpoint {checkpoint_path} missing config")

    config = ModernLLMConfig(**ckpt["config"])
    model = ModernDecoderLM(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    return model, config


def run_dpo(
    sft_checkpoint: Path,
    train_config: TrainingConfig,
    dpo_config: DPOConfig,
    preference_config: PreferenceDatasetConfig,
    tokenizer_name: str = "Xenova/text-embedding-ada-002",
) -> Path:
    """Run DPO training on an SFT model.

    Pre: sft_checkpoint exists with valid model state.
    Post: Returns path to final DPO checkpoint.
    """
    init_distributed()
    device = get_device()

    if is_main_process():
        print(f"Loading SFT model from {sft_checkpoint}")
    model, model_config = load_model_from_checkpoint(sft_checkpoint, device)
    if is_main_process():
        print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading preference dataset: {preference_config.dataset_name}")
    dataset = PreferenceDataset(
        preference_config,
        tokenizer,
        max_length=dpo_config.max_length,
    )
    print(f"Preference pairs: {len(dataset)}")

    sampler = maybe_distributed_sampler(dataset, shuffle=True, seed=train_config.seed or 42)
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.micro_batch_size,
        sampler=sampler,
        shuffle=False if sampler is not None else True,
        collate_fn=collate_preference_batch,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=train_config.max_steps)

    trainer = DPOTrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=dataloader,
        config=train_config,
        dpo_config=dpo_config,
        lr_scheduler=scheduler,
    )

    print(f"Starting DPO for {train_config.max_steps} steps (beta={dpo_config.beta})")
    trainer.train()

    final_ckpt = train_config.output_dir / f"{train_config.run_name}_final.pt"
    print(f"DPO complete. Final checkpoint: {final_ckpt}")
    return final_ckpt


def main() -> None:
    """CLI entrypoint for DPO training."""
    parser = argparse.ArgumentParser(description="Direct Preference Optimization")
    parser.add_argument(
        "--sft-checkpoint",
        type=Path,
        required=True,
        help="Path to SFT model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Pipeline config preset or JSON path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Anthropic/hh-rlhf",
        help="Preference dataset name",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO temperature parameter",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Effective batch size",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
        help="Micro batch size",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/runs"),
        help="Output directory",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="dpo",
        help="Run name",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )

    args = parser.parse_args()

    if args.config:
        if Path(args.config).exists():
            pipeline_config = PipelineConfig.load(args.config)
            train_config = pipeline_config.get_dpo_config()
            preference_dataset = pipeline_config.dpo_dataset
            beta = pipeline_config.dpo_beta
        else:
            from modern_llm.config import get_pipeline_preset
            pipeline_config = get_pipeline_preset(args.config)
            train_config = pipeline_config.get_dpo_config()
            preference_dataset = pipeline_config.dpo_dataset
            beta = pipeline_config.dpo_beta
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
            warmup_steps=50,
        )
        preference_dataset = args.dataset
        beta = args.beta

    dpo_config = DPOConfig(
        beta=beta,
        max_length=args.max_length,
    )

    preference_config = PreferenceDatasetConfig(
        dataset_name=preference_dataset,
        split="train",
    )

    run_dpo(
        sft_checkpoint=args.sft_checkpoint,
        train_config=train_config,
        dpo_config=dpo_config,
        preference_config=preference_config,
    )


if __name__ == "__main__":
    main()
