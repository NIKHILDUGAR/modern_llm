"""Verifier training entrypoint (math/QA correctness scorer).

Trains a small encoder model to predict whether a given solution to a
math problem is correct or incorrect. Uses GSM8K with synthetic negatives.
"""

from __future__ import annotations

import argparse
import random
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

from modern_llm.config import PipelineConfig, TrainingConfig
from modern_llm.models.verifier import VerifierConfig, VerifierModel
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
from modern_llm.utils.checkpointing import save_checkpoint
from modern_llm.utils.logging_utils import create_logger
from modern_llm.utils.paths import apply_env_defaults, cache_dir_for_datasets

apply_env_defaults()


@dataclass
class VerifierDatasetConfig:
    """Configuration for verifier training data."""

    dataset_name: str = "gsm8k"
    split: str = "train"
    max_length: int = 512
    num_examples: Optional[int] = None
    negative_ratio: float = 1.0  # How many negatives per positive


class VerifierDataset(Dataset):
    """Dataset for verifier training with correct/incorrect labels.

    Loads GSM8K and creates training examples:
    - Positive: problem + correct answer
    - Negative: problem + perturbed/wrong answer
    """

    def __init__(
        self,
        config: VerifierDatasetConfig,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.max_length = config.max_length

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.examples = self._load_and_process()

    def _load_and_process(self) -> list[dict]:
        """Load GSM8K and create positive/negative examples."""
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("datasets package required") from e

        raw = load_dataset(self.config.dataset_name, "main", split=self.config.split, cache_dir=cache_dir_for_datasets())

        if self.config.num_examples:
            raw = raw.select(range(min(self.config.num_examples, len(raw))))

        examples = []
        all_answers = [self._extract_answer(item["answer"]) for item in raw]

        for idx, item in enumerate(raw):
            question = item["question"]
            correct_answer = self._extract_answer(item["answer"])

            # Positive example
            pos_text = self._format_qa(question, correct_answer)
            pos_tokens = self._tokenize(pos_text)
            if pos_tokens:
                pos_tokens["labels"] = torch.tensor(1, dtype=torch.long)
                examples.append(pos_tokens)

            # Negative examples (wrong answers)
            num_negatives = int(self.config.negative_ratio)
            for _ in range(num_negatives):
                wrong_answer = self._generate_wrong_answer(correct_answer, all_answers, idx)
                neg_text = self._format_qa(question, wrong_answer)
                neg_tokens = self._tokenize(neg_text)
                if neg_tokens:
                    neg_tokens["labels"] = torch.tensor(0, dtype=torch.long)
                    examples.append(neg_tokens)

        random.shuffle(examples)
        return examples

    def _extract_answer(self, answer_text: str) -> str:
        """Extract the final numeric answer from GSM8K format.

        GSM8K answers end with #### followed by the number.
        """
        if "####" in answer_text:
            return answer_text.split("####")[-1].strip()
        return answer_text.strip().split()[-1]

    def _format_qa(self, question: str, answer: str) -> str:
        """Format question and answer for verifier input."""
        return f"Question: {question}\n\nAnswer: {answer}"

    def _generate_wrong_answer(
        self,
        correct: str,
        all_answers: list[str],
        current_idx: int,
    ) -> str:
        """Generate a plausible but wrong answer.

        Strategies:
        1. Random answer from other problems
        2. Perturbed correct answer (off by some amount)
        """
        strategy = random.choice(["other", "perturb"])

        if strategy == "other":
            # Use answer from a different problem
            candidates = [a for i, a in enumerate(all_answers) if i != current_idx and a != correct]
            if candidates:
                return random.choice(candidates)

        # Perturb the correct answer
        try:
            num = float(correct.replace(",", "").replace("$", ""))
            perturbation = random.choice([1, -1, 10, -10, 0.5, 2])
            wrong_num = num + perturbation if random.random() > 0.5 else num * perturbation
            return str(int(wrong_num) if wrong_num == int(wrong_num) else round(wrong_num, 2))
        except ValueError:
            # Not a number, just return a random other answer
            candidates = [a for i, a in enumerate(all_answers) if i != current_idx]
            return random.choice(candidates) if candidates else "0"

    def _tokenize(self, text: str) -> Optional[dict]:
        """Tokenize text for verifier input."""
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


def collate_verifier_batch(batch: list[dict]) -> dict:
    """Collate function for verifier training."""
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
    }


class VerifierTrainer:
    """Trainer for the verifier model."""

    def __init__(
        self,
        model: VerifierModel,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        config: TrainingConfig,
        eval_dataloader: Optional[DataLoader] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        init_distributed()
        if config.seed is not None:
            seed_everything(config.seed)

        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.config = config
        self.eval_dataloader = eval_dataloader
        self.lr_scheduler = lr_scheduler

        self.device = get_device()
        model.to(self.device)
        self.model = wrap_ddp(model)

        self.logger = create_logger(f"verifier.{config.run_name}")
        self.use_amp = config.mixed_precision in {"fp16", "bf16"} and self.device.type == "cuda"
        self.scaler = GradScaler() if config.mixed_precision == "fp16" else None

        self.global_step = 0
        self.micro_step = 0

    def train(self) -> None:
        """Run verifier training loop."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        accumulation_steps = self.config.gradient_accumulation_steps
        max_steps = self.config.max_steps

        epoch = 0
        with tqdm(
            total=max_steps,
            desc="Verifier Training",
            unit="step",
            disable=not is_main_process(),
        ) as pbar:
            while self.global_step < max_steps:
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

                    if (
                        self.config.eval_every > 0
                        and self.eval_dataloader
                        and self.global_step % self.config.eval_every == 0
                    ):
                        eval_metrics = self.evaluate()
                        if is_main_process():
                            self.logger.info(
                                "eval step=%d accuracy=%.2f%%",
                                self.global_step,
                                eval_metrics["accuracy"] * 100,
                            )

                    if self.config.save_every > 0 and self.global_step % self.config.save_every == 0:
                        self._save_checkpoint()

                if self.global_step >= max_steps:
                    break
                epoch += 1

        self._save_checkpoint(suffix="final")

    def _training_step(self, batch: dict, accumulation_steps: int) -> tuple[float, dict]:
        """Execute one training step."""
        batch = self._move_to_device(batch)

        autocast_dtype = torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16
        with autocast(device_type="cuda", dtype=autocast_dtype, enabled=self.use_amp):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs["loss"] / accumulation_steps

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

        # Compute accuracy
        with torch.no_grad():
            preds = outputs["logits"].argmax(dim=-1)
            accuracy = (preds == batch["labels"]).float().mean().item()

        return float(loss.item() * accumulation_steps), {"accuracy": accuracy}

    def evaluate(self) -> dict[str, float]:
        """Evaluate on held-out data."""
        if not self.eval_dataloader:
            return {"accuracy": 0.0, "loss": float("nan")}

        self.model.eval()
        total_correct = 0
        total_examples = 0
        total_loss = 0.0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = self._move_to_device(batch)
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                preds = outputs["logits"].argmax(dim=-1)
                total_correct += (preds == batch["labels"]).sum().item()
                total_examples += batch["labels"].size(0)
                total_loss += outputs["loss"].item()

        self.model.train()
        return {
            "accuracy": total_correct / max(1, total_examples),
            "loss": total_loss / max(1, len(self.eval_dataloader)),
        }

    def _move_to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) if isinstance(v, Tensor) else v for k, v in batch.items()}

    def _save_checkpoint(self, suffix: Optional[str] = None) -> None:
        if not is_main_process():
            barrier()
            return

        tag = suffix or f"step{self.global_step}"
        path = self.config.output_dir / f"{self.config.run_name}_{tag}.pt"

        bare_model = unwrap_model(self.model)
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


def run_verifier_training(
    train_config: TrainingConfig,
    verifier_config: VerifierConfig,
    dataset_config: VerifierDatasetConfig,
    tokenizer_name: str = "Xenova/text-embedding-ada-002",
    eval_split: Optional[str] = None,
) -> Path:
    """Train the verifier model.

    Pre: GSM8K dataset is accessible.
    Post: Returns path to final verifier checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Update vocab size in config
    verifier_config.vocab_size = tokenizer.vocab_size

    model = VerifierModel(verifier_config)
    print(f"Verifier: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    print(f"Loading verifier dataset: {dataset_config.dataset_name}")
    train_dataset = VerifierDataset(dataset_config, tokenizer)
    print(f"Training examples: {len(train_dataset)}")

    train_sampler = maybe_distributed_sampler(train_dataset, shuffle=True, seed=train_config.seed or 42)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_config.micro_batch_size,
        sampler=train_sampler,
        shuffle=False if train_sampler is not None else True,
        collate_fn=collate_verifier_batch,
        num_workers=0,
        pin_memory=True,
    )

    eval_dataloader = None
    if eval_split:
        eval_config = VerifierDatasetConfig(
            dataset_name=dataset_config.dataset_name,
            split=eval_split,
            max_length=dataset_config.max_length,
            num_examples=min(500, dataset_config.num_examples or 500),
        )
        eval_dataset = VerifierDataset(eval_config, tokenizer)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=train_config.micro_batch_size,
            shuffle=False,
            collate_fn=collate_verifier_batch,
        )
        print(f"Eval examples: {len(eval_dataset)}")

    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(0.9, 0.99),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=train_config.max_steps)

    trainer = VerifierTrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        config=train_config,
        eval_dataloader=eval_dataloader,
        lr_scheduler=scheduler,
    )

    print(f"Starting verifier training for {train_config.max_steps} steps")
    trainer.train()

    final_ckpt = train_config.output_dir / f"{train_config.run_name}_final.pt"
    print(f"Verifier training complete. Final checkpoint: {final_ckpt}")
    return final_ckpt


def main() -> None:
    """CLI entrypoint for verifier training."""
    parser = argparse.ArgumentParser(description="Train Verifier Model")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Pipeline config preset or JSON path",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=3000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
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
        default=4,
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
        default="verifier",
        help="Run name",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Verifier hidden dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of encoder layers",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default="test",
        help="Evaluation split",
    )

    args = parser.parse_args()

    if args.config:
        if Path(args.config).exists():
            pipeline_config = PipelineConfig.load(args.config)
            train_config = pipeline_config.get_verifier_config()
        else:
            from modern_llm.config import get_pipeline_preset
            pipeline_config = get_pipeline_preset(args.config)
            train_config = pipeline_config.get_verifier_config()
    else:
        train_config = TrainingConfig(
            run_name=args.run_name,
            dataset_name="gsm8k",
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

    verifier_config = VerifierConfig(
        vocab_size=50257,  # Will be updated from tokenizer
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_heads=8,
        max_position_embeddings=args.max_length,
    )

    dataset_config = VerifierDatasetConfig(
        dataset_name="gsm8k",
        split="train",
        max_length=args.max_length,
    )

    run_verifier_training(
        train_config=train_config,
        verifier_config=verifier_config,
        dataset_config=dataset_config,
        eval_split=args.eval_split,
    )


if __name__ == "__main__":
    main()
