"""Training configuration dataclass with validation.

Parameters capture common training heuristics from GPT-style scaling (Kaplan et
al., 2020) such as gradient accumulation, mixed precision, and logging cadence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


MixedPrecisionDtype = Literal["bf16", "fp16", "fp32"]


@dataclass
class TrainingConfig:
    """Hyperparameters and bookkeeping for a single training or finetuning run."""

    run_name: str
    dataset_name: str
    tokenizer_name: str
    output_dir: Path
    batch_size: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_steps: int
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    eval_every: int = 500
    save_every: int = 500
    log_every: int = 50
    seed: Optional[int] = 42
    mixed_precision: MixedPrecisionDtype = "bf16"
    gradient_checkpointing: bool = True
    compile_model: bool = True  # Significant speedup on modern GPUs

    def __post_init__(self) -> None:
        self._validate_positive_int("batch_size", self.batch_size)
        self._validate_positive_int("micro_batch_size", self.micro_batch_size)
        self._validate_positive_int("gradient_accumulation_steps", self.gradient_accumulation_steps)
        self._validate_positive_int("max_steps", self.max_steps)
        self._validate_non_negative_int("warmup_steps", self.warmup_steps)
        self._validate_non_negative_int("eval_every", self.eval_every)
        self._validate_non_negative_int("save_every", self.save_every)
        self._validate_non_negative_int("log_every", self.log_every)
        if self.micro_batch_size > self.batch_size:
            raise ValueError(
                f"micro_batch_size ({self.micro_batch_size}) cannot exceed batch_size ({self.batch_size})"
            )
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, received {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, received {self.weight_decay}")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0,1], received {self.min_lr_ratio}")
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm must be positive, received {self.max_grad_norm}")
        if self.mixed_precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError(f"mixed_precision must be one of bf16/fp16/fp32, received {self.mixed_precision}")
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be positive, received {value}")

    @staticmethod
    def _validate_non_negative_int(name: str, value: int) -> None:
        if value < 0:
            raise ValueError(f"{name} must be non-negative, received {value}")

