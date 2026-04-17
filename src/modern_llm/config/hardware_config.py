"""Hardware configuration for multi-device training.

Supports local (RTX 3060) and high-end GPU (A100/H100) environments with
auto-detection and preset configurations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch


MixedPrecisionDtype = Literal["bf16", "fp16", "fp32"]


@dataclass
class HardwareConfig:
    """Hardware-specific training configuration.

    Attributes:
        device: Device specifier ("auto", "cuda", "cuda:0", "cpu").
        num_gpus: Number of GPUs for distributed training.
        gpu_memory_gb: GPU memory in GB (used for auto-tuning batch sizes).
        mixed_precision: AMP dtype (bf16/fp16/fp32).
        gradient_checkpointing: Trade compute for memory savings.
        is_distributed: Whether running multi-GPU via torchrun.
        world_size: Total number of processes in distributed training.
        local_rank: Local process rank (set by torchrun).
    """

    device: str = "auto"
    num_gpus: int = 2
    gpu_memory_gb: int = 12
    mixed_precision: MixedPrecisionDtype = "bf16"
    gradient_checkpointing: bool = True
    is_distributed: bool = True
    world_size: int = 2
    local_rank: int = 0

    def __post_init__(self) -> None:
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.num_gpus < 1:
            raise ValueError(f"num_gpus must be >= 1, got {self.num_gpus}")
        if self.gpu_memory_gb < 1:
            raise ValueError(f"gpu_memory_gb must be >= 1, got {self.gpu_memory_gb}")
        if self.mixed_precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError(f"mixed_precision must be bf16/fp16/fp32, got {self.mixed_precision}")
        print(self.num_gpus)

    @classmethod
    def from_env(cls) -> HardwareConfig:
        """Create config from environment variables (set by torchrun/SLURM)."""
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        is_distributed = world_size > 1

        device = "cpu"
        num_gpus = 0
        gpu_memory_gb = 0

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print("num_gpus",num_gpus)
            device = f"cuda:{local_rank}" if is_distributed else "cuda"
            if num_gpus > 0:
                props = torch.cuda.get_device_properties(local_rank if is_distributed else 0)
                gpu_memory_gb = props.total_memory // (1024**3)
        print("====================================  num_gpus72  ",num_gpus)
        return cls(
            device=device,
            num_gpus=num_gpus,
            gpu_memory_gb=gpu_memory_gb,
            is_distributed=is_distributed,
            world_size=world_size,
            local_rank=local_rank,
        )

    def get_torch_device(self) -> torch.device:
        """Return torch.device for model/tensor placement."""
        return torch.device(self.device)


@dataclass
class DataConfig:
    """Data loading and corpus configuration.

    Attributes:
        datasets: List of dataset names/paths to mix.
        tokens_target: Target number of tokens for pretraining.
        max_epochs: Maximum number of epochs over the data.
        shuffle_buffer: Number of examples to buffer for shuffling.
        num_workers: DataLoader workers.
        prefetch_factor: Batches to prefetch per worker.
    """

    datasets: list[str] = field(default_factory=lambda: ["wikitext-2-raw-v1"])
    tokens_target: int = 50_000_000  # 50M tokens default
    max_epochs: int = 10
    shuffle_buffer: int = 10_000
    num_workers: int = 16
    prefetch_factor: int = 2

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("datasets list cannot be empty")
        if self.tokens_target < 1_000:
            raise ValueError(f"tokens_target must be >= 1000, got {self.tokens_target}")
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {self.max_epochs}")


# Preset configurations for different hardware targets

LOCAL_RTX3060 = HardwareConfig(
    device="cuda",
    num_gpus=1,
    gpu_memory_gb=12,
    mixed_precision="bf16",
    gradient_checkpointing=True,
    is_distributed=False,
)

GPU_A100 = HardwareConfig(
    device="cuda",
    num_gpus=1,
    gpu_memory_gb=80,
    mixed_precision="bf16",
    gradient_checkpointing=True,
    is_distributed=False,
)

GPU_H100 = HardwareConfig(
    device="cuda",
    num_gpus=1,
    gpu_memory_gb=80,
    mixed_precision="bf16",
    gradient_checkpointing=False,  # H100 has enough memory
    is_distributed=False,
)


def get_hardware_preset(name: str) -> HardwareConfig:
    """Get a hardware preset by name.

    Pre: name is one of "local", "a100", "h100", "auto".
    """
    presets = {
        "local": LOCAL_RTX3060,
        "rtx3060": LOCAL_RTX3060,
        "a100": GPU_A100,
        "h100": GPU_H100,
        "auto": HardwareConfig.from_env(),
    }
    if name not in presets:
        raise ValueError(f"Unknown hardware preset: {name}. Choose from {list(presets.keys())}")
    return presets[name]


def get_data_preset(name: str) -> DataConfig:
    """Get a data scale preset by name.

    Pre: name is one of "small", "medium", "large", "xl".
    """
    presets = {
        "small": DataConfig(
            datasets=["wikitext-2-raw-v1"],
            tokens_target=10_000_000,
            max_epochs=3,
        ),
        "medium": DataConfig(
            datasets=["wikitext-2-raw-v1", "roneneldan/TinyStories"],
            tokens_target=100_000_000,
            max_epochs=5,
        ),
        "large": DataConfig(
            datasets=["wikitext-2-raw-v1", "roneneldan/TinyStories", "openwebtext"],
            tokens_target=1_000_000_000,
            max_epochs=1,
        ),
        "xl": DataConfig(
            datasets=["wikitext-2-raw-v1", "roneneldan/TinyStories", "openwebtext", "bookcorpus"],
            tokens_target=5_000_000_000,
            max_epochs=1,
        ),
    }
    if name not in presets:
        raise ValueError(f"Unknown data preset: {name}. Choose from {list(presets.keys())}")
    return presets[name]



