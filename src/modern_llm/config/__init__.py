"""Configuration dataclasses used across the Modern LLM project."""

from .hardware_config import (
    DataConfig,
    HardwareConfig,
    get_data_preset,
    get_hardware_preset,
    LOCAL_RTX3060,
    GPU_A100,
    GPU_H100,
)
from .model_config import ModernLLMConfig, MoEConfig
from .pipeline_config import (
    PipelineConfig,
    get_pipeline_preset,
    local_full_config,
    local_smoke_config,
    gpu_full_config,
    gpu_smoke_config,
)
from modern_llm.quantization import QuantizationConfig
from .train_config import TrainingConfig

__all__ = [
    "DataConfig",
    "HardwareConfig",
    "get_data_preset",
    "get_hardware_preset",
    "LOCAL_RTX3060",
    "GPU_A100",
    "GPU_H100",
    "ModernLLMConfig",
    "MoEConfig",
    "PipelineConfig",
    "get_pipeline_preset",
    "local_full_config",
    "local_smoke_config",
    "gpu_full_config",
    "gpu_smoke_config",
    "QuantizationConfig",
    "TrainingConfig",
]
