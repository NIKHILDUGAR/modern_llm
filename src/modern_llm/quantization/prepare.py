"""Utilities to opt-in a dense ModernDecoderLM into low-bit training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from modern_llm.quantization.bitlinear import BitLinear, Int8DynActInt4WeightQATLinear
from modern_llm.quantization.config import QuantizationConfig


@dataclass
class QuantizationPreparationSummary:
    enabled: bool
    mode: str
    replaced_modules: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "num_replaced_modules": len(self.replaced_modules),
            "replaced_modules": list(self.replaced_modules),
        }


def _replace_linear(linear: nn.Linear, config: QuantizationConfig) -> nn.Module:
    if config.mode == "bitnet_b1_58":
        return BitLinear(
            linear,
            quant_start_step=config.quant_start_step,
            scale_method=config.scale_method,
        )
    if config.mode == "qat_8da4w":
        return Int8DynActInt4WeightQATLinear(
            linear,
            quant_start_step=config.quant_start_step,
            scale_method=config.scale_method,
        )
    return linear


def prepare_model_for_quantization(model: nn.Module, config: QuantizationConfig | None) -> QuantizationPreparationSummary:
    """Replace eligible dense linear layers with opt-in low-bit training modules."""

    if config is None or not config.enabled:
        summary = QuantizationPreparationSummary(enabled=False, mode="none", replaced_modules=())
        setattr(model, "_quantization_config", None)
        setattr(model, "_quantization_summary", summary.to_dict())
        return summary
    model_config = getattr(model, "config", None)
    if bool(getattr(model_config, "use_matformer", False)):
        raise ValueError("MatFormer is not compatible with quantization in this patch")

    if not hasattr(model, "iter_quantizable_linear_layers"):
        raise TypeError("Model must define iter_quantizable_linear_layers() for quantization preparation.")

    total_blocks = len(getattr(model, "blocks", []))
    replaced: list[str] = []
    for ref in model.iter_quantizable_linear_layers():
        if ref.group == "lm_head" and not config.quantize_lm_head:
            continue
        if ref.group == "embeddings" and not config.quantize_embeddings:
            continue
        if not config.wants_target(ref.group):
            continue
        if ref.block_index is not None:
            if ref.block_index < config.skip_first_blocks:
                continue
            if ref.block_index >= max(0, total_blocks - config.skip_last_blocks):
                continue
        replacement = _replace_linear(ref.module, config)
        if replacement is ref.module:
            continue
        setattr(ref.parent, ref.attr_name, replacement)
        replaced.append(ref.module_path)

    summary = QuantizationPreparationSummary(
        enabled=True,
        mode=config.mode,
        replaced_modules=tuple(replaced),
    )
    setattr(model, "_quantization_config", config)
    setattr(model, "_quantization_summary", summary.to_dict())
    return summary


def set_quantization_step(model: nn.Module, step: int) -> None:
    """Broadcast the trainer step to any active low-bit modules."""

    for module in model.modules():
        setter = getattr(module, "set_quantization_step", None)
        if callable(setter):
            setter(step)


def get_quantization_config(model: nn.Module) -> QuantizationConfig | None:
    return getattr(model, "_quantization_config", None)


def get_quantization_summary(model: nn.Module) -> dict[str, Any] | None:
    return getattr(model, "_quantization_summary", None)


def get_quantization_payload(model: nn.Module) -> dict[str, Any] | None:
    config = get_quantization_config(model)
    if config is None or not config.enabled:
        return None
    summary = get_quantization_summary(model) or {}
    return {
        "config": config.to_dict(),
        "summary": summary,
    }


def export_quantized_artifact(model: nn.Module, path: Path, *, metadata: dict[str, Any] | None = None) -> Path | None:
    """Persist an auxiliary low-bit artifact without replacing the dense checkpoint."""

    config = get_quantization_config(model)
    if config is None or not config.enabled or not config.export_quantized_checkpoint:
        return None

    artifact = {
        "quantization": get_quantization_payload(model),
        "modules": {},
        "metadata": metadata or {},
    }
    for name, module in model.named_modules():
        exporter = getattr(module, "export_quantized_state", None)
        if callable(exporter):
            artifact["modules"][name] = exporter()

    if not artifact["modules"]:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)
    return path
