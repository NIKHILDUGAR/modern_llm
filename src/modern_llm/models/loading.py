"""Shared checkpoint loading helpers for dense and opt-in quantized models."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from modern_llm.config.model_config import ModernLLMConfig, MoEConfig
from modern_llm.models.transformer import ModernDecoderLM
from modern_llm.quantization import QuantizationConfig, prepare_model_for_quantization
from modern_llm.utils.checkpointing import load_checkpoint


_VALID_MODEL_CONFIG_KEYS = {field.name for field in fields(ModernLLMConfig)}


def normalize_model_config_dict(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Map older checkpoint config aliases onto the current model dataclass."""

    cfg = dict(config_dict)
    if "num_layers" in cfg and "n_layers" not in cfg:
        cfg["n_layers"] = cfg.pop("num_layers")
    if "max_position_embeddings" in cfg and "max_seq_len" not in cfg:
        cfg["max_seq_len"] = cfg.pop("max_position_embeddings")
    if isinstance(cfg.get("moe_config"), dict):
        cfg["moe_config"] = MoEConfig(**cfg["moe_config"])
    return {key: value for key, value in cfg.items() if key in _VALID_MODEL_CONFIG_KEYS}


def resolve_quantization_config(
    payload: dict[str, Any],
    override: QuantizationConfig | None = None,
) -> QuantizationConfig | None:
    if override is not None and override.enabled:
        return override
    direct = payload.get("quantization")
    if isinstance(direct, dict):
        inner = direct.get("config") if isinstance(direct.get("config"), dict) else direct
        return QuantizationConfig.from_dict(inner)
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        meta_quant = metadata.get("quantization")
        if isinstance(meta_quant, dict):
            inner = meta_quant.get("config") if isinstance(meta_quant.get("config"), dict) else meta_quant
            return QuantizationConfig.from_dict(inner)
    return None


def build_model_from_checkpoint_payload(
    payload: dict[str, Any],
    *,
    device: torch.device | str | None = None,
    override_quantization: QuantizationConfig | None = None,
    strict: bool = True,
    eval_mode: bool = False,
) -> tuple[ModernDecoderLM, ModernLLMConfig, QuantizationConfig | None]:
    """Construct a model from a checkpoint payload, reapplying quantization wrappers if needed."""

    if "config" not in payload or payload["config"] is None:
        raise ValueError("Checkpoint payload is missing a 'config' entry.")

    config = ModernLLMConfig(**normalize_model_config_dict(payload["config"]))
    model = ModernDecoderLM(config)

    quantization = resolve_quantization_config(payload, override_quantization)
    if quantization is not None and quantization.enabled:
        prepare_model_for_quantization(model, quantization)

    state_dict = payload.get("model_state", payload.get("model_state_dict", payload.get("model", payload)))
    model.load_state_dict(state_dict, strict=strict)

    if device is not None:
        model.to(device)
    if eval_mode:
        model.eval()
    return model, config, quantization


def load_model_from_checkpoint(
    checkpoint_path: Path | str,
    *,
    device: torch.device | str | None = None,
    override_quantization: QuantizationConfig | None = None,
    strict: bool = True,
    eval_mode: bool = False,
) -> tuple[ModernDecoderLM, ModernLLMConfig, QuantizationConfig | None, dict[str, Any]]:
    payload = load_checkpoint(Path(checkpoint_path))
    model, config, quantization = build_model_from_checkpoint_payload(
        payload,
        device=device,
        override_quantization=override_quantization,
        strict=strict,
        eval_mode=eval_mode,
    )
    return model, config, quantization, payload
