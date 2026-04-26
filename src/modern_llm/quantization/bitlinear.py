"""Experimental low-bit linear layers used by the opt-in quantization path."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _LowBitLinearBase(nn.Module):
    """Common state for linear layers that switch from dense to quantized forward."""

    def __init__(self, linear: nn.Linear, *, quant_start_step: int) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        self.quant_start_step = quant_start_step
        self.register_buffer("_quant_step", torch.zeros((), dtype=torch.long), persistent=False)

    def set_quantization_step(self, step: int) -> None:
        self._quant_step.fill_(int(step))

    def quantization_active(self) -> bool:
        return int(self._quant_step.item()) >= self.quant_start_step

    @staticmethod
    def _ste_dequantized(reference: Tensor, quantized: Tensor) -> Tensor:
        return reference + (quantized - reference).detach()

    def export_quantized_state(self) -> dict[str, Any]:
        raise NotImplementedError


class BitLinear(_LowBitLinearBase):
    """BitNet-style ternary-weight linear layer with dense warmup."""

    def __init__(self, linear: nn.Linear, *, quant_start_step: int, scale_method: str = "median") -> None:
        super().__init__(linear, quant_start_step=quant_start_step)
        self.scale_method = scale_method

    def _weight_scale(self) -> Tensor:
        abs_weight = self.weight.detach().abs()
        if self.scale_method == "mean":
            scale = abs_weight.mean(dim=1, keepdim=True)
        else:
            scale = abs_weight.median(dim=1, keepdim=True).values
        return scale.clamp_min(1e-8)

    def _quantized_weight(self) -> tuple[Tensor, Tensor, Tensor]:
        scale = self._weight_scale()
        normalized = self.weight / scale
        ternary = torch.where(
            normalized > 0.5,
            torch.ones_like(normalized),
            torch.where(normalized < -0.5, -torch.ones_like(normalized), torch.zeros_like(normalized)),
        )
        dequantized = self._ste_dequantized(self.weight, ternary * scale)
        return dequantized, ternary.to(dtype=torch.int8), scale

    def forward(self, x: Tensor) -> Tensor:
        if not self.quantization_active():
            return F.linear(x, self.weight, self.bias)
        qweight, _, _ = self._quantized_weight()
        return F.linear(x, qweight, self.bias)

    def export_quantized_state(self) -> dict[str, Any]:
        _, ternary, scale = self._quantized_weight()
        payload: dict[str, Any] = {
            "kind": "bitlinear",
            "qweight": ternary.cpu(),
            "scale": scale.cpu(),
            "quant_start_step": self.quant_start_step,
            "scale_method": self.scale_method,
        }
        if self.bias is not None:
            payload["bias"] = self.bias.detach().cpu()
        return payload


class Int8DynActInt4WeightQATLinear(_LowBitLinearBase):
    """Small self-contained QAT layer matching an 8da4w-style training scheme."""

    def __init__(self, linear: nn.Linear, *, quant_start_step: int, scale_method: str = "mean") -> None:
        super().__init__(linear, quant_start_step=quant_start_step)
        self.scale_method = scale_method

    def _fake_quantize_activations(self, x: Tensor) -> Tensor:
        scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 127.0
        quantized = torch.clamp(torch.round(x / scale), -128, 127) * scale
        return self._ste_dequantized(x, quantized)

    def _weight_scale(self) -> Tensor:
        abs_weight = self.weight.detach().abs()
        if self.scale_method == "median":
            base = abs_weight.median(dim=1, keepdim=True).values
        else:
            base = abs_weight.amax(dim=1, keepdim=True)
        return (base / 7.0).clamp_min(1e-8)

    def _quantized_weight(self) -> tuple[Tensor, Tensor, Tensor]:
        scale = self._weight_scale()
        qweight = torch.clamp(torch.round(self.weight / scale), -8, 7)
        dequantized = self._ste_dequantized(self.weight, qweight * scale)
        return dequantized, qweight.to(dtype=torch.int8), scale

    def forward(self, x: Tensor) -> Tensor:
        if not self.quantization_active():
            return F.linear(x, self.weight, self.bias)
        qx = self._fake_quantize_activations(x)
        qweight, _, _ = self._quantized_weight()
        return F.linear(qx, qweight, self.bias)

    def export_quantized_state(self) -> dict[str, Any]:
        _, qweight, scale = self._quantized_weight()
        payload: dict[str, Any] = {
            "kind": "qat_8da4w",
            "qweight": qweight.cpu(),
            "scale": scale.cpu(),
            "quant_start_step": self.quant_start_step,
            "scale_method": self.scale_method,
        }
        if self.bias is not None:
            payload["bias"] = self.bias.detach().cpu()
        return payload
