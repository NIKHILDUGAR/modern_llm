"""Opt-in Gated DeltaNet sequence mixer.

This module intentionally lives beside, not inside, the dense attention path.
The implementation is a dependency-free PyTorch reference for experiments and
smoke tests; production-speed long-context training should eventually swap the
same interface to optimized FLA kernels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from modern_llm.models.layers import RMSNorm


@dataclass
class GatedDeltaNetConfig:
    """Configuration for the lightweight local Gated DeltaNet reference."""

    d_model: int
    num_heads: int
    conv_kernel: int = 4
    dropout: float = 0.0
    rmsnorm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.num_heads <= 0 or self.d_model % self.num_heads != 0:
            raise ValueError("num_heads must be positive and divide d_model.")
        if self.conv_kernel <= 0:
            raise ValueError("conv_kernel must be positive.")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")


class ShortDepthwiseConv1d(nn.Module):
    """Causal depthwise convolution used before the recurrent delta update."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.channels:
            raise ValueError(f"Expected last dim {self.channels}, got {x.shape[-1]}")
        x_t = x.transpose(1, 2)
        y = F.conv1d(x_t, self.weight, padding=self.kernel_size - 1, groups=self.channels)
        return y[..., : x.size(1)].transpose(1, 2)


class GatedDeltaNet(nn.Module):
    """Gated delta-rule sequence mixer with the same `(B, S, D) -> (B, S, D)` API as attention.

    The recurrent state stores a per-head associative memory. At each position,
    a learned retention gate forgets stale memory and a learned update gate
    writes the delta between the current value and the memory's prediction.
    """

    def __init__(self, config: GatedDeltaNetConfig) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.d_model // config.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.g_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.a_proj = nn.Linear(config.d_model, config.num_heads, bias=False)
        self.b_proj = nn.Linear(config.d_model, config.num_heads, bias=False)
        self.q_conv1d = ShortDepthwiseConv1d(config.d_model, config.conv_kernel)
        self.k_conv1d = ShortDepthwiseConv1d(config.d_model, config.conv_kernel)
        self.v_conv1d = ShortDepthwiseConv1d(config.d_model, config.conv_kernel)
        self.out_norm = RMSNorm(config.d_model, eps=config.rmsnorm_eps)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        residual_dtype = hidden_states.dtype

        q = self._shape_heads(F.silu(self.q_conv1d(self.q_proj(hidden_states))))
        k = self._shape_heads(F.silu(self.k_conv1d(self.k_proj(hidden_states))))
        v = self._shape_heads(F.silu(self.v_conv1d(self.v_proj(hidden_states))))
        q = F.normalize(q.float(), p=2.0, dim=-1) * self.scale
        k = F.normalize(k.float(), p=2.0, dim=-1)
        v = v.float()

        retention = torch.sigmoid(self.a_proj(hidden_states)).transpose(1, 2).float()
        update = torch.sigmoid(self.b_proj(hidden_states)).transpose(1, 2).float()

        if padding_mask is not None:
            if padding_mask.shape != hidden_states.shape[:2]:
                raise ValueError("padding_mask must have shape (batch, seq_len)")
            valid = padding_mask.to(dtype=torch.float32).unsqueeze(1)
            retention = torch.where(valid > 0, retention, torch.ones_like(retention))
            update = update * valid
            q = q * valid.unsqueeze(-1)
            k = k * valid.unsqueeze(-1)
            v = v * valid.unsqueeze(-1)

        state = torch.zeros(
            batch_size,
            self.config.num_heads,
            self.head_dim,
            self.head_dim,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        outputs = []
        for t in range(seq_len):
            kt = k[:, :, t]
            vt = v[:, :, t]
            predicted = torch.einsum("bhkv,bhk->bhv", state, kt)
            delta = (vt - predicted) * update[:, :, t].unsqueeze(-1)
            state = state * retention[:, :, t].unsqueeze(-1).unsqueeze(-1)
            state = state + torch.einsum("bhk,bhv->bhkv", kt, delta)
            yt = torch.einsum("bhk,bhkv->bhv", q[:, :, t], state)
            outputs.append(yt)

        y = torch.stack(outputs, dim=2).transpose(1, 2).contiguous()
        y = y.view(batch_size, seq_len, self.config.d_model)
        gate = F.silu(self.g_proj(hidden_states).float())
        y = self.out_norm(y) * gate
        y = self.dropout(self.out_proj(y.to(dtype=residual_dtype)))

        if padding_mask is not None:
            y = y * padding_mask.to(dtype=y.dtype).unsqueeze(-1)
        return y

    def _shape_heads(self, tensor: Tensor) -> Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.config.num_heads, self.head_dim).transpose(1, 2)
