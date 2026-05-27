"""Decoder building blocks with explicit math context.

RMSNorm follows Zhang & Sennrich (2019, Eq. 3) where activations are scaled by the
root mean square rather than the mean-centered variance used by LayerNorm.
SwiGLU implements the gated feedforward module popularized by PaLM
(Chowdhery et al., 2022, §2.3) based on the SwiGLU formulation from Shazeer (2020).
Both implementations keep the equations inline so reviewers can map code to paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (Zhang & Sennrich, 2019).

    Math:
        y = x * γ / sqrt(mean(x^2) + ε)
        where γ is a learned weight vector.

    Pre:
        - x has shape (..., hidden_dim).
        - hidden_dim matches the module configuration.
    Post:
        - returns a tensor with identical shape and bounded second moment.
    Complexity:
        - O(hidden_dim) per token because we compute the RMS over the last axis.
    Invariants:
        - Learned weights γ remain broadcast-compatible with the last dimension.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, received {hidden_dim}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, received {eps}")
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Input last dimension must match hidden_dim ({self.hidden_dim}), got {x.shape[-1]}"
            )
        # mean(x^2) is the RMS statistic from Zhang & Sennrich (2019, Eq. 3)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps)
        return normalized * self.weight


def _swish(x: Tensor) -> Tensor:
    """Swish activation σ(x) = x · sigmoid(x) (Ramachandran et al., 2018).

    Pre:
        - x is any float tensor.
    Post:
        - same shape as x with smooth non-linearity applied.
    Complexity:
        - O(1) per element.
    """

    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """SwiGLU feedforward block (Shazeer, 2020; Chowdhery et al., 2022).

    Math:
        SwiGLU(x) = W_o[(W_g x) ⊙ swish(W_v x)]
        where W_g splits into gate/value projections and ⊙ is element-wise multiply.

    Pre:
        - x.shape[-1] == in_features.
    Post:
        - returns a tensor with shape (..., out_features).
    Complexity:
        - O(in_features * hidden_features) per token due to the linear layers.
    Invariants:
        - gate/value split always halves the projected dimension (validated via chunk).
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_features <= 0 or hidden_features <= 0:
            raise ValueError("in_features and hidden_features must be positive.")
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features or in_features

        self.gate = nn.Linear(in_features, hidden_features * 2, bias=bias)
        self.proj = nn.Linear(hidden_features, self.out_features, bias=bias)

    def forward(self, x: Tensor, active_hidden_size: int | None = None) -> Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Input last dimension mismatch: expected {self.in_features}, got {x.shape[-1]}"
            )
        if active_hidden_size is None or active_hidden_size == self.hidden_features:
            gate_out, value = self.gate(x).chunk(2, dim=-1)  # gate/value split per GLU family.
            activated = _swish(gate_out)
            gated = activated * value
            return self.proj(gated)

        if (
            isinstance(active_hidden_size, bool)
            or not isinstance(active_hidden_size, int)
            or active_hidden_size <= 0
        ):
            raise ValueError("active_hidden_size must be a positive integer")
        if active_hidden_size > self.hidden_features:
            raise ValueError(
                f"active_hidden_size must be <= hidden_features "
                f"({active_hidden_size} > {self.hidden_features})"
            )

        gate_bias = self.gate.bias[:active_hidden_size] if self.gate.bias is not None else None
        value_start = self.hidden_features
        value_end = value_start + active_hidden_size
        value_bias = self.gate.bias[value_start:value_end] if self.gate.bias is not None else None

        gate_out = F.linear(x, self.gate.weight[:active_hidden_size], gate_bias)
        value = F.linear(x, self.gate.weight[value_start:value_end], value_bias)
        activated = _swish(gate_out)
        gated = activated * value
        return F.linear(gated, self.proj.weight[:, :active_hidden_size], self.proj.bias)
