"""LFM2-style dense hybrid building blocks."""

from __future__ import annotations

from torch import Tensor, nn
import torch.nn.functional as F


class LFM2ShortConv(nn.Module):
    """Gated causal depthwise short convolution used in LFM2-style hybrid blocks."""

    def __init__(self, d_model: int, kernel_size: int = 3, bias: bool = False) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive.")
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            groups=d_model,
            bias=bias,
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, hidden_states: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        if hidden_states.shape[-1] != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {hidden_states.shape[-1]}")
        batch_size, seq_len, _ = hidden_states.shape
        if padding_mask is not None:
            if padding_mask.shape != (batch_size, seq_len):
                raise ValueError("padding_mask must have shape (batch, seq_len)")
            hidden_states = hidden_states * padding_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)

        conv_input, gate, value = self.in_proj(hidden_states).chunk(3, dim=-1)
        conv_states = (conv_input * value).transpose(1, 2)
        conv_states = self.conv(conv_states)[..., :seq_len].transpose(1, 2).contiguous()
        output = self.out_proj(gate * conv_states)

        if padding_mask is not None:
            output = output * padding_mask.to(dtype=output.dtype).unsqueeze(-1)
        return output


class LFM2MLP(nn.Module):
    """Biasless SwiGLU MLP with explicit LFM2-style w1/w3/w2 projection names."""

    def __init__(self, d_model: int, hidden_size: int) -> None:
        super().__init__()
        if d_model <= 0 or hidden_size <= 0:
            raise ValueError("d_model and hidden_size must be positive.")
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.w1 = nn.Linear(d_model, hidden_size, bias=False)
        self.w3 = nn.Linear(d_model, hidden_size, bias=False)
        self.w2 = nn.Linear(hidden_size, d_model, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.shape[-1] != self.d_model:
            raise ValueError(f"Expected last dim {self.d_model}, got {hidden_states.shape[-1]}")
        return self.w2(F.silu(self.w1(hidden_states)) * self.w3(hidden_states))
