"""Multi-head attention primitives with research context.

- Scaled dot-product attention follows Vaswani et al. (2017, §3.2.1).
- Rotary positional embeddings (RoPE) match Su et al. (2021) using complex
  rotations to encode relative positions.
- Attention sinks extend Press et al. (2021) by prepending fixed vectors that
  every token can attend to, improving long-context stability.
- Grouped Query Attention (GQA) mirrors Ainslie et al. (2023) by sharing K/V
  heads across multiple Q heads to reduce KV cache memory.
- QK-norm (Henry et al., 2020; ViT-22B, Chameleon, OLMo-2) applies RMSNorm over
  each head's dimension to Q and K before the dot product, stabilising logits
  in bf16 at depth without changing the mathematical meaning of RoPE (RMSNorm
  is invariant to orthogonal rotations, so order w.r.t. RoPE is equivalent).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .layers import RMSNorm


@dataclass
class AttentionConfig:
    """Hyperparameters controlling the attention mechanism."""

    d_model: int
    n_heads: int
    use_rope: bool = True
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None
    rope_pairing: str = "half_split"
    use_attention_sinks: bool = False
    num_attention_sinks: int = 2
    use_gqa: bool = False
    gqa_groups: Optional[int] = None
    use_qk_norm: bool = False
    qk_norm_eps: float = 1e-5
    dropout: float = 0.0
    use_flash_attention: bool = True  # Use PyTorch SDPA (includes Flash Attention)

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive.")
        if self.use_attention_sinks and self.num_attention_sinks <= 0:
            raise ValueError("num_attention_sinks must be > 0 when sinks are enabled.")
        if self.use_gqa:
            if not self.gqa_groups:
                raise ValueError("gqa_groups must be set when use_gqa=True.")
            if self.n_heads % self.gqa_groups != 0:
                raise ValueError("gqa_groups must divide n_heads.")


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention per Vaswani et al. (2017).

    Math:
        Attention(Q, K, V) = softmax((Q Kᵀ + mask) / sqrt(d_k)) V
        RoPE (Su et al., 2021) applies a position-dependent rotation to Q and K.

    Pre:
        - hidden_states has shape (batch, seq, d_model).
        - attention_mask is additive with shape (batch, 1, seq, seq_k).
    Post:
        - returns tensor with identical shape.
    Complexity:
        - O(seq² · d_model / n_heads) per layer due to QKᵀ.
    Invariants:
        - head_dim = d_model / n_heads (validated on init).
    """

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if config.use_rope and (config.d_model // config.n_heads) % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")

        self.config = config
        self.head_dim = config.d_model // config.n_heads
        self.num_q_heads = config.n_heads
        self.num_kv_heads = config.gqa_groups if config.use_gqa and config.gqa_groups else config.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, kv_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.qk_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.qk_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        if config.use_attention_sinks:
            self.sink_states = nn.Parameter(
                torch.randn(config.num_attention_sinks, config.d_model) * 0.02
            )
        else:
            self.register_parameter("sink_states", None)

    def forward(self, hidden_states: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        q = self._shape_q(self.q_proj(hidden_states))
        k = self._shape_kv(self.k_proj(hidden_states))
        v = self._shape_kv(self.v_proj(hidden_states))

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.config.use_rope:
            q = self._apply_rope(q, seq_len, offset=self.config.num_attention_sinks if self.config.use_attention_sinks else 0)
            k = self._apply_rope(k, seq_len)

        if self.num_kv_heads != self.num_q_heads:
            repeat_factor = self.num_q_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        key_length = seq_len
        if self.config.use_attention_sinks and self.sink_states is not None:
            sink_states = self.sink_states.unsqueeze(0).expand(batch_size, -1, -1)
            sink_k = self._shape_kv(self.k_proj(sink_states))
            sink_v = self._shape_kv(self.v_proj(sink_states))
            if self.k_norm is not None:
                sink_k = self.k_norm(sink_k)
            if self.config.use_rope:
                sink_k = self._apply_rope(sink_k, self.config.num_attention_sinks, offset=0)
            if self.num_kv_heads != self.num_q_heads:
                repeat_factor = self.num_q_heads // self.num_kv_heads
                sink_k = sink_k.repeat_interleave(repeat_factor, dim=1)
                sink_v = sink_v.repeat_interleave(repeat_factor, dim=1)
            k = torch.cat([sink_k, k], dim=2)
            v = torch.cat([sink_v, v], dim=2)
            key_length += self.config.num_attention_sinks
            if attention_mask is not None:
                sink_bias = torch.zeros(
                    batch_size,
                    1,
                    attention_mask.size(-2),
                    self.config.num_attention_sinks,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([sink_bias, attention_mask], dim=-1)

        # Use Flash Attention (SDPA) when enabled and no attention sinks
        # SDPA is 2-4x faster and more memory efficient
        use_sdpa = (
            self.config.use_flash_attention
            and not self.config.use_attention_sinks
            and hasattr(F, "scaled_dot_product_attention")
        )
        
        if use_sdpa:
            # SDPA expects (B, H, S, D) format - already in this format
            dropout_p = self.config.dropout if self.training else 0.0
            context = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,  # Use is_causal instead for efficiency
                dropout_p=dropout_p,
                is_causal=True,
                scale=self.scale,
            )
            # Reshape from (B, H, S, D) to (B, S, d_model)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_model)
            return self.out_proj(context)
        else:
            # Fallback to manual attention (needed for attention sinks)
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if attention_mask is not None:
                if attention_mask.shape[-1] != key_length:
                    raise ValueError(
                        f"attention_mask length {attention_mask.shape[-1]} "
                        f"does not match K length {key_length}"
                    )
                attn_scores = attn_scores + attention_mask

            attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(attn_scores.dtype)
            attn_probs = self.dropout(attn_probs)
            context = torch.matmul(attn_probs, v)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.config.d_model)
            return self.out_proj(context)

    # --- helpers -----------------------------------------------------------------

    def _shape_q(self, tensor: Tensor) -> Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_q_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, tensor: Tensor) -> Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

    def _apply_rope(self, tensor: Tensor, seq_len: int, offset: int = 0) -> Tensor:
        cos, sin = self._get_rope_factors(seq_len + offset, tensor.device, tensor.dtype)
        cos = cos[offset : offset + seq_len]
        sin = sin[offset : offset + seq_len]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return (tensor * cos) + (self._rotate_half(tensor) * sin)

    def _get_rope_factors(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        inv_freq = self._get_inv_freq(device)
        freqs = torch.outer(
            torch.arange(seq_len, device=device), inv_freq
        )
        if self.config.rope_scaling:
            freqs = freqs * self.config.rope_scaling
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1).to(dtype=dtype)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1).to(dtype=dtype)
        return cos, sin

    def _get_inv_freq(self, device: torch.device) -> Tensor:
        """Return a valid RoPE frequency buffer, repairing HF meta-load damage."""

        inv_freq = getattr(self, "inv_freq", None)
        expected = self.head_dim // 2
        expected_inv_freq = 1.0 / (
            self.config.rope_theta
            ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim)
        )
        if (
            not isinstance(inv_freq, Tensor)
            or inv_freq.numel() != expected
            or inv_freq.device.type == "meta"
            or not torch.isfinite(inv_freq.detach().float()).all()
            or not torch.allclose(
                inv_freq.to(device=device).detach().float(),
                expected_inv_freq,
                rtol=1e-5,
                atol=1e-7,
            )
        ):
            self.register_buffer("inv_freq", expected_inv_freq, persistent=False)
            return expected_inv_freq
        return inv_freq.to(device=device)

    def _rotate_half(self, x: Tensor) -> Tensor:
        if self.config.rope_pairing == "interleaved":
            x_even = x[..., ::2]
            x_odd = x[..., 1::2]
            return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
        x1, x2 = x[..., : x.size(-1) // 2], x[..., x.size(-1) // 2 :]
        return torch.cat([-x2, x1], dim=-1)
