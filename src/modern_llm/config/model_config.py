"""Model configuration dataclasses with strict validation.

The fields mirror decoder-only LMs such as GPT (Radford et al., 2018) and
LLaMA (Touvron et al., 2023), capturing architectural toggles (RoPE, RMSNorm,
SwiGLU, GQA, MoE) discussed in those papers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class MoEConfig:
    """Configuration for a Mixture-of-Experts feedforward sub-layer."""

    num_experts: int = 4
    top_k: int = 2
    dropout: float = 0.0
    capacity_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError(f"num_experts must be positive, received {self.num_experts}")
        if self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError(
                f"top_k must be in [1, num_experts], received top_k={self.top_k}, num_experts={self.num_experts}"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), received {self.dropout}")
        if self.capacity_factor < 1.0:
            raise ValueError(f"capacity_factor must be >= 1.0, received {self.capacity_factor}")


@dataclass
class ModernLLMConfig:
    """Configuration for the custom decoder-only Transformer."""

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    ffn_hidden_size: int
    max_seq_len: int
    rmsnorm_eps: float = 1e-5
    dropout: float = 0.0
    initializer_range: float = 0.02
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None
    use_rope: bool = True
    use_attention_sinks: bool = True
    num_attention_sinks: int = 2
    use_swiglu: bool = True
    swiglu_multiplier: float = 2.0
    use_gqa: bool = False
    gqa_groups: Optional[int] = None
    use_qk_norm: bool = False
    use_moe: bool = False
    moe_config: Optional[MoEConfig] = None
    tie_embeddings: bool = True
    scale_embeddings: bool = False
    residual_init_scale: bool = True
    z_loss_coef: float = 0.0

    def __post_init__(self) -> None:
        self._validate_dimensions()
        self._validate_attention_settings()
        self._validate_moe_settings()

    def _validate_dimensions(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, received {self.vocab_size}")
        if self.d_model <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model must be positive and divisible by n_heads "
                f"(d_model={self.d_model}, n_heads={self.n_heads})"
            )
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, received {self.n_layers}")
        if self.ffn_hidden_size <= self.d_model:
            raise ValueError(
                f"ffn_hidden_size must exceed d_model "
                f"(ffn_hidden_size={self.ffn_hidden_size}, d_model={self.d_model})"
            )
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, received {self.max_seq_len}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), received {self.dropout}")
        if self.rmsnorm_eps <= 0:
            raise ValueError(f"rmsnorm_eps must be positive, received {self.rmsnorm_eps}")
        if self.initializer_range <= 0:
            raise ValueError(f"initializer_range must be positive, received {self.initializer_range}")
        if self.z_loss_coef < 0:
            raise ValueError(f"z_loss_coef must be non-negative, received {self.z_loss_coef}")

    def _validate_attention_settings(self) -> None:
        if self.use_attention_sinks and self.num_attention_sinks <= 0:
            raise ValueError(
                f"num_attention_sinks must be positive when use_attention_sinks is True, "
                f"received {self.num_attention_sinks}"
            )
        if self.use_gqa:
            if not self.gqa_groups:
                raise ValueError("gqa_groups must be provided when use_gqa is True")
            if self.n_heads % self.gqa_groups != 0:
                raise ValueError(
                    f"gqa_groups must divide n_heads (gqa_groups={self.gqa_groups}, n_heads={self.n_heads})"
                )
        if self.rope_scaling is not None and self.rope_scaling <= 0:
            raise ValueError(f"rope_scaling must be positive, received {self.rope_scaling}")

    def _validate_moe_settings(self) -> None:
        if self.use_moe and self.moe_config is None:
            raise ValueError("moe_config must be set when use_moe is True")
        if not self.use_moe and self.moe_config is not None:
            raise ValueError("moe_config should be None when use_moe is False")

