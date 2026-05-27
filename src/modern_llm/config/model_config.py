"""Model configuration dataclasses with strict validation.

The fields mirror decoder-only LMs such as GPT (Radford et al., 2018) and
LLaMA (Touvron et al., 2023), capturing architectural toggles (RoPE, RMSNorm,
SwiGLU, GQA, MoE) discussed in those papers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional


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
    # Legacy checkpoints used half-split rotation with interleaved cos/sin
    # factors. New training configs should set this to "interleaved", which is
    # the standard RoPE pairing for repeat-interleaved factors.
    rope_pairing: str = "half_split"
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
    sequence_mixer: str = "attention"
    gated_deltanet_layers: Optional[List[int]] = None
    gated_deltanet_num_heads: Optional[int] = None
    gated_deltanet_conv_kernel: int = 4
    lfm2_attention_layers: Optional[List[int]] = None
    lfm2_conv_kernel: int = 3
    lfm2_conv_bias: bool = False
    use_matformer: bool = False
    matformer_granularities: Optional[List[int]] = None
    matformer_train_sample: bool = False
    matformer_sampling_probs: Optional[List[float]] = None
    matformer_active_granularity: Optional[int] = None

    def __post_init__(self) -> None:
        self._validate_dimensions()
        self._validate_attention_settings()
        self._validate_moe_settings()
        self._validate_sequence_mixer_settings()
        self._validate_matformer_settings()

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
        if self.rope_pairing not in {"half_split", "interleaved"}:
            raise ValueError(
                "rope_pairing must be one of {'half_split', 'interleaved'}, "
                f"received {self.rope_pairing!r}"
            )

    def _validate_moe_settings(self) -> None:
        if self.use_moe and self.moe_config is None:
            raise ValueError("moe_config must be set when use_moe is True")
        if not self.use_moe and self.moe_config is not None:
            raise ValueError("moe_config should be None when use_moe is False")

    def _validate_sequence_mixer_settings(self) -> None:
        allowed = {"attention", "gated_deltanet", "hybrid_gated_deltanet", "hybrid_lfm2"}
        if self.sequence_mixer not in allowed:
            raise ValueError(
                f"sequence_mixer must be one of {sorted(allowed)}, received {self.sequence_mixer!r}"
            )
        if self.gated_deltanet_conv_kernel <= 0:
            raise ValueError(
                f"gated_deltanet_conv_kernel must be positive, received {self.gated_deltanet_conv_kernel}"
            )

        delta_heads = self.gated_deltanet_num_heads or self.n_heads
        if delta_heads <= 0 or self.d_model % delta_heads != 0:
            raise ValueError(
                f"gated_deltanet_num_heads must be positive and divide d_model "
                f"(gated_deltanet_num_heads={delta_heads}, d_model={self.d_model})"
            )

        if self.sequence_mixer == "hybrid_gated_deltanet" and self.gated_deltanet_layers is None:
            raise ValueError(
                "gated_deltanet_layers must be provided when sequence_mixer='hybrid_gated_deltanet'"
            )
        if self.gated_deltanet_layers is not None:
            self._validate_layer_indices("gated_deltanet", self.gated_deltanet_layers)

        if self.lfm2_conv_kernel <= 0:
            raise ValueError(f"lfm2_conv_kernel must be positive, received {self.lfm2_conv_kernel}")
        if self.sequence_mixer == "hybrid_lfm2":
            if not self.lfm2_attention_layers:
                raise ValueError(
                    "lfm2_attention_layers must be provided when sequence_mixer='hybrid_lfm2'"
                )
            if self.use_moe:
                raise ValueError("LFM2 dense hybrid blocks are not compatible with use_moe=True")
        if self.lfm2_attention_layers is not None:
            self._validate_layer_indices("lfm2_attention", self.lfm2_attention_layers)

    def _validate_layer_indices(self, name: str, layer_indices: List[int]) -> None:
        if len(set(layer_indices)) != len(layer_indices):
            raise ValueError(f"{name} layer indices must not contain duplicates")
        for layer_idx in layer_indices:
            if layer_idx < 0 or layer_idx >= self.n_layers:
                raise ValueError(f"{name} layer index {layer_idx} is outside [0, {self.n_layers})")

    def uses_gated_deltanet_layer(self, layer_index: int) -> bool:
        """Return whether a decoder block should use the opt-in Gated DeltaNet mixer."""

        if self.sequence_mixer == "attention":
            return False
        if self.sequence_mixer == "gated_deltanet":
            return True
        if self.sequence_mixer != "hybrid_gated_deltanet":
            return False
        return layer_index in set(self.gated_deltanet_layers or [])

    def uses_lfm2_attention_layer(self, layer_index: int) -> bool:
        """Return whether a hybrid LFM2 block should use attention instead of short conv."""

        if self.sequence_mixer != "hybrid_lfm2":
            return False
        return layer_index in set(self.lfm2_attention_layers or [])

    def _validate_matformer_settings(self) -> None:
        if not self.use_matformer:
            if self.matformer_granularities is not None:
                raise ValueError("matformer_granularities requires use_matformer=True")
            if self.matformer_train_sample:
                raise ValueError("matformer_train_sample requires use_matformer=True")
            if self.matformer_sampling_probs is not None:
                raise ValueError("matformer_sampling_probs requires use_matformer=True")
            if self.matformer_active_granularity is not None:
                raise ValueError("matformer_active_granularity requires use_matformer=True")
            return

        if self.use_moe:
            raise ValueError("MatFormer FFN slicing is not compatible with use_moe=True")
        if self.sequence_mixer == "hybrid_lfm2":
            raise ValueError("MatFormer FFN slicing is not compatible with sequence_mixer='hybrid_lfm2'")

        if self.matformer_granularities is None:
            if self.ffn_hidden_size != self.d_model * 4:
                raise ValueError(
                    "matformer_granularities must be provided unless "
                    "ffn_hidden_size is 4 * d_model"
                )
            self.matformer_granularities = [
                self.d_model // 2,
                self.d_model,
                self.d_model * 2,
                self.ffn_hidden_size,
            ]

        granularities = self.matformer_granularities
        if not granularities:
            raise ValueError("matformer_granularities must not be empty")
        previous = 0
        for granularity in granularities:
            if isinstance(granularity, bool) or not isinstance(granularity, int):
                raise ValueError("matformer_granularities must contain integers")
            if granularity <= 0:
                raise ValueError("matformer_granularities must be positive")
            if granularity <= previous:
                raise ValueError("matformer_granularities must be strictly increasing")
            previous = granularity
        if granularities[-1] != self.ffn_hidden_size:
            raise ValueError(
                "matformer_granularities must end at ffn_hidden_size "
                f"(last={granularities[-1]}, ffn_hidden_size={self.ffn_hidden_size})"
            )

        if (
            self.matformer_active_granularity is not None
            and self.matformer_active_granularity not in granularities
        ):
            raise ValueError("matformer_active_granularity must be one of matformer_granularities")

        if self.matformer_sampling_probs is None:
            return
        if len(self.matformer_sampling_probs) != len(granularities):
            raise ValueError("matformer_sampling_probs must match matformer_granularities length")
        total = 0.0
        for prob in self.matformer_sampling_probs:
            if not math.isfinite(prob) or prob < 0.0:
                raise ValueError("matformer_sampling_probs must be finite and non-negative")
            total += prob
        if total <= 0.0:
            raise ValueError("matformer_sampling_probs must have positive total mass")
