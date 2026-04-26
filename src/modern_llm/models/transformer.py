"""Decoder-only Transformer scaffold with research-grounded commentary.

Architecture references:
- Transformer decoder stack from Vaswani et al. (2017, §3) with causal masking.
- RoPE positional encodings per Su et al. (2021) for better extrapolation.
- RMSNorm (Zhang & Sennrich, 2019) and SwiGLU (Shazeer, 2020; PaLM, 2022).
- Attention sinks inspired by Press et al. (2021) for long-context stability.

This module documents the math/architecture before Phase 1 fleshes out the code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from modern_llm.config.model_config import ModernLLMConfig
from modern_llm.models.attention import AttentionConfig, MultiHeadAttention
from modern_llm.models.gated_deltanet import GatedDeltaNet, GatedDeltaNetConfig
from modern_llm.models.layers import RMSNorm, SwiGLU
from modern_llm.models.moe import MixtureOfExperts


@dataclass(frozen=True)
class QuantizableLinearRef:
    """Stable reference to a linear layer that can be quantized opt-in."""

    module_path: str
    parent: nn.Module
    attr_name: str
    module: nn.Linear
    block_index: Optional[int]
    group: str


class DecoderBlock(nn.Module):
    """Transformer decoder block (Vaswani et al., 2017).

    Math:
        h' = h + MultiHeadAttention(RMSNorm(h))
        h'' = h' + SwiGLU(RMSNorm(h'))
        where attention implements softmax(QKᵀ/√d_k) V with RoPE rotations.

    Pre:
        - hidden_states shape: (batch, seq, d_model)
        - attention_mask encodes causal + optional sink positions.
    Post:
        - returns tensor of same shape, ready for the next block.
    Complexity:
        - Dominated by attention: O(seq² * d_model / n_heads) per block.
    Invariants:
        - Residual connections keep tensor dimensionality constant.
    """

    def __init__(self, config: ModernLLMConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.uses_gated_deltanet = config.uses_gated_deltanet_layer(layer_index)
        if self.uses_gated_deltanet:
            self.gated_deltanet = GatedDeltaNet(
                GatedDeltaNetConfig(
                    d_model=config.d_model,
                    num_heads=config.gated_deltanet_num_heads or config.n_heads,
                    conv_kernel=config.gated_deltanet_conv_kernel,
                    dropout=config.dropout,
                    rmsnorm_eps=config.rmsnorm_eps,
                )
            )
        else:
            attn_config = AttentionConfig(
                d_model=config.d_model,
                n_heads=config.n_heads,
                use_rope=config.use_rope,
                rope_theta=config.rope_theta,
                rope_scaling=config.rope_scaling,
                use_attention_sinks=config.use_attention_sinks,
                num_attention_sinks=config.num_attention_sinks,
                use_gqa=config.use_gqa,
                gqa_groups=config.gqa_groups,
                use_qk_norm=config.use_qk_norm,
                qk_norm_eps=config.rmsnorm_eps,
                dropout=config.dropout,
            )
            self.attn = MultiHeadAttention(attn_config)
        self.attn_norm = RMSNorm(config.d_model, config.rmsnorm_eps)
        self.ffn_norm = RMSNorm(config.d_model, config.rmsnorm_eps)
        hidden = config.ffn_hidden_size
        if config.use_moe and config.moe_config is not None:
            self.ffn = MixtureOfExperts(config.d_model, config.moe_config)
        else:
            self.ffn = SwiGLU(config.d_model, hidden, out_features=config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        attn_input = self.attn_norm(hidden_states)
        if self.uses_gated_deltanet:
            attn_output = self.gated_deltanet(attn_input, padding_mask=padding_mask)
        else:
            attn_output = self.attn(attn_input, attention_mask=attention_mask)
        hidden_states = hidden_states + self.dropout(attn_output)

        ffn_input = self.ffn_norm(hidden_states)
        ffn_output = self.ffn(ffn_input)
        hidden_states = hidden_states + self.dropout(ffn_output)
        return hidden_states


class ModernDecoderLM(nn.Module):
    """Decoder-only language model with RoPE + RMSNorm stack.

    The model mirrors GPT-style causal LMs but swaps LayerNorm for RMSNorm and
    GELU for SwiGLU, matching PaLM/LLaMA-era design choices.

    Invariants:
        - Token embeddings and LM head share weights when `tie_embeddings=True`.
    Complexity:
        - O(n_layers · seq² · d_model / n_heads) per forward pass.
    """

    def __init__(self, config: ModernLLMConfig) -> None:
        super().__init__()
        self.config = config
        self._quantization_metadata: Optional[dict] = None
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [DecoderBlock(config, layer_index=i) for i in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.d_model, config.rmsnorm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embed.weight
        self.apply(self._init_weights)
        if config.residual_init_scale:
            self._apply_residual_init_scale()

    def _apply_residual_init_scale(self) -> None:
        # Megatron-style: scale residual-path output projections by 1/sqrt(2*n_layers)
        # so the variance at depth L stays O(1) regardless of depth.
        scale = 1.0 / math.sqrt(2.0 * self.config.n_layers)
        for block in self.blocks:
            attn_out = getattr(getattr(block, "attn", None), "out_proj", None)
            if isinstance(attn_out, nn.Linear):
                attn_out.weight.data.mul_(scale)
            gated_out = getattr(getattr(block, "gated_deltanet", None), "out_proj", None)
            if isinstance(gated_out, nn.Linear):
                gated_out.weight.data.mul_(scale)
            ffn = block.ffn
            ffn_out = getattr(ffn, "proj", None)
            if isinstance(ffn_out, nn.Linear):
                ffn_out.weight.data.mul_(scale)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Optional[Tensor]]:
        """Causal LM forward pass (to be implemented in Phase 1).

        Pre:
            - input_ids shape: (batch, seq)
            - attention_mask matches shape or is broadcastable.
        Post:
            - returns logits of shape (batch, seq, vocab_size) once implemented.
        """

        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch, seq_len)")
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
            )

        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        attention_mask = attention_mask.to(dtype=torch.float32)

        hidden_states = self.token_embed(input_ids)
        if self.config.scale_embeddings:
            hidden_states = hidden_states * math.sqrt(self.config.d_model)
        hidden_states = self.dropout(hidden_states)

        padding_mask = attention_mask.to(dtype=hidden_states.dtype)
        attention_bias = self._build_attention_bias(attention_mask, hidden_states.dtype)
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_bias, padding_mask=padding_mask)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            flat_logits = shift_logits.view(-1, self.config.vocab_size)
            flat_labels = shift_labels.view(-1)
            loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
            if self.config.z_loss_coef > 0.0:
                # PaLM §5.1 z-loss: penalize log(Z)^2 to keep the logit partition
                # function near 1, which stabilizes bf16 without fp32 softmax.
                valid = flat_labels != -100
                if valid.any():
                    log_z = torch.logsumexp(flat_logits[valid], dim=-1)
                    loss = loss + self.config.z_loss_coef * log_z.pow(2).mean()

        return {"logits": logits, "loss": loss}

    def _build_attention_bias(self, attention_mask: Tensor, dtype: torch.dtype) -> Tensor:
        batch_size, seq_len = attention_mask.shape
        device = attention_mask.device
        neg_inf = torch.finfo(dtype).min
        causal_mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        causal_mask = causal_mask.masked_fill(
            torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1),
            neg_inf,
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        padding_bias = (1.0 - attention_mask).unsqueeze(1).unsqueeze(2) * neg_inf
        return causal_mask + padding_bias.to(dtype)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def iter_quantizable_linear_layers(self) -> Iterator[QuantizableLinearRef]:
        """Yield stable references for opt-in low-bit replacement."""

        for block_idx, block in enumerate(self.blocks):
            attn = getattr(block, "attn", None)
            for attr_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                module = getattr(attn, attr_name, None)
                if isinstance(module, nn.Linear):
                    yield QuantizableLinearRef(
                        module_path=f"blocks.{block_idx}.attn.{attr_name}",
                        parent=block.attn,
                        attr_name=attr_name,
                        module=module,
                        block_index=block_idx,
                        group="attention",
                    )
            gated_deltanet = getattr(block, "gated_deltanet", None)
            for attr_name in ("q_proj", "k_proj", "v_proj", "g_proj", "a_proj", "b_proj", "out_proj"):
                module = getattr(gated_deltanet, attr_name, None)
                if isinstance(module, nn.Linear):
                    yield QuantizableLinearRef(
                        module_path=f"blocks.{block_idx}.gated_deltanet.{attr_name}",
                        parent=gated_deltanet,
                        attr_name=attr_name,
                        module=module,
                        block_index=block_idx,
                        group="gated_deltanet",
                    )
            for attr_name in ("gate", "proj"):
                module = getattr(block.ffn, attr_name, None)
                if isinstance(module, nn.Linear):
                    yield QuantizableLinearRef(
                        module_path=f"blocks.{block_idx}.ffn.{attr_name}",
                        parent=block.ffn,
                        attr_name=attr_name,
                        module=module,
                        block_index=block_idx,
                        group="ffn",
                    )
        if isinstance(self.lm_head, nn.Linear):
            yield QuantizableLinearRef(
                module_path="lm_head",
                parent=self,
                attr_name="lm_head",
                module=self.lm_head,
                block_index=None,
                group="lm_head",
            )
