from pathlib import Path

import torch

from modern_llm.config import ModernLLMConfig, PipelineConfig
from modern_llm.models import LFM2MLP, LFM2ShortConv, ModernDecoderLM


def test_hybrid_lfm2_builds_expected_layer_types() -> None:
    config = ModernLLMConfig(
        vocab_size=32,
        d_model=16,
        n_layers=3,
        n_heads=2,
        ffn_hidden_size=32,
        max_seq_len=8,
        sequence_mixer="hybrid_lfm2",
        lfm2_attention_layers=[1],
        use_gqa=True,
        gqa_groups=1,
        use_qk_norm=True,
        use_attention_sinks=False,
    )
    model = ModernDecoderLM(config)

    assert isinstance(model.blocks[0].conv, LFM2ShortConv)
    assert hasattr(model.blocks[1], "attn")
    assert isinstance(model.blocks[0].ffn, LFM2MLP)
    assert isinstance(model.blocks[1].ffn, LFM2MLP)


def test_hybrid_lfm2_forward_backward() -> None:
    config = ModernLLMConfig(
        vocab_size=32,
        d_model=16,
        n_layers=4,
        n_heads=2,
        ffn_hidden_size=32,
        max_seq_len=8,
        sequence_mixer="hybrid_lfm2",
        lfm2_attention_layers=[1, 3],
        use_gqa=True,
        gqa_groups=1,
        use_qk_norm=True,
        use_attention_sinks=False,
    )
    model = ModernDecoderLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, -2:] = 0

    outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)

    assert outputs["logits"].shape == (2, 8, config.vocab_size)
    assert outputs["loss"] is not None
    outputs["loss"].backward()
    assert model.blocks[0].conv.in_proj.weight.grad is not None
    assert model.blocks[1].attn.q_proj.weight.grad is not None
    assert model.blocks[0].ffn.w1.weight.grad is not None


def test_lfm2_short_conv_keeps_sequence_length_and_is_causal() -> None:
    torch.manual_seed(0)
    conv = LFM2ShortConv(d_model=4, kernel_size=3)
    x = torch.randn(1, 5, 4)

    baseline = conv(x)
    changed_future = x.clone()
    changed_future[:, 3:] = torch.randn_like(changed_future[:, 3:])
    changed = conv(changed_future)

    assert baseline.shape == x.shape
    assert torch.allclose(baseline[:, :3], changed[:, :3], atol=1e-6)


def test_hybrid_lfm2_quantizable_refs_include_conv_and_mlp() -> None:
    config = ModernLLMConfig(
        vocab_size=32,
        d_model=16,
        n_layers=2,
        n_heads=2,
        ffn_hidden_size=32,
        max_seq_len=8,
        sequence_mixer="hybrid_lfm2",
        lfm2_attention_layers=[1],
        use_gqa=True,
        gqa_groups=1,
        use_qk_norm=True,
        use_attention_sinks=False,
    )
    model = ModernDecoderLM(config)
    paths = {ref.module_path for ref in model.iter_quantizable_linear_layers()}

    assert "blocks.0.conv.in_proj" in paths
    assert "blocks.0.conv.out_proj" in paths
    assert "blocks.0.ffn.w1" in paths
    assert "blocks.0.ffn.w2" in paths
    assert "blocks.0.ffn.w3" in paths


def test_lfm2_config_parameter_count_under_75m() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs/lm_75m_2x4090_lfm2.json"
    model_config = PipelineConfig.load(config_path).get_model_config()
    assert _estimate_lfm2_lm_params(model_config) == 73_761_024
    assert _estimate_lfm2_lm_params(model_config) < 75_000_000


def _estimate_lfm2_lm_params(config: ModernLLMConfig) -> int:
    head_dim = config.d_model // config.n_heads
    kv_heads = config.gqa_groups if config.use_gqa and config.gqa_groups else config.n_heads
    kv_dim = kv_heads * head_dim

    embedding_params = config.vocab_size * config.d_model
    lm_head_params = 0 if config.tie_embeddings else config.vocab_size * config.d_model
    norm_params = 2 * config.d_model
    final_norm_params = config.d_model

    attention_params = (
        config.d_model * config.d_model
        + config.d_model * kv_dim
        + config.d_model * kv_dim
        + config.d_model * config.d_model
    )
    if config.use_qk_norm:
        attention_params += 2 * head_dim

    conv_params = (
        config.d_model * (3 * config.d_model)
        + config.d_model * config.lfm2_conv_kernel
        + config.d_model * config.d_model
    )
    if config.lfm2_conv_bias:
        conv_params += config.d_model

    ffn_params = 3 * config.d_model * config.ffn_hidden_size
    attention_layers = set(config.lfm2_attention_layers or [])
    block_params = 0
    for layer_idx in range(config.n_layers):
        operator_params = attention_params if layer_idx in attention_layers else conv_params
        block_params += operator_params + norm_params + ffn_params

    return embedding_params + lm_head_params + block_params + final_norm_params
