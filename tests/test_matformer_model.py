from pathlib import Path

import torch
import torch.nn.functional as F

from modern_llm.config import ModernLLMConfig, PipelineConfig
from modern_llm.models import ModernDecoderLM, SwiGLU


def test_swiglu_full_width_matformer_path_matches_default() -> None:
    torch.manual_seed(0)
    layer = SwiGLU(in_features=4, hidden_features=8, out_features=4)
    x = torch.randn(2, 3, 4)

    assert torch.allclose(layer(x, active_hidden_size=8), layer(x))


def test_swiglu_sliced_path_matches_manual_sliced_math() -> None:
    torch.manual_seed(0)
    layer = SwiGLU(in_features=4, hidden_features=8, out_features=4)
    x = torch.randn(2, 3, 4)
    active = 3

    gate_out = F.linear(x, layer.gate.weight[:active], layer.gate.bias[:active])
    value = F.linear(
        x,
        layer.gate.weight[layer.hidden_features : layer.hidden_features + active],
        layer.gate.bias[layer.hidden_features : layer.hidden_features + active],
    )
    manual = F.linear(
        gate_out * torch.sigmoid(gate_out) * value,
        layer.proj.weight[:, :active],
        layer.proj.bias,
    )

    assert torch.allclose(layer(x, active_hidden_size=active), manual)


def test_matformer_active_granularity_forward_backward() -> None:
    config = ModernLLMConfig(
        vocab_size=32,
        d_model=16,
        n_layers=2,
        n_heads=2,
        ffn_hidden_size=32,
        max_seq_len=8,
        use_matformer=True,
        matformer_granularities=[8, 16, 32],
        matformer_active_granularity=16,
    )
    model = ModernDecoderLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    outputs = model(input_ids, labels=input_ids)

    assert outputs["logits"].shape == (2, 8, config.vocab_size)
    assert outputs["loss"] is not None
    outputs["loss"].backward()
    assert model.blocks[0].ffn.gate.weight.grad is not None


def test_matformer_per_layer_mix_and_training_sample() -> None:
    config = ModernLLMConfig(
        vocab_size=32,
        d_model=16,
        n_layers=3,
        n_heads=2,
        ffn_hidden_size=32,
        max_seq_len=8,
        use_matformer=True,
        matformer_granularities=[8, 16, 32],
        matformer_train_sample=True,
    )
    model = ModernDecoderLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))

    mixed = model(input_ids, labels=input_ids, matformer_granularity=[8, 16, 32])
    assert mixed["logits"].shape == (2, 8, config.vocab_size)
    assert mixed["loss"] is not None

    model.train()
    sampled = model(input_ids, labels=input_ids)
    assert sampled["logits"].shape == (2, 8, config.vocab_size)
    assert sampled["loss"] is not None


def test_matformer_config_parameter_count_under_75m() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs/lm_75m_2x4090_matformer.json"
    model_config = PipelineConfig.load(config_path).get_model_config()
    assert _estimate_modern_lm_params(model_config) < 75_000_000


def _estimate_modern_lm_params(config: ModernLLMConfig) -> int:
    head_dim = config.d_model // config.n_heads
    kv_heads = config.gqa_groups if config.use_gqa and config.gqa_groups else config.n_heads
    kv_dim = kv_heads * head_dim

    embedding_params = config.vocab_size * config.d_model
    lm_head_params = 0 if config.tie_embeddings else config.vocab_size * config.d_model

    attention_params = (
        config.d_model * config.d_model
        + config.d_model * kv_dim
        + config.d_model * kv_dim
        + config.d_model * config.d_model
    )
    if config.use_attention_sinks:
        attention_params += config.num_attention_sinks * config.d_model
    if config.use_qk_norm:
        attention_params += 2 * head_dim

    block_norm_params = 2 * config.d_model
    ffn_params = (
        2 * config.ffn_hidden_size * config.d_model
        + 2 * config.ffn_hidden_size
        + config.ffn_hidden_size * config.d_model
        + config.d_model
    )

    return (
        embedding_params
        + lm_head_params
        + config.n_layers * (attention_params + block_norm_params + ffn_params)
        + config.d_model
    )
