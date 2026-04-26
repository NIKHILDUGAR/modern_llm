import pytest

from modern_llm.config import ModernLLMConfig, MoEConfig


def test_invalid_head_configuration_raises() -> None:
    with pytest.raises(ValueError):
        ModernLLMConfig(
            vocab_size=100,
            d_model=63,
            n_layers=2,
            n_heads=8,
            ffn_hidden_size=256,
            max_seq_len=128,
        )


def test_moe_requires_config_when_enabled() -> None:
    with pytest.raises(ValueError):
        ModernLLMConfig(
            vocab_size=100,
            d_model=64,
            n_layers=2,
            n_heads=8,
            ffn_hidden_size=256,
            max_seq_len=128,
            use_moe=True,
        )


def test_valid_moe_configuration_passes() -> None:
    config = ModernLLMConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=8,
        ffn_hidden_size=256,
        max_seq_len=128,
        use_moe=True,
        moe_config=MoEConfig(num_experts=2, top_k=1),
    )
    assert config.use_moe is True


def test_default_sequence_mixer_is_attention() -> None:
    config = ModernLLMConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=8,
        ffn_hidden_size=256,
        max_seq_len=128,
    )
    assert config.sequence_mixer == "attention"
    assert config.uses_gated_deltanet_layer(0) is False


def test_hybrid_gated_deltanet_layer_selection() -> None:
    config = ModernLLMConfig(
        vocab_size=100,
        d_model=64,
        n_layers=4,
        n_heads=8,
        ffn_hidden_size=256,
        max_seq_len=128,
        sequence_mixer="hybrid_gated_deltanet",
        gated_deltanet_layers=[1, 2],
    )
    assert config.uses_gated_deltanet_layer(0) is False
    assert config.uses_gated_deltanet_layer(1) is True
    assert config.uses_gated_deltanet_layer(2) is True
    assert config.uses_gated_deltanet_layer(3) is False


def test_invalid_gated_deltanet_layer_raises() -> None:
    with pytest.raises(ValueError):
        ModernLLMConfig(
            vocab_size=100,
            d_model=64,
            n_layers=2,
            n_heads=8,
            ffn_hidden_size=256,
            max_seq_len=128,
            sequence_mixer="hybrid_gated_deltanet",
            gated_deltanet_layers=[2],
        )
