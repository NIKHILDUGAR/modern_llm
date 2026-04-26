import torch

from modern_llm.config import PipelineConfig, ModernLLMConfig
from modern_llm.models import GatedDeltaNet, ModernDecoderLM


def test_dense_default_still_builds_attention_block() -> None:
    config = ModernLLMConfig(
        vocab_size=32,
        d_model=16,
        n_layers=2,
        n_heads=2,
        ffn_hidden_size=32,
        max_seq_len=8,
    )
    model = ModernDecoderLM(config)
    assert hasattr(model.blocks[0], "attn")
    assert not hasattr(model.blocks[0], "gated_deltanet")


def test_hybrid_gated_deltanet_forward_backward() -> None:
    config = ModernLLMConfig(
        vocab_size=32,
        d_model=16,
        n_layers=3,
        n_heads=2,
        ffn_hidden_size=32,
        max_seq_len=8,
        sequence_mixer="hybrid_gated_deltanet",
        gated_deltanet_layers=[1],
        gated_deltanet_num_heads=2,
    )
    model = ModernDecoderLM(config)
    assert hasattr(model.blocks[0], "attn")
    assert isinstance(model.blocks[1].gated_deltanet, GatedDeltaNet)

    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, -2:] = 0
    outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)

    assert outputs["logits"].shape == (2, 8, config.vocab_size)
    assert outputs["loss"] is not None
    outputs["loss"].backward()
    assert model.blocks[1].gated_deltanet.q_proj.weight.grad is not None


def test_pipeline_config_passes_gated_deltanet_settings(tmp_path) -> None:
    pipeline = PipelineConfig(
        output_dir=f"{tmp_path}/",
        vocab_size=32,
        d_model=16,
        n_layers=3,
        n_heads=2,
        ffn_hidden_size=32,
        max_seq_len=8,
        sequence_mixer="hybrid_gated_deltanet",
        gated_deltanet_layers=[1],
        gated_deltanet_num_heads=2,
    )
    model_config = pipeline.get_model_config()
    assert model_config.sequence_mixer == "hybrid_gated_deltanet"
    assert model_config.gated_deltanet_layers == [1]
