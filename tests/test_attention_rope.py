import torch

from modern_llm.models.attention import AttentionConfig, MultiHeadAttention


def test_interleaved_rope_rotate_pairs_even_odd_dimensions() -> None:
    attn = MultiHeadAttention(
        AttentionConfig(
            d_model=4,
            n_heads=1,
            rope_pairing="interleaved",
        )
    )
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

    rotated = attn._rotate_half(x)

    assert torch.equal(rotated, torch.tensor([[[[-2.0, 1.0, -4.0, 3.0]]]]))


def test_legacy_rope_rotate_preserves_old_checkpoint_behavior() -> None:
    attn = MultiHeadAttention(
        AttentionConfig(
            d_model=4,
            n_heads=1,
            rope_pairing="half_split",
        )
    )
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

    rotated = attn._rotate_half(x)

    assert torch.equal(rotated, torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]]))
