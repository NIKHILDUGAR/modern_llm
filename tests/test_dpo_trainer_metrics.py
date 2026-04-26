"""Tests for DPO trainer metric aggregation."""

from __future__ import annotations

import torch
from torch import nn

from modern_llm.config import TrainingConfig
from modern_llm.training.train_dpo import DPOConfig, DPOTrainer


class _TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 8, d_model: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        return {"logits": self.proj(self.embed(input_ids))}


def _batch() -> dict[str, torch.Tensor]:
    return {
        "chosen_input_ids": torch.tensor([[1, 2, 3], [1, 3, 4]], dtype=torch.long),
        "chosen_attention_mask": torch.ones(2, 3, dtype=torch.long),
        "rejected_input_ids": torch.tensor([[1, 4, 3], [1, 2, 4]], dtype=torch.long),
        "rejected_attention_mask": torch.ones(2, 3, dtype=torch.long),
    }


def test_dpo_training_step_reports_only_completed_optimizer_steps(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")

    model = _TinyLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    config = TrainingConfig(
        run_name="dpo-test",
        dataset_name="dummy",
        tokenizer_name="dummy",
        output_dir=tmp_path,
        batch_size=4,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        max_steps=2,
        mixed_precision="fp32",
        compile_model=False,
    )
    trainer = DPOTrainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=[],
        config=config,
        dpo_config=DPOConfig(beta=0.05, max_length=3),
    )

    _, _, completed = trainer._training_step(_batch(), accumulation_steps=2)
    assert completed is False
    assert trainer.global_step == 0

    loss, metrics, completed = trainer._training_step(_batch(), accumulation_steps=2)
    assert completed is True
    assert trainer.global_step == 1
    assert loss > 0
    assert 0.0 <= metrics["accuracy"] <= 1.0
