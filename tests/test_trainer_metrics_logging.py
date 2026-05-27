"""Tests for step-wise trainer metric logging."""

from __future__ import annotations

import json
import math

from modern_llm.training.trainer_base import Trainer


def test_record_metrics_writes_jsonl(tmp_path) -> None:
    trainer = object.__new__(Trainer)
    trainer.metrics_path = tmp_path / "run_metrics.jsonl"

    trainer._record_metrics(
        {
            "phase": "train",
            "step": 1,
            "loss": 2.0,
            "perplexity": Trainer._loss_to_perplexity(2.0),
            "lr": 1e-4,
        }
    )

    rows = [json.loads(line) for line in trainer.metrics_path.read_text().splitlines()]
    assert rows == [
        {
            "phase": "train",
            "step": 1,
            "loss": 2.0,
            "perplexity": math.exp(2.0),
            "lr": 1e-4,
        }
    ]


def test_loss_to_perplexity_caps_unstable_values() -> None:
    assert math.isinf(Trainer._loss_to_perplexity(20.0))
    assert math.isinf(Trainer._loss_to_perplexity(float("nan")))
