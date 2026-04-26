"""Tests for DPO preference dataset loading and normalization."""

from __future__ import annotations

from datasets import Dataset

from modern_llm.data import preference_datasets
from modern_llm.data.preference_datasets import (
    PreferenceDatasetConfig,
    _resolve_preference_load_args,
    load_preference_dataset,
)
from modern_llm.training.train_dpo import PreferenceDataset


def test_ultrafeedback_default_split_resolves_to_train_prefs() -> None:
    cfg = PreferenceDatasetConfig(dataset_name="HuggingFaceH4/ultrafeedback_binarized")

    assert _resolve_preference_load_args(cfg) == (
        "HuggingFaceH4/ultrafeedback_binarized",
        None,
        "train_prefs",
    )


def test_ultrafeedback_eval_split_resolves_to_test_prefs() -> None:
    cfg = PreferenceDatasetConfig(
        dataset_name="HuggingFaceH4/ultrafeedback_binarized",
        split="test",
    )

    assert _resolve_preference_load_args(cfg) == (
        "HuggingFaceH4/ultrafeedback_binarized",
        None,
        "test_prefs",
    )


def test_load_preference_dataset_uses_resolved_split(monkeypatch) -> None:
    captured = {}

    def fake_load_dataset(name, config_name, split, cache_dir):
        captured.update(
            {
                "name": name,
                "config_name": config_name,
                "split": split,
                "cache_dir": cache_dir,
            }
        )
        return Dataset.from_dict(
            {
                "prompt": ["p"],
                "chosen": ["good"],
                "rejected": ["bad"],
            }
        )

    monkeypatch.setattr(preference_datasets, "_require_datasets", lambda: fake_load_dataset)

    ds = load_preference_dataset(
        PreferenceDatasetConfig(dataset_name="HuggingFaceH4/ultrafeedback_binarized")
    )

    assert captured["split"] == "train_prefs"
    assert len(ds) == 1


def test_chat_preference_response_only_extracts_assistant_turn() -> None:
    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]

    assert PreferenceDataset._coerce_text(messages, response_only=True) == "answer"
