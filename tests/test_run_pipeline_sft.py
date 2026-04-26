"""Tests for SFT-stage orchestration helpers in scripts/run_pipeline.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from modern_llm.config import PipelineConfig


_RUN_PIPELINE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"
_SPEC = importlib.util.spec_from_file_location("run_pipeline_for_test", _RUN_PIPELINE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_RUN_PIPELINE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_RUN_PIPELINE)
_infer_sft_examples_per_dataset = _RUN_PIPELINE._infer_sft_examples_per_dataset


def test_infer_sft_examples_per_dataset_uses_requested_training_budget() -> None:
    cfg = PipelineConfig(
        sft_max_steps=4000,
        sft_batch_size=32,
        sft_datasets=[f"dataset-{i}" for i in range(8)],
    )

    assert _infer_sft_examples_per_dataset(cfg, dataset_count=8) == 16000


def test_infer_sft_examples_per_dataset_respects_explicit_cap() -> None:
    cfg = PipelineConfig(sft_num_examples_per_dataset=4096)

    assert _infer_sft_examples_per_dataset(cfg, dataset_count=8) == 4096


def test_infer_sft_examples_per_dataset_rejects_invalid_explicit_cap() -> None:
    cfg = PipelineConfig(sft_num_examples_per_dataset=0)

    with pytest.raises(ValueError, match="must be positive"):
        _infer_sft_examples_per_dataset(cfg, dataset_count=8)
