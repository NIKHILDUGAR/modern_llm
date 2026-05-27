"""Regression checks for the packed-pretrain source mix."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "data" / "tokenize_pretrain.py"
_SPEC = importlib.util.spec_from_file_location("tokenize_pretrain_for_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_default_pretrain_mix_matches_75m_v2_balance() -> None:
    weights = {source.key: source.weight for source in _MODULE.DEFAULT_SOURCES}

    assert weights == pytest.approx(
        {
            "fineweb-edu": 0.55,
            "dclm": 0.20,
            "openwebmath": 0.07,
            "wikipedia": 0.10,
            "rp-stackexchange": 0.05,
            "rp-arxiv": 0.03,
        }
    )
    assert sum(weights.values()) == pytest.approx(1.0)
