"""Tests for src/modern_llm/data/sft_mixture.py.

We inject an in-memory loader via the `_loader` seam so tests never
hit the HF Hub. Schema-adapter correctness is verified per-dataset;
interleave weighting is verified by checking empirical source counts
against the configured probabilities (within tolerance).
"""

from __future__ import annotations

import pytest

from datasets import Dataset

from modern_llm.data.sft_mixture import (
    _normalize_weights,
    _resolve_dataset_load_args,
    build_sft_mixture,
    get_adapter,
)


class _DummyTokenizer:
    """Character-level tokenizer good enough for masking/tensor tests.

    Produces deterministic integer ids and honors max_length/padding.
    Returns a HF-style BatchEncoding-like dict (has .squeeze via tensor).
    """

    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1

    def __call__(
        self,
        text,
        truncation: bool = False,
        max_length: int | None = None,
        padding: str | bool = False,
        return_tensors: str | None = None,
    ):
        import torch

        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)

        all_ids = []
        all_mask = []
        for t in texts:
            ids = [2 + (ord(c) % 30) for c in t]
            if truncation and max_length is not None:
                ids = ids[:max_length]
            mask = [1] * len(ids)
            if padding == "max_length" and max_length is not None:
                pad_n = max_length - len(ids)
                if pad_n > 0:
                    ids = ids + [self.pad_token_id] * pad_n
                    mask = mask + [0] * pad_n
            all_ids.append(ids)
            all_mask.append(mask)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(all_ids, dtype=torch.long),
                "attention_mask": torch.tensor(all_mask, dtype=torch.long),
            }
        return {"input_ids": all_ids, "attention_mask": all_mask}


# --------------------------------------------------------------------- #
# Unit tests for helpers
# --------------------------------------------------------------------- #

def test_normalize_weights_none_is_uniform() -> None:
    assert _normalize_weights(None, 4) == [0.25, 0.25, 0.25, 0.25]


def test_normalize_weights_sums_to_one() -> None:
    out = _normalize_weights([2.0, 3.0, 5.0], 3)
    assert sum(out) == pytest.approx(1.0)
    assert out == pytest.approx([0.2, 0.3, 0.5])


def test_normalize_weights_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must match"):
        _normalize_weights([0.5, 0.5], 3)


def test_normalize_weights_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _normalize_weights([-1.0, 2.0], 2)


def test_normalize_weights_rejects_zero_sum() -> None:
    with pytest.raises(ValueError, match="positive value"):
        _normalize_weights([0.0, 0.0], 2)


def test_resolve_dataset_load_args_uses_known_config() -> None:
    assert _resolve_dataset_load_args("HuggingFaceTB/smoltalk") == (
        "HuggingFaceTB/smoltalk",
        "all",
    )


def test_resolve_dataset_load_args_supports_inline_config() -> None:
    assert _resolve_dataset_load_args("org/name::config-a") == ("org/name", "config-a")
    assert _resolve_dataset_load_args("org/name:config-b") == ("org/name", "config-b")


def test_adapter_messages_extracts_final_turn() -> None:
    adapter = get_adapter("HuggingFaceTB/smoltalk")
    row = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
    }
    norm = adapter(row)
    assert norm is not None
    assert "2+2" in norm["instruction"]
    assert norm["output"] == "4"
    assert norm["input"] == ""


def test_adapter_sharegpt() -> None:
    adapter = get_adapter("teknium/OpenHermes-2.5")
    row = {
        "conversations": [
            {"from": "human", "value": "Define entropy"},
            {"from": "gpt", "value": "A measure of disorder."},
        ]
    }
    norm = adapter(row)
    assert norm == {"instruction": "Define entropy", "input": "", "output": "A measure of disorder."}


def test_adapter_metamathqa() -> None:
    adapter = get_adapter("meta-math/MetaMathQA")
    norm = adapter({"query": "solve x+1=2", "response": "x=1"})
    assert norm == {"instruction": "solve x+1=2", "input": "", "output": "x=1"}


def test_adapter_openmathinstruct2_prefers_solution() -> None:
    adapter = get_adapter("nvidia/OpenMathInstruct-2")
    norm = adapter(
        {"problem": "P", "generated_solution": "long CoT", "expected_answer": "42"}
    )
    assert norm["output"] == "long CoT"


def test_adapter_coqa_collapses_first_turn() -> None:
    adapter = get_adapter("stanfordnlp/coqa")
    norm = adapter(
        {
            "story": "A story about ants.",
            "questions": ["What do ants do?", "Follow-up?"],
            "answers": {"input_text": ["They work.", "Yes."]},
        }
    )
    assert norm == {
        "instruction": "What do ants do?",
        "input": "A story about ants.",
        "output": "They work.",
    }


def test_adapter_ifeval_prompt_response() -> None:
    adapter = get_adapter("argilla/ifeval-like-data")
    norm = adapter({"prompt": "List 3 colors", "response": "red green blue"})
    assert norm["instruction"] == "List 3 colors"


def test_adapter_returns_none_on_empty() -> None:
    adapter = get_adapter("HuggingFaceTB/smoltalk")
    assert adapter({"messages": []}) is None
    assert adapter({"messages": [{"role": "user", "content": "hi"}]}) is None


# --------------------------------------------------------------------- #
# End-to-end mixture test with injected loader (no network).
# --------------------------------------------------------------------- #

def _make_messages_ds(prefix: str, n: int) -> Dataset:
    return Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": f"{prefix}-q{i}"},
                    {"role": "assistant", "content": f"{prefix}-a{i}"},
                ]
                for i in range(n)
            ]
        }
    )


def test_build_sft_mixture_schema_and_length() -> None:
    ds_a = _make_messages_ds("A", 20)
    ds_b = _make_messages_ds("B", 20)
    fakes = {
        "HuggingFaceTB/smoltalk": ds_a,
        "HuggingFaceH4/no_robots": ds_b,
    }

    def fake_loader(name: str, split: str):
        return fakes[name]

    tok = _DummyTokenizer()
    mix = build_sft_mixture(
        dataset_names=list(fakes.keys()),
        weights=None,
        tokenizer=tok,
        max_length=64,
        seed=0,
        _loader=fake_loader,
    )

    # Every example must carry the three tensor keys with correct shape.
    assert len(mix) > 0
    sample = mix[0]
    assert set(sample.keys()) == {"input_ids", "attention_mask", "labels"}
    assert sample["input_ids"].shape == (64,)
    assert sample["attention_mask"].shape == (64,)
    assert sample["labels"].shape == (64,)
    # Labels must contain at least one masked position (-100).
    assert (sample["labels"] == -100).any().item()


def test_build_sft_mixture_respects_weights() -> None:
    """With weights [0.8, 0.2] and all_exhausted, the minority source
    gets upsampled until the majority is exhausted. We check that the
    empirical share of rows drawn from source A is substantially higher
    than from B — interleave uses the probabilities sample-by-sample.
    """
    ds_a = _make_messages_ds("A", 100)
    ds_b = _make_messages_ds("B", 100)
    fakes = {"alphA": ds_a, "bravO": ds_b}

    def fake_loader(name: str, split: str):
        return fakes[name]

    tok = _DummyTokenizer()
    # Use a custom adapter-free pass: we reuse messages adapter via the
    # fallback by naming keys that contain a registered substring.
    # "smoltalk" and "no_robots" both route to _adapt_messages.
    fakes = {"HuggingFaceTB/smoltalk": ds_a, "HuggingFaceH4/no_robots": ds_b}

    # We cannot directly observe provenance after adapter removes the
    # _source_id tag, so instead we inspect the interleaved raw rows by
    # patching the loader to return datasets we can identify via content.
    # The normalized "instruction" preserves the "A-qN" / "B-qN" prefix.
    mix = build_sft_mixture(
        dataset_names=list(fakes.keys()),
        weights=[0.8, 0.2],
        tokenizer=tok,
        max_length=64,
        seed=0,
        _loader=lambda name, split: fakes[name],
    )

    # Reach into the stored normalized examples (pre-tokenization).
    insts = [ex["instruction"] for ex in mix._examples]
    a_count = sum(1 for s in insts if s.startswith("A-"))
    b_count = sum(1 for s in insts if s.startswith("B-"))
    total = a_count + b_count
    assert total == len(insts) and total > 0
    a_frac = a_count / total
    # Expect ~0.8; allow wide tolerance since all_exhausted replays the
    # minority source which slightly pulls fractions toward 0.5.
    # The invariant we really care about is "A is the majority".
    assert a_frac > 0.6, f"A fraction {a_frac} too low; counts A={a_count} B={b_count}"


def test_build_sft_mixture_caps_each_source() -> None:
    ds_a = _make_messages_ds("A", 20)
    ds_b = _make_messages_ds("B", 20)
    fakes = {
        "HuggingFaceTB/smoltalk": ds_a,
        "HuggingFaceH4/no_robots": ds_b,
    }

    mix = build_sft_mixture(
        dataset_names=list(fakes.keys()),
        weights=None,
        tokenizer=_DummyTokenizer(),
        max_length=64,
        seed=0,
        num_examples_per_dataset=5,
        _loader=lambda name, split: fakes[name],
    )

    insts = [ex["instruction"] for ex in mix._examples]
    assert insts
    assert all(s.split("-q", 1)[1].isdigit() and int(s.split("-q", 1)[1]) < 5 for s in insts)


def test_build_sft_mixture_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_sft_mixture(
            dataset_names=[],
            weights=None,
            tokenizer=_DummyTokenizer(),
            max_length=32,
            _loader=lambda n, s: Dataset.from_dict({"messages": []}),
        )
