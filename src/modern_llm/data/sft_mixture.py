"""Multi-dataset SFT mixture via HF `interleave_datasets`.

Normalizes heterogeneous instruction-tuning datasets to a common
`{instruction, input, output}` schema, then interleaves them with
configurable sampling probabilities. The interleaved rows are
tokenized + response-masked by reusing `InstructionDataset._tokenize`
so downstream `Trainer` code is unchanged.

Supported source schemas (one adapter per family):
    1. Messages list (OpenAI chat format):
         tulu-3-sft-mixture, smoltalk, no_robots
         row = {"messages": [{"role", "content"}, ...]}
    2. ShareGPT conversations list:
         OpenHermes-2.5
         row = {"conversations": [{"from", "value"}, ...]}
    3. Direct prompt/response pairs:
         MetaMathQA ({query, response}), OpenMathInstruct-2
         ({problem, generated_solution}), ifeval-like-data
         ({prompt, response}), Alpaca ({instruction, input, output})
    4. Multi-turn QA over a passage:
         stanfordnlp/coqa — collapsed to (story + first question) -> first answer.
         Rationale: full multi-turn unrolling would require a separate
         dataset schema; collapsing preserves signal without dropping the
         source (see docs in `_adapt_coqa`).

Why interleave (not concat): Concatenation makes sampling a function of
raw-dataset size, which for SFT is undesirable (e.g. OpenHermes-2.5
would dominate tulu-3 by 3x). Interleave + explicit probabilities gives
us data-mixture control identical to the pretraining recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from modern_llm.data.instruction_datasets import (
    InstructionDatasetConfig,
    format_instruction,
)
from modern_llm.utils.paths import cache_dir_for_datasets


# Runtime loader config registry. Keep this aligned with
# scripts/data/download_sft_mix.py and scripts/data/tokenize_sft.py.
_DATASET_CONFIG_REGISTRY: dict[str, str] = {
    "HuggingFaceTB/smoltalk": "all",
}


# --------------------------------------------------------------------- #
# Per-dataset schema adapters: raw HF row -> (instruction, input, output)
# Returns None if the row is unusable (empty, missing fields) and should
# be skipped.
# --------------------------------------------------------------------- #

def _last_user_assistant_pair(messages: list) -> Optional[tuple[str, str]]:
    """Extract the final (user, assistant) turn from a chat-format messages list.

    Upstream chat datasets vary in turn count; we train on the final
    assistant response conditioned on the full user prompt (concatenated
    prior turns if any). This matches standard SFT-on-final-turn practice.
    """
    if not messages:
        return None
    # Find last assistant turn.
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break
    if last_assistant_idx is None or last_assistant_idx == 0:
        return None
    assistant = messages[last_assistant_idx].get("content", "")
    # User prompt is the concatenation of all prior non-assistant turns.
    user_parts = []
    for msg in messages[:last_assistant_idx]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue
        if role == "system":
            user_parts.append(f"[System] {content}")
        else:
            user_parts.append(content)
    user = "\n\n".join(user_parts).strip()
    if not user or not assistant:
        return None
    return user, assistant


def _adapt_messages(row: dict) -> Optional[dict]:
    """tulu-3-sft-mixture, smoltalk, no_robots (OpenAI chat format)."""
    pair = _last_user_assistant_pair(row.get("messages") or [])
    if pair is None:
        return None
    user, assistant = pair
    return {"instruction": user, "input": "", "output": assistant}


def _adapt_sharegpt(row: dict) -> Optional[dict]:
    """OpenHermes-2.5 (ShareGPT `conversations` list)."""
    convs = row.get("conversations") or []
    # ShareGPT uses "from" in {"human", "gpt", "system"} and "value".
    messages = []
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    for turn in convs:
        role = role_map.get(turn.get("from", ""), "user")
        messages.append({"role": role, "content": turn.get("value", "")})
    pair = _last_user_assistant_pair(messages)
    if pair is None:
        return None
    user, assistant = pair
    return {"instruction": user, "input": "", "output": assistant}


def _adapt_alpaca(row: dict) -> Optional[dict]:
    """Alpaca-style: {instruction, input, output} already."""
    instr = row.get("instruction")
    out = row.get("output")
    if not instr or not out:
        return None
    return {"instruction": instr, "input": row.get("input", "") or "", "output": out}


def _adapt_prompt_response(row: dict) -> Optional[dict]:
    """ifeval-like-data, and fallback for any (prompt, response) schema."""
    p = row.get("prompt") or row.get("query") or row.get("question")
    r = row.get("response") or row.get("answer")
    if not p or not r:
        return None
    return {"instruction": p, "input": "", "output": r}


def _adapt_metamathqa(row: dict) -> Optional[dict]:
    """meta-math/MetaMathQA: {query, response}."""
    q, r = row.get("query"), row.get("response")
    if not q or not r:
        return None
    return {"instruction": q, "input": "", "output": r}


def _adapt_openmathinstruct2(row: dict) -> Optional[dict]:
    """nvidia/OpenMathInstruct-2: {problem, generated_solution, expected_answer}.

    We train on the worked solution (generated_solution) since that is
    the chain-of-thought signal we want the model to emit.
    """
    p = row.get("problem")
    s = row.get("generated_solution") or row.get("expected_answer")
    if not p or not s:
        return None
    return {"instruction": p, "input": "", "output": s}


def _adapt_coqa(row: dict) -> Optional[dict]:
    """stanfordnlp/coqa: {story, questions: [...], answers: {input_text: [...]}}.

    CoQA is multi-turn conversational QA over a passage. We collapse to:
        instruction = first question
        input       = story passage
        output      = first ground-truth answer
    Multi-turn expansion (one row per turn with accumulating dialogue
    context) would require changing the dataset cardinality and is out
    of scope for this mixture loader. This collapsed form still yields
    passage-grounded QA signal, which is the primary reason to include
    CoQA in an SFT mix.
    """
    story = row.get("story")
    questions = row.get("questions") or []
    answers = row.get("answers") or {}
    ans_texts = answers.get("input_text") if isinstance(answers, dict) else None
    if not story or not questions or not ans_texts:
        return None
    q0 = questions[0] if isinstance(questions, list) else None
    a0 = ans_texts[0] if isinstance(ans_texts, list) and ans_texts else None
    if not q0 or not a0:
        return None
    return {"instruction": q0, "input": story, "output": a0}


# Registry: dataset name prefix -> adapter. Matched by substring so we can
# handle split specs like "tulu-3-sft-mixture:train" without being brittle.
_ADAPTER_REGISTRY: list[tuple[str, Callable[[dict], Optional[dict]]]] = [
    ("tulu-3-sft-mixture", _adapt_messages),
    ("smoltalk", _adapt_messages),
    ("no_robots", _adapt_messages),
    ("OpenHermes-2.5", _adapt_sharegpt),
    ("MetaMathQA", _adapt_metamathqa),
    ("OpenMathInstruct-2", _adapt_openmathinstruct2),
    ("ifeval-like-data", _adapt_prompt_response),
    ("coqa", _adapt_coqa),
    ("alpaca", _adapt_alpaca),
]


def get_adapter(dataset_name: str) -> Callable[[dict], Optional[dict]]:
    """Return the schema adapter for a given HF dataset name.

    Pre: dataset_name is a recognised entry from _ADAPTER_REGISTRY, or a
    generic prompt/response schema.
    Post: Returns a callable row -> normalized-dict-or-None.
    """
    for key, adapter in _ADAPTER_REGISTRY:
        if key.lower() in dataset_name.lower():
            return adapter
    # Fallback: try alpaca-style first, then prompt/response. The adapter
    # returns None for rows it can't handle, so the caller will skip.
    def _fallback(row: dict) -> Optional[dict]:
        out = _adapt_alpaca(row)
        if out is not None:
            return out
        return _adapt_prompt_response(row)
    return _fallback


# --------------------------------------------------------------------- #
# Mixture loader
# --------------------------------------------------------------------- #

@dataclass
class _MixtureExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class SFTMixtureDataset(Dataset):
    """Torch wrapper over an interleaved HF dataset.

    Holds the already-interleaved raw HF dataset and lazily applies
    (per-dataset adapter -> format -> tokenize + mask) on __getitem__.
    Rows that adapt-to-None are rare in practice, but to keep __len__
    honest we pre-filter at construction time.
    """

    def __init__(
        self,
        rows: list[dict],
        adapters: list[Callable[[dict], Optional[dict]]],
        source_ids: list[int],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Precompute normalized triples; skip None.
        self._examples: list[dict] = []
        for row, src_id in zip(rows, source_ids):
            adapter = adapters[src_id]
            norm = adapter(row)
            if norm is None:
                continue
            self._examples.append(norm)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict:
        norm = self._examples[idx]
        text = format_instruction(norm["instruction"], norm["input"], norm["output"])
        return _tokenize_with_response_mask(text, self.tokenizer, self.max_length)


def _tokenize_with_response_mask(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict:
    """Tokenize and mask prompt tokens with -100 (response-only loss).

    Duplicates logic from InstructionDataset._tokenize so we don't have
    to instantiate an InstructionDataset per row. Kept in sync by
    delegating the marker convention.
    """
    response_marker = "### Response:\n"
    response_start = text.find(response_marker)

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].squeeze(0)
    attention_mask = tokens["attention_mask"].squeeze(0)
    labels = input_ids.clone()

    if response_start != -1:
        prompt_text = text[: response_start + len(response_marker)]
        prompt_tokens = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        prompt_len = int(prompt_tokens["attention_mask"].sum().item())
        labels[:prompt_len] = -100

    labels[attention_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _normalize_weights(
    weights: Optional[list[float]],
    n: int,
) -> list[float]:
    """Validate and normalize interleave weights to sum to 1.0.

    When weights is None, returns uniform [1/n] * n.
    """
    if weights is None:
        return [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(
            f"sft_dataset_weights length ({len(weights)}) must match "
            f"sft_datasets length ({n})"
        )
    if any(w < 0 for w in weights):
        raise ValueError("sft_dataset_weights must be non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError("sft_dataset_weights must sum to a positive value")
    return [w / total for w in weights]


def _resolve_dataset_load_args(dataset_name: str) -> tuple[str, Optional[str]]:
    """Resolve a configured SFT dataset into HF path + optional config name."""

    if "::" in dataset_name:
        name, config = dataset_name.split("::", 1)
        return name, config or None

    if ":" in dataset_name:
        name, config = dataset_name.rsplit(":", 1)
        if name and config:
            return name, config

    return dataset_name, _DATASET_CONFIG_REGISTRY.get(dataset_name)


def _load_raw(dataset_name: str, split: str = "train"):
    """Import-gated HF dataset load; kept as a seam for testing."""
    from datasets import load_dataset

    hf_name, hf_config = _resolve_dataset_load_args(dataset_name)
    return load_dataset(
        hf_name,
        hf_config,
        split=split,
        cache_dir=cache_dir_for_datasets(),
    )


def build_sft_mixture(
    dataset_names: list[str],
    weights: Optional[list[float]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    seed: int = 42,
    split: str = "train",
    num_examples_per_dataset: Optional[int] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    _loader: Optional[Callable[[str, str], object]] = None,
) -> SFTMixtureDataset:
    """Load + normalize + interleave multiple SFT datasets.

    Pre:
        - dataset_names is a non-empty list of HF Hub IDs with registered
          adapters (see _ADAPTER_REGISTRY).
        - weights is None or same length as dataset_names (will be
          normalized to sum to 1.0).
    Post:
        - Returns a torch Dataset over the interleaved rows with
          per-dataset schema adapters applied.
        - Interleave uses stopping_strategy="all_exhausted" so smaller
          sources are upsampled (re-shuffled and replayed) until every
          source has been exhausted at least once — this matches the
          "mixture fidelity" interpretation of the weights.

    The `_loader` seam lets tests inject an in-memory Dataset factory
    without hitting the network.
    """
    from datasets import interleave_datasets

    if not dataset_names:
        raise ValueError("dataset_names must be non-empty")

    probs = _normalize_weights(weights, len(dataset_names))
    loader = _loader if _loader is not None else _load_raw

    raw_datasets = []
    adapters: list[Callable[[dict], Optional[dict]]] = []
    for name in dataset_names:
        load_split = split
        if num_examples_per_dataset is not None and _loader is None:
            load_split = f"{split}[:{num_examples_per_dataset}]"
        if log_fn is not None:
            log_fn(f"Loading SFT source {name} ({load_split})")
        ds = loader(name, load_split)
        if num_examples_per_dataset is not None:
            ds = ds.select(range(min(num_examples_per_dataset, len(ds))))
        if log_fn is not None:
            log_fn(f"Loaded SFT source {name}: {len(ds)} examples")
        # Tag rows with their source index so we can pick the right adapter
        # after interleave (which loses per-row provenance otherwise).
        raw_datasets.append(ds)
        adapters.append(get_adapter(name))

    # Add a `_source_id` column before interleaving.
    tagged = []
    for src_id, ds in enumerate(raw_datasets):
        tagged.append(ds.add_column("_source_id", [src_id] * len(ds)))

    interleaved = interleave_datasets(
        tagged,
        probabilities=probs,
        seed=seed,
        stopping_strategy="all_exhausted",
    )

    # Materialize — these are SFT-scale datasets; we want a fixed-length
    # torch Dataset for Trainer. Streaming would only matter if the
    # mixture exceeded RAM which is not the case for the target configs.
    if log_fn is not None:
        log_fn("Materializing capped SFT mixture")
    rows = [dict(r) for r in interleaved]
    source_ids = [r.pop("_source_id") for r in rows]
    dataset = SFTMixtureDataset(
        rows=rows,
        adapters=adapters,
        source_ids=source_ids,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    if log_fn is not None:
        log_fn(f"Built SFT mixture: {len(dataset)} usable examples")
    return dataset
