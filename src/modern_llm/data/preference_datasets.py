"""Preference pair datasets for DPO or similar alignment procedures.

Typical sources include Anthropic HH-RLHF (Bai et al., 2022) or synthetic judge
labels derived from instruction-following corpora.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from modern_llm.utils.paths import apply_env_defaults, cache_dir_for_datasets

apply_env_defaults()


_DATASET_SPLIT_REGISTRY: dict[str, dict[str, str]] = {
    "huggingfaceh4/ultrafeedback_binarized": {
        "train": "train_prefs",
        "validation": "test_prefs",
        "eval": "test_prefs",
        "test": "test_prefs",
    },
}


@dataclass
class PreferenceDatasetConfig:
    """Configuration describing a pairwise preference dataset."""

    dataset_name: str
    dataset_config_name: Optional[str] = None
    split: str = "train"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    prompt_field: Optional[str] = None  # None for datasets that embed prompt in responses

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name is required for preference loading.")
        if self.chosen_field == self.rejected_field:
            raise ValueError("chosen_field and rejected_field must differ.")


def _require_datasets():
    try:
        from datasets import load_dataset  # type: ignore

        return load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required. Install it with `pip install datasets`."
        ) from exc


def _resolve_preference_load_args(
    config: PreferenceDatasetConfig,
) -> tuple[str, Optional[str], str]:
    """Resolve runtime HF load args for known preference datasets."""
    dataset_name = config.dataset_name
    split = config.split
    split_map = _DATASET_SPLIT_REGISTRY.get(dataset_name.lower())
    if split_map is not None:
        split = split_map.get(split, split)
    return dataset_name, config.dataset_config_name, split


def _extract_prompt_and_response_hh(text: str) -> tuple[str, str]:
    """Extract prompt and response from Anthropic HH-RLHF format.
    
    The format is:
        Human: <prompt>
        
        Assistant: <response>
        
        Human: <follow-up> (optional)
        ...
    
    Returns the full conversation up to the last Assistant turn as prompt,
    and the last Assistant response as the response.
    """
    # Find the last "Assistant:" marker
    matches = list(re.finditer(r"\n\nAssistant:\s*", text))
    if not matches:
        # No assistant marker, treat whole thing as prompt with empty response
        return text.strip(), ""
    
    last_match = matches[-1]
    prompt = text[:last_match.start()].strip()
    response = text[last_match.end():].strip()
    return prompt, response


def _process_hh_rlhf(dataset):
    """Process Anthropic HH-RLHF dataset to extract prompt/chosen/rejected."""
    
    def extract_fields(example):
        chosen_text = example.get("chosen", "")
        rejected_text = example.get("rejected", "")
        
        prompt_from_chosen, chosen_response = _extract_prompt_and_response_hh(chosen_text)
        prompt_from_rejected, rejected_response = _extract_prompt_and_response_hh(rejected_text)
        
        # Use prompt from chosen (they should be the same)
        return {
            "prompt": prompt_from_chosen,
            "chosen": chosen_response,
            "rejected": rejected_response,
        }
    
    return dataset.map(extract_fields, remove_columns=dataset.column_names)


def load_preference_dataset(config: PreferenceDatasetConfig):
    """Return a dataset that yields prompt, chosen, rejected triples.

    Pre:
        - dataset exposes the configured fields.
    Post:
        - returns Hugging Face Dataset ready for DPO batching with prompt/chosen/rejected.
    Complexity:
        - O(num_examples) to scan column names and instantiate the dataset.
    Invariants:
        - Each row keeps prompt/response ordering intact.
    """

    load_dataset = _require_datasets()
    dataset_name, dataset_config_name, split = _resolve_preference_load_args(config)
    dataset = load_dataset(
        dataset_name,
        dataset_config_name,
        split=split,
        cache_dir=cache_dir_for_datasets(),
    )
    column_names = dataset.column_names
    
    # Check required fields
    for field in [config.chosen_field, config.rejected_field]:
        if field not in column_names:
            raise ValueError(f"Field '{field}' missing from dataset columns: {column_names}")
    
    # Handle datasets without explicit prompt field (like Anthropic HH-RLHF)
    if config.prompt_field is None or config.prompt_field not in column_names:
        # Check if this looks like HH-RLHF format (chosen/rejected contain full conversations)
        if "Anthropic" in config.dataset_name or "hh-rlhf" in config.dataset_name.lower():
            return _process_hh_rlhf(dataset)
        # For other datasets, try to extract or use chosen field as-is
        if "prompt" not in column_names:
            # Create empty prompt field
            dataset = dataset.map(lambda x: {"prompt": ""})
    
    return dataset
