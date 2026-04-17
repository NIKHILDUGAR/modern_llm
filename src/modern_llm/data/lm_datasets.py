"""Causal language modeling data prep (e.g., WikiText-2, TinyStories, OpenWebText).

WikiText-2 (Merity et al., 2016) and TinyStories (Gao et al., 2023) are the
primary corpora; this module standardizes how we fetch and tokenize them so the
training scripts can assume reproducible, research-grade preprocessing.

DDP notes
---------
- For map-style tokenized datasets (the path used by `load_causal_lm_dataset`),
  callers should construct dataloaders via `make_lm_dataloader` below, which
  attaches a `DistributedSampler` when `WORLD_SIZE > 1` and otherwise behaves
  exactly like the previous single-process loader.
- For HF streaming datasets, use `datasets.distributed.split_dataset_by_node`
  so each rank consumes a disjoint stream shard.
- All `load_dataset` calls route through `cache_dir=` so the data lands in
  `data/raw/hf_cache/` (configurable via `HF_DATASETS_CACHE`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from modern_llm.training.distributed import (
    is_distributed,
    maybe_distributed_sampler,
    split_iterable_by_rank,
    world_size,
)
from modern_llm.utils.paths import apply_env_defaults, cache_dir_for_datasets


# Apply env defaults at import time so any subsequent `from datasets import ...`
# inside this process picks up the correct HF_DATASETS_CACHE.
apply_env_defaults()


# Dataset name -> (hf_name, hf_config, text_field) mapping
DATASET_REGISTRY = {
    "wikitext-2-raw-v1": ("wikitext", "wikitext-2-raw-v1", "text"),
    "wikitext-103-raw-v1": ("wikitext", "wikitext-103-raw-v1", "text"),
    "roneneldan/TinyStories": ("roneneldan/TinyStories", None, "text"),
    "openwebtext": ("Skylion007/openwebtext", None, "text"),
    "wikipedia": ("wikimedia/wikipedia", "20231101.en", "text"),
}


@dataclass
class LanguageModelingDatasetConfig:
    """Configure a Hugging Face dataset for causal LM use."""

    dataset_name: str
    dataset_config_name: Optional[str] = None
    split: str = "train"
    text_field: str = "text"
    max_length: int = 1024
    num_proc: Optional[int] = None
    streaming: bool = False
    cache_dir: Optional[str] = None  # Defaults to HF_DATASETS_CACHE / data/raw/hf_cache

    def __post_init__(self) -> None:
        if not self.dataset_name:
            raise ValueError("dataset_name must be a non-empty string")
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, received {self.max_length}")


def _require_datasets():
    try:
        from datasets import load_dataset  # type: ignore

        return load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required. Install it with `pip install datasets`."
        ) from exc


def load_causal_lm_dataset(
    config: LanguageModelingDatasetConfig,
    tokenizer: PreTrainedTokenizerBase,
):
    """Load and tokenize a dataset for causal language modeling.

    Pre:
        - `tokenizer` is a causal LM tokenizer with `pad_token_id` defined.
        - the Hugging Face dataset specified in `config` is reachable.
    Post:
        - returns a tokenized `datasets.Dataset` (map-style) with `input_ids`,
          `attention_mask`, and `labels` columns. For streaming datasets, returns
          a tokenized `IterableDataset` already sharded by rank when distributed.
    Complexity:
        - O(num_examples · max_length) due to tokenization work.
    """
    load_dataset = _require_datasets()
    cache_dir = config.cache_dir or cache_dir_for_datasets()
    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config_name,
        split=config.split,
        streaming=config.streaming,
        cache_dir=cache_dir,
    )

    if config.streaming:
        # Shard the stream so each DDP rank sees a disjoint subset before
        # tokenizing — this avoids redundant compute across ranks.
        if is_distributed():
            dataset = split_iterable_by_rank(dataset)

        def _stream_tokenize(example):
            text = example[config.text_field]
            outputs = tokenizer(
                text,
                truncation=True,
                max_length=config.max_length,
                padding="max_length",
                return_tensors=None,
            )
            mask = outputs["attention_mask"]
            ids = outputs["input_ids"]
            labels = [tok if mask[idx] == 1 else -100 for idx, tok in enumerate(ids)]
            return {
                "input_ids": ids,
                "attention_mask": mask,
                "labels": labels,
            }

        return dataset.map(_stream_tokenize, remove_columns=dataset.column_names if dataset.column_names else None)

    column_names = dataset.column_names
    if config.text_field not in column_names:
        raise ValueError(
            f"text_field '{config.text_field}' not present in dataset columns: {column_names}"
        )

    def _tokenize(batch):
        texts = batch[config.text_field]
        outputs = tokenizer(
            texts,
            truncation=True,
            max_length=config.max_length,
            padding="max_length",
            return_tensors=None,
        )
        labels = []
        for ids, mask in zip(outputs["input_ids"], outputs["attention_mask"]):
            label_row = [token if mask[idx] == 1 else -100 for idx, token in enumerate(ids)]
            labels.append(label_row)
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
            "labels": labels,
        }

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=column_names,
        num_proc=config.num_proc,
        desc=f"Tokenizing {config.dataset_name}",
    )

    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def make_lm_dataloader(
    dataset,
    micro_batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    seed: int = 42,
) -> DataLoader:
    """Build a DataLoader that is correct under DDP.

    Pre: `dataset` is a map-style dataset (HF Dataset or torch Dataset). For
         iterable/streaming datasets, do NOT use this — just wrap with
         `DataLoader(ds, batch_size=...)` since ranks are pre-sharded.
    Post: returns a DataLoader; uses DistributedSampler when WORLD_SIZE > 1.
    """
    sampler = maybe_distributed_sampler(dataset, shuffle=shuffle, seed=seed, drop_last=drop_last)
    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        # When a sampler is used, shuffle must be False.
        shuffle=False if sampler is not None else shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def _parse_dataset_spec(spec: str) -> tuple:
    """Parse dataset spec like 'name:100000' into (name, max_samples).

    Examples:
        'wikitext-103-raw-v1' -> ('wikitext-103-raw-v1', None)
        'roneneldan/TinyStories:100000' -> ('roneneldan/TinyStories', 100000)
    """
    if ":" in spec:
        parts = spec.rsplit(":", 1)
        name = parts[0]
        try:
            max_samples = int(parts[1])
        except ValueError:
            return spec, None
        return name, max_samples
    return spec, None


def load_multi_dataset(
    dataset_names: List[str],
    tokenizer: PreTrainedTokenizerBase,
    split: str = "train",
    max_length: int = 1024,
    max_samples_per_dataset: Optional[int] = None,
):
    """Load and concatenate multiple datasets for pretraining.

    Pre: dataset_names are keys in DATASET_REGISTRY or valid HF dataset paths.
         Supports 'name:N' syntax to cap individual datasets (e.g. 'TinyStories:100000').
    Post: Returns concatenated tokenized dataset.
    """
    from datasets import concatenate_datasets

    all_datasets = []

    for spec in dataset_names:
        name, per_dataset_cap = _parse_dataset_spec(spec)
        print(f"Loading dataset: {name}" + (f" (capped to {per_dataset_cap})" if per_dataset_cap else ""))

        # Look up in registry or use as-is
        if name in DATASET_REGISTRY:
            hf_name, hf_config, text_field = DATASET_REGISTRY[name]
        else:
            hf_name = name
            hf_config = None
            text_field = "text"

        try:
            config = LanguageModelingDatasetConfig(
                dataset_name=hf_name,
                dataset_config_name=hf_config,
                split=split,
                text_field=text_field,
                max_length=max_length,
            )
            dataset = load_causal_lm_dataset(config, tokenizer)

            cap = per_dataset_cap or max_samples_per_dataset
            if cap and len(dataset) > cap:
                dataset = dataset.select(range(cap))
                print(f"  Capped to {cap} samples")

            print(f"  Loaded {len(dataset)} samples from {name}")
            all_datasets.append(dataset)

        except Exception as e:
            print(f"  WARNING: Failed to load {name}: {e}")
            continue

    if not all_datasets:
        raise ValueError("No datasets were successfully loaded")

    combined = concatenate_datasets(all_datasets)
    combined = combined.shuffle(seed=42)
    combined.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    print(f"Combined dataset: {len(combined)} total samples")

    return combined
