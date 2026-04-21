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

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
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


class PackedShardDataset(Dataset):
    """Map-style causal-LM dataset over packed uint32 token shards.

    Expects `<data_dir>/index.json` + `<data_dir>/shard_*.bin` produced by
    `scripts/data/tokenize_pretrain.py`. The shards are conceptually
    concatenated into one flat token stream and split into non-overlapping
    windows of length `seq_len`. Windows that straddle shard boundaries are
    spliced together transparently.

    `__getitem__` returns `{input_ids, attention_mask, labels}` with
    `labels == input_ids` (the model's forward handles the shift). This
    matches the existing HF map-style dataset interface so the rest of the
    trainer (DistributedSampler, DataLoader, CE loss, z-loss) is unchanged.

    Uses `np.memmap` (zero-copy page-cache reads). Memmaps are opened
    lazily on first access inside each worker process to avoid FD storms
    when `num_workers` is large.
    """

    def __init__(
        self,
        data_dir: str | Path,
        seq_len: int,
        window_range: Optional[tuple] = None,
    ) -> None:
        data_dir = Path(data_dir)
        idx_path = data_dir / "index.json"
        if not idx_path.exists():
            raise FileNotFoundError(f"No index.json at {idx_path}")
        idx = json.loads(idx_path.read_text())
        if idx.get("dtype") != "uint32":
            raise ValueError(f"PackedShardDataset requires uint32 shards, got {idx.get('dtype')}")
        shards = idx.get("shards") or []
        if not shards:
            raise ValueError(f"index.json at {idx_path} lists no shards")
        self.seq_len = int(seq_len)
        self._paths: List[Path] = [data_dir / s["path"] for s in shards]
        self._sizes: List[int] = [int(s["tokens"]) for s in shards]
        self._mmaps: List[Optional[np.memmap]] = [None] * len(self._paths)
        # _cum[i] = total tokens in shards [0..i). Last entry = grand total.
        self._cum = np.zeros(len(self._paths) + 1, dtype=np.int64)
        self._cum[1:] = np.cumsum(self._sizes)
        self.total_tokens = int(self._cum[-1])
        total_windows = self.total_tokens // self.seq_len
        if total_windows <= 0:
            raise ValueError(
                f"Not enough tokens ({self.total_tokens}) for seq_len={self.seq_len}"
            )

        if window_range is None:
            self._start_window = 0
            self._end_window = total_windows
        else:
            start, end = int(window_range[0]), int(window_range[1])
            if not (0 <= start < end <= total_windows):
                raise ValueError(
                    f"window_range {window_range} out of bounds for {total_windows} windows"
                )
            self._start_window = start
            self._end_window = end
        self.num_windows = self._end_window - self._start_window

    def __len__(self) -> int:
        return self.num_windows

    def _get_shard(self, i: int) -> np.memmap:
        m = self._mmaps[i]
        if m is None:
            m = np.memmap(self._paths[i], dtype=np.uint32, mode="r")
            # Trust shard size from index (verified by size check below).
            if m.size != self._sizes[i]:
                raise RuntimeError(
                    f"shard {self._paths[i]} size mismatch: "
                    f"mmap={m.size} index={self._sizes[i]}"
                )
            self._mmaps[i] = m
        return m

    def _read_flat(self, start: int, n: int) -> np.ndarray:
        shard_idx = int(np.searchsorted(self._cum, start, side="right") - 1)
        local = start - int(self._cum[shard_idx])
        pieces: List[np.ndarray] = []
        remaining = n
        while remaining > 0:
            m = self._get_shard(shard_idx)
            take = min(int(m.size) - local, remaining)
            # Copy out of the memmap so the tensor owns its buffer — avoids
            # use-after-close issues if the mmap is reopened.
            pieces.append(np.asarray(m[local : local + take], dtype=np.uint32).copy())
            remaining -= take
            shard_idx += 1
            local = 0
        return np.concatenate(pieces) if len(pieces) > 1 else pieces[0]

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= self.num_windows:
            raise IndexError(idx)
        start = (self._start_window + int(idx)) * self.seq_len
        arr = self._read_flat(start, self.seq_len)
        # Model embeddings use int64 indices.
        ids = torch.from_numpy(arr.astype(np.int64, copy=False))
        return {
            "input_ids": ids,
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": ids.clone(),
        }


def load_packed_pretrain_dataset(data_dir: str | Path, seq_len: int) -> PackedShardDataset:
    """Build a PackedShardDataset over a tokenize_pretrain.py output dir."""
    ds = PackedShardDataset(data_dir, seq_len=seq_len)
    print(
        f"[packed-shards] {data_dir}: {len(ds._paths)} shards, "
        f"{ds.total_tokens:,} tokens, {len(ds):,} windows of {seq_len}"
    )
    return ds


def load_packed_pretrain_train_eval_split(
    data_dir: str | Path,
    seq_len: int,
    eval_windows: int,
) -> tuple:
    """Split packed shards into (train, eval) views.

    The last `eval_windows` windows of the concatenated token stream become the
    eval set; training sees the preceding prefix. Eval sits at the tail of the
    corpus so train windows start at offset 0, matching prior behavior.

    Pre: eval_windows > 0 and strictly less than total_windows.
    Post: returns (train_ds, eval_ds) — both PackedShardDataset instances
          backed by the same mmaps but disjoint window indices.
    """
    probe = PackedShardDataset(data_dir, seq_len=seq_len)
    total = probe.num_windows
    if eval_windows <= 0 or eval_windows >= total:
        raise ValueError(
            f"eval_windows={eval_windows} invalid for total_windows={total}"
        )
    split_at = total - eval_windows
    train_ds = PackedShardDataset(
        data_dir, seq_len=seq_len, window_range=(0, split_at)
    )
    eval_ds = PackedShardDataset(
        data_dir, seq_len=seq_len, window_range=(split_at, total)
    )
    print(
        f"[packed-shards-split] {data_dir}: train={len(train_ds):,} windows, "
        f"eval={len(eval_ds):,} windows (seq_len={seq_len}, "
        f"eval tokens={len(eval_ds) * seq_len:,})"
    )
    return train_ds, eval_ds
