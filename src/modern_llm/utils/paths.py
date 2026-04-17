"""Repo-relative paths and HF cache redirection.

Single source of truth for where datasets, tokenizers, and the HF hub cache
live. Imported early at process start (see `apply_env_defaults()`); also
exposes `data_root()` so application code can pass `cache_dir=` explicitly
to `load_dataset` / `AutoTokenizer.from_pretrained` instead of relying only
on the env vars (which is brittle when subprocesses are spawned).
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the absolute path of the repo root (works from any cwd)."""
    return Path(__file__).resolve().parents[3]


def data_root() -> Path:
    """`<repo>/data/raw` — the canonical home for HF datasets cache."""
    return repo_root() / "data" / "raw"


def hf_home() -> Path:
    """`<repo>/data/raw/hf_home` — what HF_HOME points at by default."""
    return data_root() / "hf_home"


def hf_datasets_cache() -> Path:
    """`<repo>/data/raw/hf_cache` — what HF_DATASETS_CACHE points at."""
    return data_root() / "hf_cache"


def hf_hub_cache() -> Path:
    """`<repo>/data/raw/hf_home/hub` — model weights cache."""
    return hf_home() / "hub"


def tokenized_root() -> Path:
    """`<repo>/data/tokenized` — packed uint32 shards from our own tokenize_*.py."""
    return repo_root() / "data" / "tokenized"


def tokenizers_root() -> Path:
    """`<repo>/tokenizers` — custom-trained BPE artifacts live here."""
    return repo_root() / "tokenizers"


def apply_env_defaults() -> None:
    """Set HF_HOME / HF_DATASETS_CACHE / HF_HUB_CACHE if unset.

    Idempotent. Safe to call from any entry point. If the env vars are
    already set (e.g. by `scripts/launch.sh`), we respect those and only
    create the directories.
    """
    os.environ.setdefault("HF_HOME", str(hf_home()))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_datasets_cache()))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_hub_cache()))
    # Quietly ensure target dirs exist (they may be symlinks created by
    # scripts/data/migrate_hf_cache.sh — that's fine).
    for env_var in ("HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE"):
        try:
            Path(os.environ[env_var]).mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            # If the path is a broken symlink or similar, skip — let the
            # actual loader raise a clearer error.
            pass


def cache_dir_for_datasets() -> str:
    """The string to pass as `cache_dir=` to `datasets.load_dataset`."""
    return os.environ.get("HF_DATASETS_CACHE", str(hf_datasets_cache()))
