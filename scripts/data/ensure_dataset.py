#!/usr/bin/env python3
"""Symlink-or-download helper for HuggingFace datasets.

Checks whether a dataset's arrow cache already exists under
`data/raw/hf_cache/` (possibly via the symlinks that
`scripts/data/migrate_hf_cache.sh` creates). If yes, it's a no-op. If no,
it issues a minimal `load_dataset()` call so HF pulls the data into the
canonical cache location.

Streaming sources are "touched" (iter a few examples) instead of
fully downloaded, since they don't populate the arrow cache at all.

Usage
-----
    # CLI form — single dataset
    python3 scripts/data/ensure_dataset.py \
        --name HuggingFaceFW/fineweb-edu --config sample-10BT --streaming

    # Library form — from other scripts
    from scripts.data.ensure_dataset import ensure_dataset
    ensure_dataset("wikimedia/wikipedia", config="20231101.en")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "src"))

from modern_llm.utils.paths import (  # noqa: E402
    apply_env_defaults,
    cache_dir_for_datasets,
    hf_datasets_cache,
    hf_hub_cache,
)

apply_env_defaults()


def _cached_slug(name: str) -> str:
    # HF's on-disk layout: `owner___dataset` under hf_cache; `datasets--owner--dataset` under hub.
    return name.replace("/", "___")


def _hub_slug(name: str) -> str:
    return "datasets--" + name.replace("/", "--")


def _already_cached(name: str) -> bool:
    if (hf_datasets_cache() / _cached_slug(name)).exists():
        return True
    if (hf_hub_cache() / _hub_slug(name)).exists():
        return True
    return False


def ensure_dataset(
    name: str,
    config: Optional[str] = None,
    split: str = "train",
    streaming: bool = False,
    smoke_iter: int = 4,
) -> None:
    """Make sure `name` is resolvable from the repo cache. Cheap if already cached."""
    from datasets import load_dataset

    if _already_cached(name) and not streaming:
        print(f"[ensure] cached: {name} ({config or '-'})", flush=True)
        return

    print(f"[ensure] loading: {name} ({config or '-'}, streaming={streaming})", flush=True)
    ds = load_dataset(
        name,
        config,
        split=split,
        streaming=streaming,
        cache_dir=cache_dir_for_datasets(),
    )
    if streaming:
        # Streaming doesn't populate a local cache — just prove it iterates.
        for i, _ in enumerate(ds):
            if i + 1 >= smoke_iter:
                break
    else:
        _ = len(ds)
    print(f"[ensure] ok: {name}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--name", required=True, help="HF dataset id, e.g. wikimedia/wikipedia")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--streaming", action="store_true")
    p.add_argument("--smoke-iter", type=int, default=4,
                   help="Streaming-only: number of examples to iterate as smoke.")
    args = p.parse_args()
    ensure_dataset(args.name, args.config, args.split, args.streaming, args.smoke_iter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
