#!/usr/bin/env python3
"""Create a small packed-shard view over an in-progress tokenized dataset.

This is useful while `tokenize_pretrain.py` is still running: completed shard
files already exist, but the final `index.json` is only written at the end.
The subset directory contains symlinks to the first N complete shards plus a
minimal index that `PackedShardDataset` can load.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _shard_tokens(path: Path) -> int:
    size = path.stat().st_size
    if size % 4 != 0:
        raise ValueError(f"{path} size is not divisible by 4 bytes; shard may be incomplete")
    return size // 4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-dir", required=True, help="Directory with shard_*.bin files.")
    parser.add_argument("--output-dir", required=True, help="Directory for symlinks + index.json.")
    parser.add_argument("--max-shards", type=int, default=10, help="Number of completed shards to expose.")
    parser.add_argument("--tokenizer", default="tokenizers/cl_small_bpe_16k")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(source_dir.glob("shard_*.bin"))[: args.max_shards]
    if len(shards) < args.max_shards:
        raise RuntimeError(
            f"Need {args.max_shards} shards in {source_dir}, found {len(shards)}"
        )

    manifest = []
    for shard in shards:
        link = output_dir / shard.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(shard)
        manifest.append(
            {
                "path": link.name,
                "tokens": _shard_tokens(shard),
                "source_counts": {},
            }
        )

    index = {
        "dtype": "uint32",
        "shard_size_tokens": manifest[0]["tokens"],
        "tokenizer": args.tokenizer,
        "sources": [],
        "shards": manifest,
        "total_tokens": sum(item["tokens"] for item in manifest),
        "total_examples": 0,
        "elapsed_s": 0,
        "enforce_quotas": True,
        "emitted_by_source": {},
        "target_by_source": {},
    }
    (output_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(
        f"[packed-subset] wrote {output_dir / 'index.json'} "
        f"with {len(manifest)} shards / {index['total_tokens']:,} tokens"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
