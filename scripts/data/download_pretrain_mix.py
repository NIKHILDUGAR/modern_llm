#!/usr/bin/env python3
"""Resolve (cache or stream-touch) every dataset in the pretrain mix.

The plan's §3.1 pretrain mix, codified here so later stages never have to
re-type the list. Runs `ensure_dataset` against each source. Streaming
sources are smoke-iterated (no full download) because the pretrainer
streams them at train time; on-disk arrow sources are materialized under
`data/raw/hf_cache/`.

Usage
-----
    python3 scripts/data/download_pretrain_mix.py
    python3 scripts/data/download_pretrain_mix.py --only fineweb-edu
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from modern_llm.utils.paths import apply_env_defaults  # noqa: E402

apply_env_defaults()

from ensure_dataset import ensure_dataset  # noqa: E402


@dataclass
class Source:
    key: str                    # short name for --only filter
    hf_name: str
    hf_config: Optional[str]
    split: str
    streaming: bool             # True for huge web-scale sources
    # Optional: RedPajama-style URL-manifest shim. When set, the loader bypasses
    # ensure_dataset() and streams the jsonl shards listed inside
    # `urls/<rp_manifest>.txt` under `hf_name` directly. `hf_config` is ignored.
    rp_manifest: Optional[str] = None


PRETRAIN_MIX = [
    Source("fineweb-edu",        "HuggingFaceFW/fineweb-edu",             "sample-350BT",   "train", True),
    Source("dclm",               "mlfoundations/dclm-baseline-1.0",       None,             "train", True),
    Source("stack-v1-smol",      "bigcode/the-stack-smol",                None,             "train", False),
    Source("openwebmath",        "open-web-math/open-web-math",           None,             "train", True),
    Source("wikipedia",          "wikimedia/wikipedia",                   "20231101.en",    "train", False),
    Source("rp-stackexchange",   "togethercomputer/RedPajama-Data-1T",    None,             "train", True,  rp_manifest="stackexchange"),
    Source("rp-arxiv",           "togethercomputer/RedPajama-Data-1T",    None,             "train", True,  rp_manifest="arxiv"),
]


def _ensure_rp_manifest(src: Source, smoke_iter: int) -> None:
    """Stream-touch a RedPajama-Data-1T subset via its `urls/<subset>.txt`.

    `datasets>=4.x` dropped loading-script support, so we bypass the repo's
    `RedPajama-Data-1T.py` and feed the shard URLs directly to the `json`
    loader in streaming mode. Nothing is cached to disk here — the full
    download happens later inside `tokenize_pretrain.py` as it streams.
    """
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    manifest_path = hf_hub_download(
        src.hf_name, f"urls/{src.rp_manifest}.txt", repo_type="dataset",
    )
    with open(manifest_path) as fh:
        urls = [ln.strip() for ln in fh if ln.strip()]
    if not urls:
        raise RuntimeError(f"RedPajama manifest urls/{src.rp_manifest}.txt is empty")
    print(f"[ensure] RP manifest {src.rp_manifest}: {len(urls)} shard urls", flush=True)
    ds = load_dataset("json", data_files=urls[:1], split=src.split, streaming=True)
    for i, _ in enumerate(ds):
        if i + 1 >= smoke_iter:
            break
    print(f"[ensure] ok: {src.hf_name}[{src.rp_manifest}] ({len(urls)} shards ready)", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="+", default=None,
                   help="Restrict to these source keys (default: all).")
    p.add_argument("--smoke-iter", type=int, default=4)
    args = p.parse_args()

    want = set(args.only) if args.only else None
    failures: list[tuple[str, str]] = []
    for src in PRETRAIN_MIX:
        if want and src.key not in want:
            continue
        try:
            if src.rp_manifest is not None:
                _ensure_rp_manifest(src, args.smoke_iter)
            else:
                ensure_dataset(src.hf_name, src.hf_config, src.split, src.streaming, args.smoke_iter)
        except Exception as exc:
            print(f"[pretrain-mix] FAIL {src.key} ({src.hf_name}): {exc}", flush=True)
            failures.append((src.key, str(exc)))

    if failures:
        print(f"\n[pretrain-mix] {len(failures)} failures:", flush=True)
        for k, msg in failures:
            print(f"  - {k}: {msg}")
        return 1
    print("\n[pretrain-mix] all ok", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
