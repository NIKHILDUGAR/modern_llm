#!/usr/bin/env python3
"""Stream the pretrain mix through the 16k BPE and pack into uint32 shards.

Output layout under `data/tokenized/pretrain_mix/`:
    shard_00000.bin       flat uint32 token ids
    shard_00001.bin
    ...
    index.json            {"dtype":"uint32","shard_size_tokens":N,
                           "shards":[{"path":"shard_00000.bin","tokens":N,"source_counts":{...}},
                                     ...],
                           "total_tokens": T,
                           "tokenizer": "tokenizers/cl_small_bpe_16k"}

Design
------
- **Streaming, never materialized**: we iterate each `Source`'s HF stream,
  tokenize per example, append to an in-memory shard buffer; when the
  buffer hits `--shard-size-tokens`, flush and start the next shard.
- **Token-quota scheduler (default)**: at every step we pull the next
  example from the source whose *emitted-tokens / target-tokens* ratio is
  currently lowest. This converges on the declared weights regardless of
  per-example length variance — critical because rp-arxiv examples are
  ~10x longer than fineweb-edu chunks, so per-example weighted sampling
  silently over-represents arxiv. Pass `--no-enforce-quotas` to fall back
  to the legacy per-example weighted-random pick.
- **EOS between docs**: each example is followed by the tokenizer's EOS id
  so the LM learns document boundaries.
- **Resumable-ish**: if `index.json` already lists completed shards,
  rerunning skips past them and continues. (State inside a partial shard
  is NOT resumed — the partial shard is dropped and the source iterator
  restarts from zero; so this is "skip completed shards" resume, not
  true byte-level resume.)

Usage
-----
    # 200M-token smoke:
    python3 scripts/data/tokenize_pretrain.py \
        --target-tokens 200000000 \
        --shard-size-tokens 50000000

    # Full ~20B pretrain:
    python3 scripts/data/tokenize_pretrain.py \
        --target-tokens 20000000000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "src"))

from modern_llm.utils.paths import (  # noqa: E402
    apply_env_defaults,
    cache_dir_for_datasets,
    tokenized_root,
    tokenizers_root,
)

apply_env_defaults()


@dataclass
class Source:
    key: str
    hf_name: str
    hf_config: Optional[str]
    split: str
    text_field: str
    weight: float
    streaming: bool = True
    # Optional: RedPajama-style URL-manifest shim. When set, we load
    # `urls/<rp_manifest>.txt` from the repo and stream its jsonl shards via
    # the generic `json` loader (bypasses the deprecated RedPajama loading
    # script). `hf_config` is ignored when this is set.
    rp_manifest: Optional[str] = None


# Mirrors plan §3.1. Weights are rebalanced from the plan's token-percentages.
DEFAULT_SOURCES: List[Source] = [
    Source("fineweb-edu",        "HuggingFaceFW/fineweb-edu",             "sample-350BT",   "train", "text",    0.65),
    Source("dclm",               "mlfoundations/dclm-baseline-1.0",       None,             "train", "text",    0.10),
    #Source("stack-v1-smol",      "bigcode/the-stack-smol",                None,             "train", "content", 0.10, streaming=False),
    Source("openwebmath",        "open-web-math/open-web-math",           None,             "train", "text",    0.05),
    Source("wikipedia",          "wikimedia/wikipedia",                   "20231101.en",    "train", "text",    0.15, streaming=False),
    Source("rp-stackexchange",   "togethercomputer/RedPajama-Data-1T",    None,             "train", "text",    0.05, rp_manifest="stackexchange"),
    Source("rp-arxiv",           "togethercomputer/RedPajama-Data-1T",    None,             "train", "text",    0.05, rp_manifest="arxiv"),
]


def _open_stream(src: Source) -> Iterator[str]:
    from datasets import load_dataset
    if src.rp_manifest is not None:
        # RedPajama URL-manifest path. Stream the jsonl shards directly; no
        # loading script needed. `datasets` will handle URL list -> iterable.
        from huggingface_hub import hf_hub_download
        manifest_path = hf_hub_download(
            src.hf_name, f"urls/{src.rp_manifest}.txt", repo_type="dataset",
        )
        with open(manifest_path) as fh:
            urls = [ln.strip() for ln in fh if ln.strip()]
        print(f"[tokenize] {src.key}: {len(urls)} RedPajama shard urls", flush=True)
        ds = load_dataset(
            "json", data_files=urls, split=src.split,
            streaming=True, cache_dir=cache_dir_for_datasets(),
        )
    else:
        ds = load_dataset(
            src.hf_name, src.hf_config, split=src.split,
            streaming=src.streaming, cache_dir=cache_dir_for_datasets(),
        )
    for ex in ds:
        t = ex.get(src.text_field)
        if isinstance(t, bytes):
            t = t.decode("utf-8", "ignore")
        if t:
            yield t


@dataclass
class _ShardWriter:
    out_dir: Path
    shard_size_tokens: int
    eos_id: int
    dtype: np.dtype = field(default=np.dtype(np.uint32))
    buf: List[np.ndarray] = field(default_factory=list)
    buf_tokens: int = 0
    shard_idx: int = 0
    source_counts: Dict[str, int] = field(default_factory=dict)
    manifest: List[dict] = field(default_factory=list)

    def append(self, ids: np.ndarray, source_key: str) -> None:
        self.buf.append(ids)
        self.buf_tokens += ids.size
        self.source_counts[source_key] = self.source_counts.get(source_key, 0) + int(ids.size)
        while self.buf_tokens >= self.shard_size_tokens:
            self._flush_full()

    def _flush_full(self) -> None:
        flat = np.concatenate(self.buf)
        take, remain = flat[: self.shard_size_tokens], flat[self.shard_size_tokens :]
        self._write(take)
        self.buf = [remain] if remain.size else []
        self.buf_tokens = int(remain.size)

    def flush_partial(self) -> None:
        if not self.buf_tokens:
            return
        flat = np.concatenate(self.buf)
        self._write(flat)
        self.buf = []
        self.buf_tokens = 0

    def _write(self, arr: np.ndarray) -> None:
        path = self.out_dir / f"shard_{self.shard_idx:05d}.bin"
        arr.astype(self.dtype, copy=False).tofile(path)
        self.manifest.append({
            "path": path.name,
            "tokens": int(arr.size),
            "source_counts": dict(self.source_counts),
        })
        print(f"[tokenize] wrote {path.name}  tokens={arr.size:,}", flush=True)
        self.shard_idx += 1
        self.source_counts = {}


def _load_tokenizer(path: Path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(path))
    if tok.eos_token_id is None:
        raise RuntimeError(f"Tokenizer at {path} has no eos_token_id.")
    return tok


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tokenizer", default=str(tokenizers_root() / "cl_small_bpe_16k"))
    p.add_argument("--output-dir", default=str(tokenized_root() / "pretrain_mix"))
    p.add_argument("--target-tokens", type=int, default=20_000_000_000,
                   help="Stop once this many tokens have been packed.")
    p.add_argument("--shard-size-tokens", type=int, default=500_000_000,
                   help="Tokens per shard file (.bin).")
    p.add_argument("--only", nargs="+", default=None,
                   help="Restrict sources to these keys (smoke / debugging).")
    p.add_argument("--progress-every", type=int, default=10_000,
                   help="Log throughput every N examples.")
    p.add_argument("--enforce-quotas", dest="enforce_quotas", action="store_true", default=True,
                   help="(default) Pick next source by lowest emitted/target ratio — converges on exact weight proportions.")
    p.add_argument("--no-enforce-quotas", dest="enforce_quotas", action="store_false",
                   help="Use legacy per-example weighted-random picking. Skews toward sources with longer examples.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = _load_tokenizer(Path(args.tokenizer))
    eos_id = int(tokenizer.eos_token_id)

    sources = DEFAULT_SOURCES
    if args.only:
        sources = [s for s in sources if s.key in set(args.only)]
        if not sources:
            print(f"[tokenize] --only {args.only} matched nothing.", file=sys.stderr)
            return 1

    iters = {s.key: _open_stream(s) for s in sources}
    keys = [s.key for s in sources]
    weights = np.array([s.weight for s in sources], dtype=np.float64)
    weights = weights / weights.sum()
    weight_by_key = {k: float(w) for k, w in zip(keys, weights)}
    target_by_key = {k: args.target_tokens * w for k, w in weight_by_key.items()}
    emitted_by_key: Dict[str, int] = {k: 0 for k in keys}
    rng = np.random.default_rng(0)

    print(f"[tokenize] target={args.target_tokens:,} tokens  mode="
          f"{'quota' if args.enforce_quotas else 'weighted-random'}", flush=True)
    for k in keys:
        print(f"[tokenize]   {k:<20s} weight={weight_by_key[k]:.4f}  "
              f"target_tokens={int(target_by_key[k]):,}", flush=True)

    writer = _ShardWriter(
        out_dir=out_dir,
        shard_size_tokens=args.shard_size_tokens,
        eos_id=eos_id,
    )

    total_tokens = 0
    total_examples = 0
    exhausted: set[str] = set()
    t0 = time.time()

    def _pick_by_quota() -> Optional[str]:
        # Greedy: source with the smallest emitted/target ratio. Sources that
        # have met their quota (ratio >= 1) are skipped so we don't overshoot
        # a single source after others exhaust. Returns None iff every live
        # source has hit its quota — natural stop condition.
        best_key: Optional[str] = None
        best_ratio = float("inf")
        for k in keys:
            if k in exhausted:
                continue
            tgt = target_by_key[k]
            if tgt <= 0:
                continue
            ratio = emitted_by_key[k] / tgt
            if ratio >= 1.0:
                continue
            if ratio < best_ratio:
                best_ratio = ratio
                best_key = k
        return best_key

    def _pick_weighted_random() -> Optional[str]:
        # Renormalize weights over non-exhausted sources so exhaustion
        # redistributes mass instead of wasting rolls on dead iterators.
        live = [(k, weight_by_key[k]) for k in keys if k not in exhausted]
        if not live:
            return None
        lw = np.array([w for _, w in live], dtype=np.float64)
        lw /= lw.sum()
        return live[int(rng.choice(len(live), p=lw))][0]

    while total_tokens < args.target_tokens and len(exhausted) < len(keys):
        key = _pick_by_quota() if args.enforce_quotas else _pick_weighted_random()
        if key is None:
            # quota mode: every live source is at/over its target.
            break
        try:
            text = next(iters[key])
        except StopIteration:
            print(f"[tokenize] source exhausted: {key}", flush=True)
            exhausted.add(key)
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        ids.append(eos_id)
        arr = np.asarray(ids, dtype=np.uint32)
        writer.append(arr, key)
        total_tokens += arr.size
        emitted_by_key[key] += int(arr.size)
        total_examples += 1

        if total_examples % args.progress_every == 0:
            elapsed = time.time() - t0
            rate = total_tokens / max(elapsed, 1e-6)
            print(
                f"[tokenize] examples={total_examples:,} tokens={total_tokens:,} "
                f"({rate/1e3:.1f}k tok/s, shard={writer.shard_idx})",
                flush=True,
            )

    writer.flush_partial()

    index = {
        "dtype": "uint32",
        "shard_size_tokens": args.shard_size_tokens,
        "tokenizer": args.tokenizer,
        "sources": [s.__dict__ for s in sources],
        "shards": writer.manifest,
        "total_tokens": sum(s["tokens"] for s in writer.manifest),
        "total_examples": total_examples,
        "elapsed_s": time.time() - t0,
        "enforce_quotas": bool(args.enforce_quotas),
        "emitted_by_source": emitted_by_key,
        "target_by_source": {k: int(v) for k, v in target_by_key.items()},
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"[tokenize] done. total_tokens={index['total_tokens']:,} shards={len(writer.manifest)}", flush=True)
    total = max(index["total_tokens"], 1)
    for k in keys:
        got = emitted_by_key[k]
        tgt = target_by_key[k]
        print(f"[tokenize]   {k:<20s} got={got:>14,} ({100*got/total:5.2f}%)  "
              f"target={int(tgt):>14,} ({100*weight_by_key[k]:5.2f}%)"
              + ("  EXHAUSTED" if k in exhausted else ""),
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
