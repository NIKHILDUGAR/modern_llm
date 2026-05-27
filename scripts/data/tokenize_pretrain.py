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
  batch examples, tokenize them with one or more worker processes, append
  to an in-memory shard buffer; when the next document would exceed
  `--shard-size-tokens`, flush and start the next shard.
- **Batched tokenization**: Hugging Face's fast tokenizer is called on lists
  of documents (`tokenizer(texts, ...)`) and can be sharded across
  `--tokenizer-workers` processes. The auto setting is capped so large CPU
  nodes do not spawn hundreds of tokenizer processes by accident. This keeps
  compatibility with streaming datasets, where `Dataset.map(..., num_proc=...)`
  is not available.
- **Token-quota scheduler (default)**: at every step we pull the next
  example from the source whose *emitted-tokens / target-tokens* ratio is
  currently lowest. This converges on the declared weights regardless of
  per-example length variance — critical because rp-arxiv examples are
  ~10x longer than fineweb-edu chunks, so per-example weighted sampling
  silently over-represents arxiv. Pass `--no-enforce-quotas` to fall back
  to the legacy per-example weighted-random pick.
- **EOS between docs**: each example is followed by the tokenizer's EOS id
  so the LM learns document boundaries.
- **Resumable at shard boundaries**: after each completed shard, `index.json`
  is atomically checkpointed with per-source token/example counts. Rerunning
  the same command skips already-written source examples and continues at the
  next shard. Any in-memory partial shard is intentionally dropped.

Usage
-----
    # 200M-token smoke:
    python3 scripts/data/tokenize_pretrain.py \
        --target-tokens 200000000 \
        --shard-size-tokens 50000000 \
        --tokenizer-workers 8 \
        --tokenizer-batch-size 128

    # Full ~20B pretrain:
    python3 scripts/data/tokenize_pretrain.py \
        --target-tokens 20000000000
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, Iterator, List, Optional, Tuple

import numpy as np

warnings.filterwarnings(
    "ignore",
    message="optree is installed but the version is too old.*",
    category=FutureWarning,
)

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "src"))

from modern_llm.utils.paths import (  # noqa: E402
    apply_env_defaults,
    cache_dir_for_datasets,
    tokenized_root,
    tokenizers_root,
)

apply_env_defaults()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


TextBatch = List[Tuple[str, str]]
TokenizedBatch = List[Tuple[str, np.ndarray]]
_WORKER_TOKENIZER = None
_WORKER_EOS_ID: Optional[int] = None
DEFAULT_TOKENIZER_WORKER_CAP = 32
DEFAULT_TOKENIZER_BATCH_SIZE = 128


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


# Target mix for the 75M v2 recipe. Keep these weights summing to 1.0 so the
# startup log matches the intended token percentages exactly.
DEFAULT_SOURCES: List[Source] = [
    Source("fineweb-edu",        "HuggingFaceFW/fineweb-edu",             "sample-350BT",   "train", "text",    0.65),
    Source("dclm",               "mlfoundations/dclm-baseline-1.0",       None,             "train", "text",    0.2),
    #Source("stack-v1-smol",      "bigcode/the-stack-smol",                None,             "train", "content", 0.10, streaming=False),
    Source("openwebmath",        "open-web-math/open-web-math",           None,             "train", "text",    0.05),
    Source("wikipedia",          "wikimedia/wikipedia",                   "20231101.en",    "train", "text",    0.05, streaming=False),
    Source("rp-stackexchange",   "togethercomputer/RedPajama-Data-1T",    None,             "train", "text",    0.02, rp_manifest="stackexchange"),
    Source("rp-arxiv",           "togethercomputer/RedPajama-Data-1T",    None,             "train", "text",    0.03, rp_manifest="arxiv"),
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
    on_checkpoint: Optional[Callable[[], None]] = None
    dtype: np.dtype = field(default=np.dtype(np.uint32))
    buf: List[np.ndarray] = field(default_factory=list)
    buf_tokens: int = 0
    shard_idx: int = 0
    source_counts: Dict[str, int] = field(default_factory=dict)
    example_counts: Dict[str, int] = field(default_factory=dict)
    manifest: List[dict] = field(default_factory=list)
    completed_tokens_by_source: Dict[str, int] = field(default_factory=dict)
    completed_examples_by_source: Dict[str, int] = field(default_factory=dict)
    completed_total_examples: int = 0

    def append(self, ids: np.ndarray, source_key: str) -> None:
        if self.buf_tokens and self.buf_tokens + ids.size > self.shard_size_tokens:
            self._flush_buffer()
        self.buf.append(ids)
        self.buf_tokens += ids.size
        self.source_counts[source_key] = self.source_counts.get(source_key, 0) + int(ids.size)
        self.example_counts[source_key] = self.example_counts.get(source_key, 0) + 1
        if self.buf_tokens >= self.shard_size_tokens:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        if not self.buf_tokens:
            return
        flat = np.concatenate(self.buf)
        self._write(flat)
        self.buf = []
        self.buf_tokens = 0

    def flush_partial(self) -> None:
        self._flush_buffer()

    def _write(self, arr: np.ndarray) -> None:
        path = self.out_dir / f"shard_{self.shard_idx:05d}.bin"
        tmp_path = path.with_name(path.name + ".tmp")
        arr.astype(self.dtype, copy=False).tofile(tmp_path)
        os.replace(tmp_path, path)
        source_counts = dict(self.source_counts)
        example_counts = dict(self.example_counts)
        self.manifest.append({
            "path": path.name,
            "tokens": int(arr.size),
            "source_counts": source_counts,
            "example_counts": example_counts,
        })
        for key, count in source_counts.items():
            self.completed_tokens_by_source[key] = (
                self.completed_tokens_by_source.get(key, 0) + int(count)
            )
        for key, count in example_counts.items():
            self.completed_examples_by_source[key] = (
                self.completed_examples_by_source.get(key, 0) + int(count)
            )
            self.completed_total_examples += int(count)
        print(f"[tokenize] wrote {path.name}  tokens={arr.size:,}", flush=True)
        self.shard_idx += 1
        self.source_counts = {}
        self.example_counts = {}
        if self.on_checkpoint is not None:
            self.on_checkpoint()


def _load_tokenizer(path: Path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(path))
    if tok.eos_token_id is None:
        raise RuntimeError(f"Tokenizer at {path} has no eos_token_id.")
    return tok


def _tokenize_batch(tokenizer, eos_id: int, batch: TextBatch) -> TokenizedBatch:
    if not batch:
        return []
    keys, texts = zip(*batch)
    encoded = tokenizer(
        list(texts),
        add_special_tokens=False,
        padding=False,
        truncation=False,
        verbose=False,
    )
    input_ids = encoded["input_ids"]
    out: TokenizedBatch = []
    for key, ids in zip(keys, input_ids):
        if not ids:
            continue
        ids = list(ids)
        ids.append(eos_id)
        out.append((key, np.asarray(ids, dtype=np.uint32)))
    return out


def _init_tokenizer_worker(tokenizer_path: str, eos_id: int) -> None:
    global _WORKER_TOKENIZER, _WORKER_EOS_ID
    _WORKER_TOKENIZER = _load_tokenizer(Path(tokenizer_path))
    _WORKER_EOS_ID = eos_id


def _tokenize_batch_in_worker(batch: TextBatch) -> TokenizedBatch:
    if _WORKER_TOKENIZER is None or _WORKER_EOS_ID is None:
        raise RuntimeError("Tokenizer worker was not initialized.")
    return _tokenize_batch(_WORKER_TOKENIZER, _WORKER_EOS_ID, batch)


def _chunks(items: TextBatch, chunk_size: int) -> Iterator[TextBatch]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def _available_cpu_count() -> int:
    for env_name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(env_name)
        if not value:
            continue
        try:
            cpus = int(value)
        except ValueError:
            continue
        if cpus > 0:
            return cpus

    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or multiprocessing.cpu_count())


def _resolve_tokenizer_workers(requested: int, cap: int) -> int:
    if requested < 0:
        raise ValueError("--tokenizer-workers must be >= 0.")
    if cap < 1:
        raise ValueError("--tokenizer-worker-cap must be >= 1.")
    if requested > 0:
        return requested
    return max(1, min(_available_cpu_count(), cap))


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def _shard_tokens(path: Path, dtype: np.dtype = np.dtype(np.uint32)) -> int:
    size_bytes = path.stat().st_size
    if size_bytes % dtype.itemsize:
        raise RuntimeError(f"{path} size is not divisible by {dtype.itemsize} bytes")
    return size_bytes // dtype.itemsize


def _source_signature(sources: List[Source]) -> List[dict]:
    return [s.__dict__ for s in sources]


def _validate_resume_index(index: dict, out_dir: Path, args, sources: List[Source]) -> None:
    if index.get("dtype") != "uint32":
        raise RuntimeError(f"Cannot resume {out_dir}: dtype is {index.get('dtype')!r}, expected 'uint32'.")
    if index.get("tokenizer") != args.tokenizer:
        raise RuntimeError(
            f"Cannot resume {out_dir}: tokenizer changed from "
            f"{index.get('tokenizer')!r} to {args.tokenizer!r}."
        )
    if int(index.get("shard_size_tokens", -1)) != int(args.shard_size_tokens):
        raise RuntimeError(
            f"Cannot resume {out_dir}: shard size changed from "
            f"{index.get('shard_size_tokens')!r} to {args.shard_size_tokens!r}."
        )
    if index.get("sources") != _source_signature(sources):
        raise RuntimeError(f"Cannot resume {out_dir}: source mix changed.")
    if "examples_by_source" not in index:
        raise RuntimeError(
            f"Cannot resume {out_dir}: index.json does not contain examples_by_source. "
            "It was likely produced by an older tokenizer script."
        )

    for shard in index.get("shards", []):
        path = out_dir / shard["path"]
        if not path.exists():
            raise RuntimeError(f"Cannot resume {out_dir}: missing shard {path.name}.")
        tokens = _shard_tokens(path)
        if tokens != int(shard["tokens"]):
            raise RuntimeError(
                f"Cannot resume {out_dir}: {path.name} has {tokens:,} tokens, "
                f"index.json says {int(shard['tokens']):,}."
            )


def _archive_unindexed_shards(out_dir: Path, indexed_shards: List[dict]) -> None:
    indexed_names = {str(shard["path"]) for shard in indexed_shards}
    stale = [path for path in sorted(out_dir.glob("shard_*.bin")) if path.name not in indexed_names]
    stale.extend(sorted(out_dir.glob("shard_*.bin.tmp")))
    if not stale:
        return
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for path in stale:
        archived = path.with_name(f"{path.name}.stale-{stamp}")
        path.rename(archived)
        print(
            f"[tokenize] archived unindexed partial shard {path.name} -> {archived.name}",
            flush=True,
        )


def _skip_examples(iterator: Iterator[str], n: int, source_key: str) -> Iterator[str]:
    if n <= 0:
        yield from iterator
        return

    print(f"[tokenize] resume: skipping {n:,} already-written examples from {source_key}", flush=True)
    skipped = 0
    while skipped < n:
        try:
            next(iterator)
        except StopIteration:
            print(
                f"[tokenize] resume: {source_key} exhausted while skipping "
                f"({skipped:,}/{n:,})",
                flush=True,
            )
            return
        skipped += 1
        if skipped % 100_000 == 0:
            print(f"[tokenize] resume: skipped {skipped:,}/{n:,} from {source_key}", flush=True)
    print(f"[tokenize] resume: finished skipping {source_key}", flush=True)
    yield from iterator


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
    p.add_argument("--tokenizer-workers", type=int, default=0,
                   help="Tokenizer worker processes. 0 means auto, capped by --tokenizer-worker-cap.")
    p.add_argument("--tokenizer-worker-cap", type=int, default=DEFAULT_TOKENIZER_WORKER_CAP,
                   help="Maximum workers used by --tokenizer-workers 0.")
    p.add_argument("--tokenizer-batch-size", type=int, default=DEFAULT_TOKENIZER_BATCH_SIZE,
                   help="Documents per tokenizer call in each worker.")
    p.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="Resume from output-dir/index.json when present.")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Start fresh; refuses to overwrite existing shards.")
    p.add_argument("--enforce-quotas", dest="enforce_quotas", action="store_true", default=True,
                   help="(default) Pick next source by lowest emitted/target ratio — converges on exact weight proportions.")
    p.add_argument("--no-enforce-quotas", dest="enforce_quotas", action="store_false",
                   help="Use legacy per-example weighted-random picking. Skews toward sources with longer examples.")
    args = p.parse_args()

    if args.tokenizer_batch_size < 1:
        print("[tokenize] --tokenizer-batch-size must be >= 1.", file=sys.stderr)
        return 1
    try:
        tokenizer_workers = _resolve_tokenizer_workers(
            args.tokenizer_workers,
            args.tokenizer_worker_cap,
        )
    except ValueError as exc:
        print(f"[tokenize] {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    tokenizer = _load_tokenizer(Path(args.tokenizer))
    eos_id = int(tokenizer.eos_token_id)

    sources = DEFAULT_SOURCES
    if args.only:
        sources = [s for s in sources if s.key in set(args.only)]
        if not sources:
            print(f"[tokenize] --only {args.only} matched nothing.", file=sys.stderr)
            return 1

    keys = [s.key for s in sources]
    weights = np.array([s.weight for s in sources], dtype=np.float64)
    weights = weights / weights.sum()
    weight_by_key = {k: float(w) for k, w in zip(keys, weights)}
    target_by_key = {k: args.target_tokens * w for k, w in weight_by_key.items()}

    resume_index: Optional[dict] = None
    existing_shards = sorted(out_dir.glob("shard_*.bin"))
    if not args.resume and (index_path.exists() or existing_shards):
        print(
            f"[tokenize] {out_dir} already contains tokenized output. "
            "Use --resume or a fresh --output-dir.",
            file=sys.stderr,
        )
        return 1
    if args.resume and index_path.exists():
        resume_index = json.loads(index_path.read_text())
        try:
            _validate_resume_index(resume_index, out_dir, args, sources)
        except RuntimeError as exc:
            print(f"[tokenize] {exc}", file=sys.stderr)
            return 1
        _archive_unindexed_shards(out_dir, resume_index.get("shards", []))
    elif existing_shards:
        print(
            f"[tokenize] {out_dir} has shard files but no resumable index.json. "
            "Use a fresh --output-dir or build an index from a known-good completed run.",
            file=sys.stderr,
        )
        return 1

    emitted_by_key: Dict[str, int] = {k: 0 for k in keys}
    examples_by_key: Dict[str, int] = {k: 0 for k in keys}
    if resume_index is not None:
        resumed_tokens = resume_index.get("emitted_by_source", {})
        resumed_examples = resume_index.get("examples_by_source", {})
        emitted_by_key = {k: int(resumed_tokens.get(k, 0)) for k in keys}
        examples_by_key = {k: int(resumed_examples.get(k, 0)) for k in keys}
    mean_tokens_by_key: Dict[str, float] = {
        k: (emitted_by_key[k] / examples_by_key[k] if examples_by_key[k] else 1024.0)
        for k in keys
    }
    ready_by_key: Dict[str, Deque[np.ndarray]] = {k: deque() for k in keys}
    rng = np.random.default_rng(0)

    print(f"[tokenize] target={args.target_tokens:,} tokens  mode="
          f"{'quota' if args.enforce_quotas else 'weighted-random'}", flush=True)
    print(f"[tokenize] tokenizer_workers={tokenizer_workers}  "
          f"tokenizer_batch_size={args.tokenizer_batch_size}  "
          f"auto_worker_cap={args.tokenizer_worker_cap}", flush=True)
    for k in keys:
        print(f"[tokenize]   {k:<20s} weight={weight_by_key[k]:.4f}  "
              f"target_tokens={int(target_by_key[k]):,}", flush=True)

    writer = _ShardWriter(
        out_dir=out_dir,
        shard_size_tokens=args.shard_size_tokens,
        eos_id=eos_id,
    )
    if resume_index is not None:
        writer.manifest = list(resume_index.get("shards", []))
        writer.shard_idx = len(writer.manifest)
        writer.completed_tokens_by_source = dict(emitted_by_key)
        writer.completed_examples_by_source = dict(examples_by_key)
        writer.completed_total_examples = sum(examples_by_key.values())

    total_tokens = sum(int(shard["tokens"]) for shard in writer.manifest)
    total_examples = writer.completed_total_examples
    exhausted: set[str] = set()
    t0 = time.time()

    def _build_index() -> dict:
        completed_tokens = sum(int(s["tokens"]) for s in writer.manifest)
        return {
            "dtype": "uint32",
            "shard_size_tokens": args.shard_size_tokens,
            "tokenizer": args.tokenizer,
            "sources": _source_signature(sources),
            "shards": writer.manifest,
            "total_tokens": completed_tokens,
            "total_examples": writer.completed_total_examples,
            "elapsed_s": time.time() - t0,
            "enforce_quotas": bool(args.enforce_quotas),
            "tokenizer_workers": tokenizer_workers,
            "tokenizer_batch_size": args.tokenizer_batch_size,
            "tokenizer_worker_cap": args.tokenizer_worker_cap,
            "emitted_by_source": {
                k: int(writer.completed_tokens_by_source.get(k, 0)) for k in keys
            },
            "examples_by_source": {
                k: int(writer.completed_examples_by_source.get(k, 0)) for k in keys
            },
            "target_by_source": {k: int(v) for k, v in target_by_key.items()},
            "resume": True,
        }

    def _write_checkpoint() -> None:
        _atomic_write_json(index_path, _build_index())

    writer.on_checkpoint = _write_checkpoint

    if resume_index is not None:
        print(
            f"[tokenize] resume: loaded {len(writer.manifest)} completed shards, "
            f"{total_tokens:,} tokens, {total_examples:,} examples",
            flush=True,
        )

    iters = {s.key: _open_stream(s) for s in sources}
    if resume_index is not None:
        iters = {
            k: _skip_examples(iters[k], examples_by_key.get(k, 0), k)
            for k in keys
        }

    def _pick_by_quota(reserved_by_key: Optional[Dict[str, float]] = None) -> Optional[str]:
        # Greedy: source with the smallest emitted/target ratio. Sources that
        # have met their quota (ratio >= 1) are skipped so we don't overshoot
        # a single source after others exhaust. Returns None iff every live
        # source has hit its quota — natural stop condition.
        best_key: Optional[str] = None
        best_ratio = float("inf")
        for k in keys:
            if k in exhausted and not ready_by_key[k]:
                continue
            tgt = target_by_key[k]
            if tgt <= 0:
                continue
            reserved = 0.0 if reserved_by_key is None else reserved_by_key.get(k, 0.0)
            ratio = (emitted_by_key[k] + reserved) / tgt
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

    def _read_text_batch(max_examples: int) -> TextBatch:
        batch: TextBatch = []
        reserved_by_key: Dict[str, float] = {}
        while len(batch) < max_examples and len(exhausted) < len(keys):
            if args.enforce_quotas:
                key = _pick_by_quota(reserved_by_key)
            else:
                key = _pick_weighted_random()
            if key is None:
                # Quota mode: every live source is at/over its target.
                break
            try:
                text = next(iters[key])
            except StopIteration:
                print(f"[tokenize] source exhausted: {key}", flush=True)
                exhausted.add(key)
                continue

            batch.append((key, text))
            reserved_by_key[key] = (
                reserved_by_key.get(key, 0.0)
                + max(mean_tokens_by_key.get(key, 1024.0), 1.0)
            )
        return batch

    def _write_tokenized(key: str, arr: np.ndarray) -> None:
        nonlocal total_tokens, total_examples
        writer.append(arr, key)
        total_tokens += arr.size
        emitted_by_key[key] += int(arr.size)
        total_examples += 1
        examples_by_key[key] += 1
        mean_tokens_by_key[key] = emitted_by_key[key] / max(examples_by_key[key], 1)

        if total_examples % args.progress_every == 0:
            elapsed = time.time() - t0
            rate = total_tokens / max(elapsed, 1e-6)
            print(
                f"[tokenize] examples={total_examples:,} tokens={total_tokens:,} "
                f"({rate/1e3:.1f}k tok/s, shard={writer.shard_idx})",
                flush=True,
            )

    def _enqueue_tokenized_batch(batch: TokenizedBatch) -> None:
        for key, arr in batch:
            ready_by_key[key].append(arr)

    def _drain_ready_queues() -> bool:
        while total_tokens < args.target_tokens:
            key = _pick_by_quota()
            if key is None:
                return False
            if not ready_by_key[key]:
                return True
            _write_tokenized(key, ready_by_key[key].popleft())
        return False

    max_examples_per_round = args.tokenizer_batch_size * tokenizer_workers
    pool: Optional[multiprocessing.pool.Pool] = None
    terminate_pool = False
    keep_running = True
    try:
        if tokenizer_workers > 1:
            pool = multiprocessing.Pool(
                processes=tokenizer_workers,
                initializer=_init_tokenizer_worker,
                initargs=(args.tokenizer, eos_id),
            )

        while keep_running and total_tokens < args.target_tokens and len(exhausted) < len(keys):
            keep_running = _drain_ready_queues()
            if not keep_running:
                break

            text_batch = _read_text_batch(max_examples_per_round)
            if not text_batch:
                break

            chunked = list(_chunks(text_batch, args.tokenizer_batch_size))
            if pool is None:
                tokenized_batches = (
                    _tokenize_batch(tokenizer, eos_id, chunk)
                    for chunk in chunked
                )
            else:
                tokenized_batches = pool.imap(_tokenize_batch_in_worker, chunked)

            for tokenized_batch in tokenized_batches:
                if keep_running:
                    _enqueue_tokenized_batch(tokenized_batch)
                    keep_running = _drain_ready_queues()
            if not keep_running:
                break
        if keep_running and total_tokens < args.target_tokens:
            keep_running = _drain_ready_queues()
    except BaseException:
        terminate_pool = True
        raise
    finally:
        if pool is not None:
            if terminate_pool:
                pool.terminate()
            else:
                pool.close()
            pool.join()

    writer.flush_partial()

    index = _build_index()
    _atomic_write_json(index_path, index)
    print(f"[tokenize] done. total_tokens={index['total_tokens']:,} shards={len(writer.manifest)}", flush=True)
    total = max(index["total_tokens"], 1)
    for k in keys:
        got = int(index["emitted_by_source"].get(k, 0))
        tgt = target_by_key[k]
        print(f"[tokenize]   {k:<20s} got={got:>14,} ({100*got/total:5.2f}%)  "
              f"target={int(tgt):>14,} ({100*weight_by_key[k]:5.2f}%)"
              + ("  EXHAUSTED" if k in exhausted else ""),
              flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    if rc == 0:
        os._exit(0)
    sys.exit(rc)
