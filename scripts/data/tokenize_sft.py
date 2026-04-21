#!/usr/bin/env python3
"""Tokenize the SFT mix into ChatML-formatted packed shards with loss masks.

Each training example becomes a pair (token_ids, loss_mask), both uint32,
where `loss_mask[i] == 1` iff token i is an assistant token (so the LM
loss is computed only on assistant replies, not on prompts).

Shards are packed: multiple short conversations are concatenated with
EOS separators up to `--shard-size-tokens` tokens per file.

Output layout under `data/tokenized/sft_mix/`:
    shard_00000.ids.bin        uint32 token ids
    shard_00000.mask.bin       uint8  loss mask (0/1)
    ...
    index.json

Input schema
------------
We accept three common conversation schemas and normalize to a list of
`{"role": "user"|"assistant"|"system", "content": str}`:

  1. `messages: [{"role", "content"}, ...]`       — Tulu-3, SmolTalk, NoRobots
  2. `conversations: [{"from": "human"|"gpt"|"system", "value"}, ...]`  — OpenHermes
  3. `{"question"|"problem": ..., "answer"|"solution": ...}`  — MetaMath, OpenMathInstruct

Anything that doesn't match is skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

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


# (key, hf_name, config, split, weight)
SFT_SOURCES = [
    ("tulu-3",           "allenai/tulu-3-sft-mixture",      None,  "train", 0.40),
    ("smoltalk",         "HuggingFaceTB/smoltalk",          "all", "train", 0.20),
    ("openhermes-2.5",   "teknium/OpenHermes-2.5",          None,  "train", 0.15),
    ("metamath",         "meta-math/MetaMathQA",            None,  "train", 0.10),
    ("openmathinstruct", "nvidia/OpenMathInstruct-2",       None,  "train", 0.05),
    ("ifeval-like",      "argilla/ifeval-like-data",        None,  "train", 0.05),
    ("no-robots",        "HuggingFaceH4/no_robots",         None,  "train", 0.03),
    ("coqa-train",       "stanfordnlp/coqa",                None,  "train", 0.02),
]


Msg = Dict[str, str]  # {"role": ..., "content": ...}

_SHAREGPT_ROLE_MAP = {"human": "user", "user": "user", "gpt": "assistant",
                      "chatgpt": "assistant", "assistant": "assistant",
                      "system": "system"}


def _normalize(example: dict) -> Optional[List[Msg]]:
    if isinstance(example.get("messages"), list):
        msgs = []
        for m in example["messages"]:
            role = m.get("role"); content = m.get("content")
            if role in ("user", "assistant", "system") and isinstance(content, str):
                msgs.append({"role": role, "content": content})
        return msgs or None

    if isinstance(example.get("conversations"), list):
        msgs = []
        for m in example["conversations"]:
            role = _SHAREGPT_ROLE_MAP.get(m.get("from", "").lower())
            content = m.get("value")
            if role and isinstance(content, str):
                msgs.append({"role": role, "content": content})
        return msgs or None

    q = example.get("question") or example.get("problem") or example.get("instruction") \
        or example.get("query")
    a = example.get("answer") or example.get("solution") or example.get("response") \
        or example.get("generated_solution") or example.get("output")
    if isinstance(q, str) and isinstance(a, str):
        return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]

    # CoQA: story + parallel questions / answers.input_text
    story = example.get("story")
    questions = example.get("questions")
    answers = example.get("answers")
    if isinstance(story, str) and isinstance(questions, list) and isinstance(answers, dict):
        ans_texts = answers.get("input_text")
        if isinstance(ans_texts, list) and len(ans_texts) == len(questions):
            msgs: List[Msg] = [{"role": "system", "content": story}]
            for q_i, a_i in zip(questions, ans_texts):
                if isinstance(q_i, str) and isinstance(a_i, str):
                    msgs.append({"role": "user", "content": q_i})
                    msgs.append({"role": "assistant", "content": a_i})
            if len(msgs) > 1:
                return msgs
    return None


def _chatml_segments(tokenizer, msgs: List[Msg]) -> List[Tuple[List[int], int]]:
    """Return [(token_ids, loss_mask_value), ...] segments.

    Each assistant *content* segment is marked 1; everything else (role tags,
    user/system content, structural tokens) is 0. Loss is computed only on
    assistant tokens.
    """
    segs: List[Tuple[List[int], int]] = []
    for m in msgs:
        role, content = m["role"], m["content"]
        header = f"<|im_start|>{role}\n"
        trailer = "<|im_end|>\n"
        header_ids = tokenizer.encode(header, add_special_tokens=False)
        trailer_ids = tokenizer.encode(trailer, add_special_tokens=False)
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        segs.append((header_ids, 0))
        segs.append((content_ids, 1 if role == "assistant" else 0))
        segs.append((trailer_ids, 0))
    return segs


def _open(name: str, config: Optional[str], split: str) -> Iterator[dict]:
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split, cache_dir=cache_dir_for_datasets())
    for ex in ds:
        yield ex


@dataclass
class _ShardWriter:
    out_dir: Path
    shard_size_tokens: int
    shard_idx: int = 0
    buf_ids: List[np.ndarray] = field(default_factory=list)
    buf_mask: List[np.ndarray] = field(default_factory=list)
    buf_tokens: int = 0
    source_counts: Dict[str, int] = field(default_factory=dict)
    manifest: List[dict] = field(default_factory=list)

    def append(self, ids: np.ndarray, mask: np.ndarray, source_key: str) -> None:
        assert ids.shape == mask.shape
        self.buf_ids.append(ids)
        self.buf_mask.append(mask)
        self.buf_tokens += ids.size
        self.source_counts[source_key] = self.source_counts.get(source_key, 0) + int(ids.size)
        while self.buf_tokens >= self.shard_size_tokens:
            self._flush_full()

    def _flush_full(self) -> None:
        ids = np.concatenate(self.buf_ids); mask = np.concatenate(self.buf_mask)
        take_n = self.shard_size_tokens
        self._write(ids[:take_n], mask[:take_n])
        r_ids, r_mask = ids[take_n:], mask[take_n:]
        self.buf_ids = [r_ids] if r_ids.size else []
        self.buf_mask = [r_mask] if r_mask.size else []
        self.buf_tokens = int(r_ids.size)

    def flush_partial(self) -> None:
        if not self.buf_tokens:
            return
        ids = np.concatenate(self.buf_ids); mask = np.concatenate(self.buf_mask)
        self._write(ids, mask)
        self.buf_ids = []; self.buf_mask = []; self.buf_tokens = 0

    def _write(self, ids: np.ndarray, mask: np.ndarray) -> None:
        ids_path = self.out_dir / f"shard_{self.shard_idx:05d}.ids.bin"
        mask_path = self.out_dir / f"shard_{self.shard_idx:05d}.mask.bin"
        ids.astype(np.uint32, copy=False).tofile(ids_path)
        mask.astype(np.uint8, copy=False).tofile(mask_path)
        self.manifest.append({
            "ids": ids_path.name, "mask": mask_path.name,
            "tokens": int(ids.size),
            "assistant_tokens": int(mask.sum()),
            "source_counts": dict(self.source_counts),
        })
        print(f"[tokenize-sft] wrote {ids_path.name} tokens={ids.size:,} asst={int(mask.sum()):,}", flush=True)
        self.shard_idx += 1
        self.source_counts = {}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tokenizer", default=str(tokenizers_root() / "cl_small_bpe_16k"))
    p.add_argument("--output-dir", default=str(tokenized_root() / "sft_mix"))
    p.add_argument("--shard-size-tokens", type=int, default=50_000_000)
    p.add_argument("--target-tokens", type=int, default=600_000_000,
                   help="Stop once this many tokens have been packed (~1 epoch of the mix).")
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--max-seq-len", type=int, default=4096,
                   help="Drop conversations longer than this after tokenization.")
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = int(tokenizer.eos_token_id)

    sources = SFT_SOURCES
    if args.only:
        sources = [s for s in sources if s[0] in set(args.only)]
        if not sources:
            print(f"[tokenize-sft] --only {args.only} matched nothing", file=sys.stderr)
            return 1

    iters = {key: _open(name, cfg, split) for key, name, cfg, split, _ in sources}
    weights = np.array([s[4] for s in sources], dtype=np.float64); weights /= weights.sum()
    keys = [s[0] for s in sources]
    rng = np.random.default_rng(0)

    writer = _ShardWriter(out_dir=out_dir, shard_size_tokens=args.shard_size_tokens)
    total = 0; examples = 0; skipped = 0; exhausted: set[str] = set()
    t0 = time.time()

    while total < args.target_tokens and len(exhausted) < len(keys):
        key = keys[int(rng.choice(len(keys), p=weights))]
        if key in exhausted:
            continue
        try:
            ex = next(iters[key])
        except StopIteration:
            print(f"[tokenize-sft] exhausted: {key}", flush=True)
            exhausted.add(key); continue

        msgs = _normalize(ex)
        if not msgs or not any(m["role"] == "assistant" for m in msgs):
            skipped += 1; continue

        segs = _chatml_segments(tokenizer, msgs)
        ids = np.fromiter((t for seg, _ in segs for t in seg), dtype=np.uint32)
        mask = np.fromiter((v for seg, v in segs for _ in seg), dtype=np.uint8)
        if ids.size > args.max_seq_len:
            skipped += 1; continue

        # EOS between conversations; mask=0 for the separator.
        ids = np.concatenate([ids, np.asarray([eos_id], dtype=np.uint32)])
        mask = np.concatenate([mask, np.asarray([0], dtype=np.uint8)])

        writer.append(ids, mask, key)
        total += int(ids.size); examples += 1

        if examples % 2000 == 0:
            rate = total / max(time.time() - t0, 1e-6)
            print(f"[tokenize-sft] ex={examples:,} tok={total:,} skipped={skipped:,} ({rate/1e3:.1f}k tok/s)", flush=True)

    writer.flush_partial()
    (out_dir / "index.json").write_text(json.dumps({
        "dtype_ids": "uint32", "dtype_mask": "uint8",
        "shard_size_tokens": args.shard_size_tokens,
        "tokenizer": args.tokenizer,
        "sources": [{"key": k, "hf_name": n, "config": c, "split": s, "weight": w}
                    for (k, n, c, s, w) in sources],
        "shards": writer.manifest,
        "total_tokens": sum(s["tokens"] for s in writer.manifest),
        "total_examples": examples,
        "skipped": skipped,
        "elapsed_s": time.time() - t0,
    }, indent=2))
    print(f"[tokenize-sft] done. tokens={total:,} examples={examples:,} skipped={skipped:,}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
