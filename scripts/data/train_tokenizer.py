#!/usr/bin/env python3
"""Train a custom 16k BPE tokenizer for the Modern LLM project.

Why we train our own tokenizer
------------------------------
The previous setup used `Xenova/text-embedding-ada-002` (cl100k, vocab 100,261).
At 75M params total, the embedding+lm_head ate 51M of the budget and forced an
8-layer model. A 16k BPE drops the embedding cost from 51M to ~8.4M (with
d_model=512 and tied input/output embeddings), freeing budget for a much
deeper transformer stack (18-22 layers) — strictly better depth/width per
MobileLLM ("Depth over width for sub-1B LMs").

Trade-off: 16k gives us slightly worse compression on code/math than 32k,
but for a 4k context window and a pretrain mix that's majority web text,
the depth gain dominates.

Sources
-------
We mix three corpora that already live in the HF cache (FineWeb-Edu sample,
The Stack v2 smol, OpenWebMath). This matches the pretrain mix and keeps the
tokenizer's compression high on our actual training data.

Usage
-----
    python scripts/data/train_tokenizer.py \
        --num-samples-per-source 200000 \
        --output-dir tokenizers/cl_small_bpe_16k

The output is a directory loadable via `AutoTokenizer.from_pretrained(<dir>)`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `modern_llm` importable when run from any CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "src"))

from modern_llm.utils.paths import (  # noqa: E402
    apply_env_defaults,
    cache_dir_for_datasets,
    tokenizers_root,
)

apply_env_defaults()

# Sources, weights, and how to extract the text field. Tweak weights to bias
# the merge frequencies toward the domain mix you care about.
SOURCES = [
    # (hf_name, hf_config, split, text_field, streaming)
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", True),
    ("open-web-math/open-web-math", None, "train", "text", True),
]


def text_iter(num_samples_per_source: int):
    """Yield strings from each source up to N samples each.

    Pre: sources are reachable via the HF cache (cache_dir env applies).
    Post: yields (text, source_index) pairs to keep merge counts roughly balanced.
    """
    from datasets import load_dataset

    cache_dir = cache_dir_for_datasets()
    for hf_name, hf_config, split, text_field, streaming in SOURCES:
        print(f"[tokenizer] streaming {hf_name} ({hf_config}, {split})", flush=True)
        ds = load_dataset(
            hf_name,
            hf_config,
            split=split,
            streaming=streaming,
            cache_dir=cache_dir,
        )
        n = 0
        for ex in ds:
            text = ex.get(text_field) or ""
            if not isinstance(text, str) or len(text) < 16:
                continue
            yield text
            n += 1
            if n >= num_samples_per_source:
                break
        print(f"[tokenizer]   used {n} samples from {hf_name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocab-size", type=int, default=16000, help="Final vocab size (incl. specials).")
    parser.add_argument(
        "--num-samples-per-source",
        type=int,
        default=200_000,
        help="Cap on rows pulled from each source. Default 200k yields ~1.5GB of text.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=tokenizers_root() / "cl_small_bpe_16k",
        help="Where the trained tokenizer is saved (load with AutoTokenizer.from_pretrained).",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum pair frequency to consider for merge (BPE).",
    )
    args = parser.parse_args()

    try:
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
        from transformers import PreTrainedTokenizerFast
    except ImportError as e:
        print("ERROR: requires `tokenizers` and `transformers`. pip install tokenizers transformers", file=sys.stderr)
        raise SystemExit(1) from e

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Byte-level BPE (GPT/LLaMA style). Pre-tokenization on bytes preserves
    # roundtrip and avoids OOV — a strong default for mixed web/code/math.
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    special_tokens = [
        "<|endoftext|>",
        "<|pad|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|user|>",
        "<|assistant|>",
        "<|system|>",
    ]
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    tok.train_from_iterator(text_iter(args.num_samples_per_source), trainer=trainer)

    # Wrap as a `transformers` fast tokenizer so the rest of the codebase
    # (AutoTokenizer.from_pretrained) picks it up unchanged.
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|pad|>",
        additional_special_tokens=[
            "<|im_start|>",
            "<|im_end|>",
            "<|user|>",
            "<|assistant|>",
            "<|system|>",
        ],
        model_max_length=4096,
    )
    fast_tok.save_pretrained(args.output_dir)
    print(f"[tokenizer] saved to {args.output_dir} (vocab={fast_tok.vocab_size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
