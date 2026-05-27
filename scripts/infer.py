"""Single-prompt inference for ModernDecoderLM checkpoints.

Usage:
    python scripts/infer.py --checkpoint path/to/model.pt --text "Your prompt here"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer

from modern_llm.config import ModernLLMConfig
from modern_llm.models import ModernDecoderLM
from modern_llm.training.train_lm import generate_text
from modern_llm.utils.checkpointing import load_checkpoint


def load_model(checkpoint_path: Path, device: torch.device) -> ModernDecoderLM:
    payload = load_checkpoint(checkpoint_path)
    if "config" not in payload or "model_state" not in payload:
        raise ValueError(f"{checkpoint_path} is not a ModernDecoderLM checkpoint.")

    config = ModernLLMConfig(**payload["config"])
    model = ModernDecoderLM(config)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on a ModernDecoderLM checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--text", type=str, required=True, help="Prompt text.")
    parser.add_argument("--tokenizer", type=str, default="tokenizers/cl_small_bpe_16k")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or cuda:N (default: auto).")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.checkpoint, device)
    tokenizer.model_max_length = model.config.max_seq_len

    top_k = args.top_k if args.top_k > 0 else None
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
    )
    print(output)


if __name__ == "__main__":
    main()
