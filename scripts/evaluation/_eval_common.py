#!/usr/bin/env python3
"""Shared evaluation helpers.

Not a benchmark runner — just small utilities (model loading, multiple-choice
log-likelihood scoring, simple greedy generation) that are reused across the
per-benchmark eval_*.py scripts in this directory. Each eval script still works
standalone, but this module keeps the boilerplate in one place.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from transformers import AutoTokenizer

# Ensure modern_llm is importable when eval scripts are run from any CWD.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


# Tokenizer resolution order (first existing wins, else falls back to HF hub):
#   1. $MODERN_LLM_TOKENIZER env var — explicit override
#   2. <repo>/tokenizers/cl_small_bpe_16k — our custom 16k BPE (produced by
#      scripts/data/train_tokenizer.py). This matches the post-4090 runs.
#   3. "Xenova/text-embedding-ada-002" — legacy cl100k tokenizer used by the
#      archived pre-4090 checkpoints; kept so old eval sweeps still load.
import os as _os

_CUSTOM_BPE_DIR = PROJECT_ROOT / "tokenizers" / "cl_small_bpe_16k"
if _os.environ.get("MODERN_LLM_TOKENIZER"):
    DEFAULT_TOKENIZER = _os.environ["MODERN_LLM_TOKENIZER"]
elif _CUSTOM_BPE_DIR.exists():
    DEFAULT_TOKENIZER = str(_CUSTOM_BPE_DIR)
else:
    DEFAULT_TOKENIZER = "Xenova/text-embedding-ada-002"


_VALID_CONFIG_KEYS = {
    "vocab_size", "d_model", "n_layers", "n_heads", "ffn_hidden_size",
    "max_seq_len", "rmsnorm_eps", "dropout", "initializer_range",
    "rope_theta", "rope_scaling", "use_rope", "use_attention_sinks",
    "num_attention_sinks", "use_swiglu", "swiglu_multiplier", "use_gqa",
    "gqa_groups", "use_qk_norm", "use_moe", "moe_config", "tie_embeddings",
    "scale_embeddings", "residual_init_scale", "z_loss_coef",
    "sequence_mixer", "gated_deltanet_layers", "gated_deltanet_num_heads",
    "gated_deltanet_conv_kernel",
}


class _HFCausalLMAdapter(torch.nn.Module):
    """Wrap a HF CausalLM so it mimics ModernDecoderLM's forward interface.

    Eval scripts expect `model(input_ids)["logits"]` and `model.config.max_seq_len`.
    HF models return `CausalLMOutputWithPast.logits` and advertise max length via
    `config.max_position_embeddings` (GPT-2) or `config.max_position_embeddings`
    (SmolLM2). This shim bridges the two without touching any eval_*.py.
    """

    def __init__(self, hf_model) -> None:
        super().__init__()
        self.hf = hf_model
        hf_cfg = hf_model.config

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.max_seq_len = int(getattr(hf_cfg, "max_position_embeddings", None)
                              or getattr(hf_cfg, "n_positions", 1024))
        cfg.vocab_size = int(hf_cfg.vocab_size)
        self.config = cfg

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.hf(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return {"logits": out.logits, "loss": getattr(out, "loss", None)}


def _looks_like_hf_id(spec: str) -> bool:
    # HF IDs are "org/name" or well-known bare names ("gpt2"); checkpoints are .pt/.bin/.safetensors files
    p = Path(spec)
    if p.exists() and p.is_file():
        return False
    if spec.endswith((".pt", ".bin", ".safetensors", ".ckpt")):
        return False
    return True


def load_scratch_model(checkpoint_path: str, device: str, tokenizer_name: str = DEFAULT_TOKENIZER):
    """Load a ModernDecoderLM checkpoint + matching tokenizer.

    Tokenizer resolution: see DEFAULT_TOKENIZER at module top. The 16k custom
    BPE (`tokenizers/cl_small_bpe_16k/`) is preferred when present; it matches
    the post-4090 runs. Falls back to the legacy cl100k tokenizer for archived
    pre-4090 checkpoints. Callers can override with --tokenizer or the
    MODERN_LLM_TOKENIZER env var.

    If `checkpoint_path` doesn't point at a local file, it's treated as a HF
    model id (e.g. "gpt2", "HuggingFaceTB/SmolLM2-135M") and loaded via
    transformers. The returned model wraps the HF forward to match the
    scratch model's `{"logits", "loss"}` dict interface.
    """
    from modern_llm.config.model_config import ModernLLMConfig
    from modern_llm.models.transformer import ModernDecoderLM

    if _looks_like_hf_id(checkpoint_path):
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        hf_model.to(device)
        hf_model.eval()
        model = _HFCausalLMAdapter(hf_model)
        # Tokenizer: prefer explicit override, else use the HF model's own tokenizer.
        tok_name = tokenizer_name if tokenizer_name and tokenizer_name != DEFAULT_TOKENIZER else checkpoint_path
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config" in checkpoint:
        cfg = dict(checkpoint["config"])
        if "num_layers" in cfg and "n_layers" not in cfg:
            cfg["n_layers"] = cfg.pop("num_layers")
        if "max_position_embeddings" in cfg and "max_seq_len" not in cfg:
            cfg["max_seq_len"] = cfg.pop("max_position_embeddings")
        cfg = {k: v for k, v in cfg.items() if k in _VALID_CONFIG_KEYS}
        config = ModernLLMConfig(**cfg)
    else:
        raise ValueError(f"Checkpoint {checkpoint_path} missing 'config' key.")

    model = ModernDecoderLM(config)
    state_dict = checkpoint.get("model_state", checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint)))
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@torch.no_grad()
def score_completion(model, tokenizer, prompt: str, completion: str, device: str) -> float:
    """Return sum of log-probs assigned to the completion tokens given the prompt.

    Math:
        score = sum_{t in completion} log p(x_t | x_<t)
    where p is the model's next-token distribution over the concatenated
    prompt+completion sequence. Longer completions get more-negative scores, so
    callers that compare completions of different lengths should length-normalize.
    """
    max_len = getattr(model.config, "max_seq_len", 1024)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if len(completion_ids) == 0:
        return 0.0
    # Truncate from the left of the prompt to fit context
    total_budget = max_len - 1
    if len(prompt_ids) + len(completion_ids) > total_budget:
        keep = total_budget - len(completion_ids)
        prompt_ids = prompt_ids[-max(keep, 0):]
    input_ids = torch.tensor([prompt_ids + completion_ids], device=device, dtype=torch.long)
    outputs = model(input_ids)
    logits = outputs["logits"]  # (1, T, V)
    # next-token prediction: logits[:, t, :] predicts token at position t+1
    log_probs = torch.log_softmax(logits[0, :-1, :], dim=-1)
    target = input_ids[0, 1:]
    # Sum log-probs over the completion positions only
    prompt_len = len(prompt_ids)
    comp_positions = slice(prompt_len - 1, prompt_len - 1 + len(completion_ids))
    gathered = log_probs[comp_positions, :].gather(1, target[comp_positions].unsqueeze(1)).squeeze(1)
    return float(gathered.sum().item())


@torch.no_grad()
def mc_argmax(
    model,
    tokenizer,
    prompt: str,
    choices: Sequence[str],
    device: str,
    length_normalize: bool = False,
) -> int:
    """Return the index of the highest-scoring choice.

    length_normalize=True divides by completion token count (useful when
    choices differ greatly in length, e.g. HellaSwag endings).
    """
    scores = []
    for c in choices:
        s = score_completion(model, tokenizer, prompt, c, device)
        if length_normalize:
            tok_len = max(1, len(tokenizer.encode(c, add_special_tokens=False)))
            s = s / tok_len
        scores.append(s)
    return int(max(range(len(scores)), key=lambda i: scores[i]))


@torch.no_grad()
def greedy_generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 128,
    stop_strings: Optional[List[str]] = None,
) -> str:
    """Greedy decode with optional string-level stopping.

    The scratch model has no KV cache, so this is O(max_new_tokens * seq)
    per call. Kept simple on purpose.
    """
    max_len = getattr(model.config, "max_seq_len", 1024)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(prompt_ids) >= max_len:
        prompt_ids = prompt_ids[-(max_len - 1):]
    generated = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    produced: List[int] = []
    eos = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        if generated.shape[1] >= max_len:
            break
        outputs = model(generated)
        next_token = int(outputs["logits"][0, -1, :].argmax().item())
        produced.append(next_token)
        generated = torch.cat([generated, torch.tensor([[next_token]], device=device)], dim=1)
        if eos is not None and next_token == eos:
            break
        if stop_strings:
            partial = tokenizer.decode(produced, skip_special_tokens=True)
            if any(s in partial for s in stop_strings):
                for s in stop_strings:
                    if s in partial:
                        partial = partial.split(s)[0]
                return partial
    return tokenizer.decode(produced, skip_special_tokens=True)
