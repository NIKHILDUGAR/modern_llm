"""HF-compatible inference/export entrypoint for ModernDecoderLM checkpoints.

The script supports the repo-native dense Transformer, opt-in BitNet/QAT
quantized checkpoints, and opt-in Gated DeltaNet / hybrid configs by exporting
raw ``.pt`` checkpoints into a small HuggingFace ``AutoModelForCausalLM``
wrapper directory.

Examples:
    python3 hfinfer.py \
      --model experiments/runs/.../lm-75m-2x4090-pretrain_final.pt \
      --prompt "Once upon a time"
    python3 hfinfer.py \
      --model ./hf_exports/lm-75m-bitnet \
      --prompt "Explain transformers simply."
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    TextStreamer,
)
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modern_llm.config.model_config import ModernLLMConfig
from modern_llm.models.loading import load_model_from_checkpoint, normalize_model_config_dict
from modern_llm.models.transformer import ModernDecoderLM
from modern_llm.quantization import QuantizationConfig, prepare_model_for_quantization, set_quantization_step


def _plain_dataclass_dict(value: Any) -> Any:
    """Convert nested dataclasses into JSON-serializable plain containers."""

    if is_dataclass(value):
        return {key: _plain_dataclass_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _plain_dataclass_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_dataclass_dict(item) for item in value]
    return value


def _modern_config_to_dict(config: ModernLLMConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, ModernLLMConfig):
        return _plain_dataclass_dict(config)
    return _plain_dataclass_dict(normalize_model_config_dict(config))


def _quantization_to_dict(config: QuantizationConfig | dict[str, Any] | None) -> dict[str, Any] | None:
    if config is None:
        return None
    if isinstance(config, QuantizationConfig):
        return config.to_dict() if config.enabled else None
    parsed = QuantizationConfig.from_dict(config)
    return parsed.to_dict() if parsed is not None and parsed.enabled else None


def _activate_quantization_for_inference(
    model: nn.Module,
    config: QuantizationConfig | None,
) -> None:
    """Switch warmup-gated low-bit layers into their inference quantized path."""

    if config is not None and config.enabled:
        set_quantization_step(model, config.quant_start_step)


class ModernLLMHFConfig(PretrainedConfig):
    """Thin HF config that stores the repo-native model config verbatim."""

    model_type = "modern_llm"

    def __init__(
        self,
        model_config: Optional[dict[str, Any]] = None,
        quantization: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        model_config = _modern_config_to_dict(model_config or {})
        kwargs.setdefault("is_decoder", True)
        kwargs.setdefault("is_encoder_decoder", False)
        kwargs.setdefault("use_cache", False)
        kwargs.setdefault("vocab_size", model_config.get("vocab_size", 0))
        kwargs.setdefault("hidden_size", model_config.get("d_model", 0))
        kwargs.setdefault("num_hidden_layers", model_config.get("n_layers", 0))
        kwargs.setdefault("num_attention_heads", model_config.get("n_heads", 0))
        kwargs.setdefault("max_position_embeddings", model_config.get("max_seq_len", 0))
        kwargs.setdefault("tie_word_embeddings", bool(model_config.get("tie_embeddings", True)))
        kwargs.setdefault("architectures", ["ModernLLMForCausalLM"])
        super().__init__(**kwargs)
        self.model_config = model_config
        self.quantization = _quantization_to_dict(quantization)

    @classmethod
    def from_modern_config(
        cls,
        model_config: ModernLLMConfig,
        quantization: QuantizationConfig | None = None,
        *,
        tokenizer: Any = None,
    ) -> "ModernLLMHFConfig":
        config_dict = _modern_config_to_dict(model_config)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        bos_token_id = getattr(tokenizer, "bos_token_id", None) or eos_token_id
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or eos_token_id
        return cls(
            model_config=config_dict,
            quantization=_quantization_to_dict(quantization),
            vocab_size=config_dict["vocab_size"],
            max_position_embeddings=config_dict["max_seq_len"],
            hidden_size=config_dict["d_model"],
            num_hidden_layers=config_dict["n_layers"],
            num_attention_heads=config_dict["n_heads"],
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=bool(config_dict.get("tie_embeddings", True)),
        )


class ModernLLMForCausalLM(PreTrainedModel, GenerationMixin):
    """HF ``PreTrainedModel`` wrapper around ``ModernDecoderLM``."""

    config_class = ModernLLMHFConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = ["DecoderBlock"]
    _tied_weights_keys = {"model.lm_head.weight": "model.token_embed.weight"}
    supports_gradient_checkpointing = False

    def __init__(
        self,
        config: ModernLLMHFConfig,
        modern_model: Optional[ModernDecoderLM] = None,
    ) -> None:
        super().__init__(config)
        self.model = modern_model if modern_model is not None else self._build_modern_model(config)
        self.config.vocab_size = self.model.config.vocab_size
        if hasattr(self, "generation_config"):
            self.generation_config.use_cache = False
        # We do not call post_init() here because repo checkpoints are already
        # initialized/loaded; however HF still needs this metadata to handle
        # tied token embeddings during save_pretrained/from_pretrained.
        self.all_tied_weights_keys = self.get_expanded_tied_weights_keys(all_submodels=False)

    @staticmethod
    def _build_modern_model(config: ModernLLMHFConfig) -> ModernDecoderLM:
        model_config = ModernLLMConfig(**normalize_model_config_dict(config.model_config))
        model = ModernDecoderLM(model_config)
        quantization = QuantizationConfig.from_dict(config.quantization)
        if quantization is not None and quantization.enabled:
            prepare_model_for_quantization(model, quantization)
            _activate_quantization_for_inference(model, quantization)
        return model

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.token_embed

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.token_embed = value
        if self.model.config.tie_embeddings:
            self.model.lm_head.weight = value.weight

    def get_output_embeddings(self) -> nn.Linear:
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.model.lm_head = new_embeddings

    def tie_weights(self, *args: Any, **kwargs: Any) -> None:
        if self.model.config.tie_embeddings:
            self.model.lm_head.weight = self.model.token_embed.weight

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        state = super().state_dict(*args, **kwargs)
        # HF removes declared tied tensors from checkpoints and then reports
        # lm_head as missing on reload. PyTorch .bin checkpoints can safely
        # store this small clone, which keeps reload logs honest.
        for key in tuple(state.keys()):
            if key.endswith("model.lm_head.weight") and isinstance(state[key], torch.Tensor):
                state[key] = state[key].clone()
        return state

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor, ...]:
        if input_ids is None:
            raise ValueError("ModernLLMForCausalLM requires input_ids; inputs_embeds are not supported.")
        return_dict = True if return_dict is None else return_dict
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        logits = outputs["logits"]
        if not return_dict:
            values: tuple[torch.Tensor, ...] = (logits,)
            return ((loss,) + values) if loss is not None else values
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)


def register_with_huggingface_auto() -> None:
    """Register the local model type with HF Auto classes for this process."""

    try:
        AutoConfig.register(ModernLLMHFConfig.model_type, ModernLLMHFConfig)
    except ValueError:
        pass
    try:
        AutoModelForCausalLM.register(ModernLLMHFConfig, ModernLLMForCausalLM)
    except ValueError:
        pass


def _save_tokenizer(tokenizer_dir: Path, output_dir: Path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(output_dir)
        return tokenizer
    except Exception as exc:
        known_files = (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
        )
        copied = False
        for name in known_files:
            source = tokenizer_dir / name
            if source.exists():
                shutil.copy2(source, output_dir / name)
                copied = True
        if not copied:
            raise RuntimeError(f"Could not save tokenizer from {tokenizer_dir}: {exc}") from exc
        return load_tokenizer(output_dir)


def export_checkpoint_to_hf(
    checkpoint_path: Path,
    output_dir: Path,
    tokenizer_dir: Path,
    *,
    safe_serialization: bool = False,
) -> Path:
    """Export a repo checkpoint into an HF ``save_pretrained`` directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    modern_model, model_config, quantization, _ = load_model_from_checkpoint(
        checkpoint_path,
        device="cpu",
        strict=True,
        eval_mode=True,
    )
    _activate_quantization_for_inference(modern_model, quantization)
    tokenizer = _save_tokenizer(tokenizer_dir, output_dir)
    hf_config = ModernLLMHFConfig.from_modern_config(
        model_config,
        quantization,
        tokenizer=tokenizer,
    )
    hf_model = ModernLLMForCausalLM(hf_config, modern_model=modern_model)
    hf_model.eval()
    hf_model.save_pretrained(output_dir, safe_serialization=safe_serialization)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer with a Modern LLM checkpoint via HF AutoModelForCausalLM.")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to either the original .pt checkpoint or an exported HF model directory.",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Prompt text to complete.")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Export a raw .pt checkpoint to an HF directory and exit without generation.",
    )
    parser.add_argument(
        "--hf-dir",
        type=Path,
        default=None,
        help="Where to export the HF model when --model points to a .pt checkpoint.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=REPO_ROOT / "tokenizers" / "cl_small_bpe_16k",
        help="Tokenizer directory used when exporting from a raw .pt checkpoint.",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or cuda:N. Default: auto.")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Inference dtype. Default picks bf16 on CUDA and fp32 on CPU.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=125)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=2.0)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Print generated text incrementally as tokens are produced.",
    )
    return parser.parse_args()


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(dtype_arg: str, device: torch.device) -> Optional[torch.dtype]:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16
    return None


def ensure_hf_dir(model_path: Path, hf_dir: Optional[Path], tokenizer_dir: Path) -> Path:
    if model_path.is_dir():
        return model_path
    if model_path.suffix != ".pt":
        raise ValueError(f"Expected a .pt checkpoint or HF directory, got: {model_path}")

    output_dir = hf_dir or model_path.with_suffix("")
    export_checkpoint_to_hf(
        checkpoint_path=model_path,
        output_dir=output_dir,
        tokenizer_dir=tokenizer_dir,
        safe_serialization=False,
    )
    return output_dir


def log_timing(step: str, started_at: float) -> None:
    elapsed = time.perf_counter() - started_at
    print(f"[timing] {step}: {elapsed:.3f}s", file=sys.stderr, flush=True)


def log_tokens_per_second(output_ids: torch.Tensor, prompt_length: int, started_at: float) -> None:
    elapsed = time.perf_counter() - started_at
    generated_tokens = max(int(output_ids.shape[-1]) - prompt_length, 0)
    tokens_per_second = (generated_tokens / elapsed) if elapsed > 0 else 0.0
    print(
        f"[timing] generate_tokens_per_second: {tokens_per_second:.2f} tok/s "
        f"({generated_tokens} tokens in {elapsed:.3f}s)",
        file=sys.stderr,
        flush=True,
    )


def load_tokenizer(model_dir: Path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(model_dir / "tokenizer.json"),
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
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(
    model_path: Path,
    hf_dir: Optional[Path],
    tokenizer_dir: Path,
    device: torch.device,
    dtype: Optional[torch.dtype],
):
    """Load from either a raw checkpoint or an already-exported HF model dir."""

    model_dir = ensure_hf_dir(model_path.resolve(), hf_dir, tokenizer_dir.resolve())
    tokenizer = load_tokenizer(model_dir)

    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)
    except ValueError:
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, **model_kwargs)

    model.to(device)
    model.eval()
    return model_dir, model, tokenizer


def main() -> None:
    total_started_at = time.perf_counter()
    args = parse_args()
    if args.prompt is None and not args.export_only:
        raise SystemExit("--prompt is required unless --export-only is set.")

    step_started_at = time.perf_counter()
    register_with_huggingface_auto()
    log_timing("register_with_huggingface_auto", step_started_at)

    step_started_at = time.perf_counter()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    log_timing("resolve_device_and_dtype", step_started_at)

    step_started_at = time.perf_counter()
    os.environ.setdefault("HF_HOME", "/tmp/huggingface")
    os.environ.setdefault("HF_MODULES_CACHE", "/tmp/huggingface/modules")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/huggingface/transformers")
    if args.export_only:
        model_dir = ensure_hf_dir(args.model.resolve(), args.hf_dir, args.tokenizer.resolve())
        print(model_dir)
        log_timing("export_only", step_started_at)
        log_timing("total", total_started_at)
        return

    model_dir, model, tokenizer = load_model_and_tokenizer(
        model_path=args.model,
        hf_dir=args.hf_dir,
        tokenizer_dir=args.tokenizer,
        device=device,
        dtype=dtype,
    )
    log_timing("load_model_and_tokenizer", step_started_at)

    step_started_at = time.perf_counter()
    assert args.prompt is not None
    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs.pop("token_type_ids", None)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    log_timing("prepare_inputs", step_started_at)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            }
        )
        if args.top_k > 0:
            generation_kwargs["top_k"] = args.top_k
    else:
        generation_kwargs["do_sample"] = False

    if args.stream:
        generation_kwargs["streamer"] = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    step_started_at = time.perf_counter()
    prompt_length = int(inputs["input_ids"].shape[-1])
    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    log_timing("generate", step_started_at)
    log_tokens_per_second(output_ids, prompt_length, step_started_at)

    step_started_at = time.perf_counter()
    if not args.stream:
        print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    else:
        print(file=sys.stdout, flush=True)
    log_timing("decode_and_print", step_started_at)
    log_timing("total", total_started_at)


if __name__ == "__main__":
    main()
