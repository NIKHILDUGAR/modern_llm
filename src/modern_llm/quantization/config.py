"""Configuration types for opt-in low-bit training modes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


QuantizationMode = Literal["none", "bitnet_b1_58", "qat_8da4w"]
QuantizationScaleMethod = Literal["median", "mean"]

_SUPPORTED_TARGETS = {"attention", "ffn", "lm_head", "embeddings"}


@dataclass
class QuantizationConfig:
    """Opt-in configuration for experimental quantized training."""

    mode: QuantizationMode = "none"
    target_modules: tuple[str, ...] = ("attention", "ffn")
    scale_method: QuantizationScaleMethod = "median"
    quant_start_step: int = 2000
    skip_first_blocks: int = 2
    skip_last_blocks: int = 2
    quantize_lm_head: bool = False
    quantize_embeddings: bool = False
    export_quantized_checkpoint: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"none", "bitnet_b1_58", "qat_8da4w"}:
            raise ValueError(f"Unsupported quantization mode: {self.mode}")
        if self.scale_method not in {"median", "mean"}:
            raise ValueError(f"Unsupported quantization scale_method: {self.scale_method}")
        self.target_modules = tuple(self.target_modules)
        invalid = [name for name in self.target_modules if name not in _SUPPORTED_TARGETS]
        if invalid:
            raise ValueError(
                f"Unsupported quantization target_modules: {invalid}. "
                f"Supported targets: {sorted(_SUPPORTED_TARGETS)}"
            )
        if self.quant_start_step < 0:
            raise ValueError("quant_start_step must be non-negative.")
        if self.skip_first_blocks < 0 or self.skip_last_blocks < 0:
            raise ValueError("skip_first_blocks and skip_last_blocks must be non-negative.")
        if self.quantize_embeddings and self.mode != "none":
            raise ValueError("Embedding quantization is not supported in v1; keep quantize_embeddings=False.")

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def wants_target(self, name: str) -> bool:
        return name in self.target_modules

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "QuantizationConfig | None":
        if data is None:
            return None
        return cls(**data)
