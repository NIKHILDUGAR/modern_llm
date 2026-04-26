"""Unified pipeline configuration for end-to-end training.

Combines model, training, hardware, and data configs into a single
JSON-serializable structure for run_pipeline.py orchestration.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from .hardware_config import DataConfig, HardwareConfig, get_data_preset, get_hardware_preset
from .model_config import ModernLLMConfig, MoEConfig
from .train_config import TrainingConfig
from modern_llm.quantization import QuantizationConfig
from modern_llm.training.distributed import scale_grad_accum_for_world_size


@dataclass
class PipelineConfig:
    """Full configuration for pretrain -> SFT -> DPO -> Verifier pipeline.

    Each stage has its own training config, but they share model and hardware.
    """

    # Model architecture
    vocab_size: int = 50257
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    ffn_hidden_size: int = 1536
    max_seq_len: int = 1024
    dropout: float = 0.1
    use_rope: bool = True
    use_attention_sinks: bool = True
    num_attention_sinks: int = 4
    use_swiglu: bool = True
    tie_embeddings: bool = True
    use_gqa: bool = True
    gqa_groups: Optional[int] = 2
    use_qk_norm: bool = False
    use_moe: bool = False
    scale_embeddings: bool = False
    residual_init_scale: bool = True
    z_loss_coef: float = 0.0
    sequence_mixer: str = "attention"
    gated_deltanet_layers: Optional[List[int]] = None
    gated_deltanet_num_heads: Optional[int] = None
    gated_deltanet_conv_kernel: int = 4
    compile_model: Optional[bool] = None

    # Hardware
    hardware_preset: str = "auto"

    # Data scale
    data_preset: str = "small"
    
    # Pretrain datasets (list of dataset names from DATASET_REGISTRY).
    # Used when `pretrain_packed_shards` is not set.
    pretrain_datasets: Optional[List[str]] = None

    # If set, pretrain reads packed uint32 shards from this directory
    # (produced by scripts/data/tokenize_pretrain.py) instead of streaming
    # + tokenizing raw HF datasets. Takes precedence over pretrain_datasets.
    pretrain_packed_shards: Optional[str] = None

    # Pretraining
    pretrain_max_steps: int = 20000
    pretrain_lr: float = 3e-4
    pretrain_batch_size: int = 2
    pretrain_micro_batch_size: int = 2
    pretrain_warmup_steps: int = 500
    pretrain_min_lr_ratio: float = 0.1
    # Number of windows held out from the tail of the packed shards as eval.
    # Only used when `pretrain_packed_shards` is set. Eval tokens = N * max_seq_len.
    pretrain_eval_windows: int = 256

    # SFT
    sft_max_steps: int = 5000
    sft_lr: float = 1e-5
    sft_batch_size: int = 32
    sft_micro_batch_size: int = 2
    sft_dataset: str = "tatsu-lab/alpaca"
    sft_datasets: Optional[List[str]] = None  # Multiple SFT datasets (overrides sft_dataset)
    # Interleave probabilities for `sft_datasets` (must match its length, normalized to sum to 1.0).
    # When None, all listed datasets get equal weight.
    sft_dataset_weights: Optional[List[float]] = None
    # Optional cap for each SFT source when building a multi-dataset mixture.
    # When None, the pipeline infers a cap from sft_max_steps * sft_batch_size.
    sft_num_examples_per_dataset: Optional[int] = None

    # DPO
    dpo_max_steps: int = 2000
    dpo_lr: float = 5e-6
    dpo_batch_size: int = 16
    dpo_micro_batch_size: int = 1
    dpo_beta: float = 0.1
    dpo_dataset: str = "Anthropic/hh-rlhf"
    dpo_num_examples: Optional[int] = None

    # Verifier
    verifier_max_steps: int = 3000
    verifier_lr: float = 1e-4
    verifier_batch_size: int = 32
    verifier_micro_batch_size: int = 4

    # Paths
    from datetime import datetime
    now = datetime.now()
    now= str(now).replace(" ","")
    now= str(now).replace(":","")

    output_dir: str = "experiments/runs/"+now
    run_name: str = "modern-llm-pipeline1"+now+str(d_model) + str(n_layers)+str(n_heads)
    tokenizer_name: str = "Xenova/text-embedding-ada-002"

    # Misc
    seed: int = 42
    mixed_precision: str = "bf16"
    eval_every: int = 500
    save_every: int = 2000
    log_every: int = 1000
    quantization: Optional[QuantizationConfig] = None

    def __post_init__(self) -> None:
        if isinstance(self.quantization, dict):
            self.quantization = QuantizationConfig.from_dict(self.quantization)

    def get_model_config(self) -> ModernLLMConfig:
        """Build ModernLLMConfig from pipeline settings."""
        moe_config = None
        import json
        with open(self.output_dir+'config.txt', 'w') as fp:
            for key, value in self.__dict__.items():    
                fp.write(f"{key}: {value}")
                print(f"{key}: {value}")

        if self.use_moe:
            moe_config = MoEConfig()

        print("get_hardware_preset(self.hardware_preset",get_hardware_preset(self.hardware_preset))
        return ModernLLMConfig(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            ffn_hidden_size=self.ffn_hidden_size,
            max_seq_len=self.max_seq_len,
            dropout=self.dropout,
            use_rope=self.use_rope,
            use_attention_sinks=self.use_attention_sinks,
            num_attention_sinks=self.num_attention_sinks,
            use_swiglu=self.use_swiglu,
            tie_embeddings=self.tie_embeddings,
            use_gqa=self.use_gqa,
            gqa_groups=self.gqa_groups,
            use_qk_norm=self.use_qk_norm,
            use_moe=self.use_moe,
            moe_config=moe_config,
            scale_embeddings=self.scale_embeddings,
            residual_init_scale=self.residual_init_scale,
            z_loss_coef=self.z_loss_coef,
            sequence_mixer=self.sequence_mixer,
            gated_deltanet_layers=self.gated_deltanet_layers,
            gated_deltanet_num_heads=self.gated_deltanet_num_heads,
            gated_deltanet_conv_kernel=self.gated_deltanet_conv_kernel,
        )

    def _resolve_compile_model(self) -> bool:
        """Default ON for Ada/Hopper+ (cap >= 8.9); OFF for Ampere/older. Overridable."""
        if self.quantization is not None and self.quantization.enabled:
            return False
        if self.compile_model is not None:
            return self.compile_model
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            major, minor = torch.cuda.get_device_capability()
            return (major, minor) >= (8, 9)
        except Exception:
            return False

    def get_hardware_config(self) -> HardwareConfig:
        """Get hardware config from preset or auto-detect."""
        return get_hardware_preset(self.hardware_preset)

    def get_data_config(self) -> DataConfig:
        """Get data config from preset."""
        return get_data_preset(self.data_preset)

    @staticmethod
    def _global_grad_accum_steps(batch_size: int, micro_batch_size: int) -> int:
        """Interpret batch_size as desired global batch across all ranks."""
        return scale_grad_accum_for_world_size(batch_size, micro_batch_size)

    def get_pretrain_config(self) -> TrainingConfig:
        """Build TrainingConfig for pretraining stage."""
        return TrainingConfig(
            run_name=f"{self.run_name}-pretrain",
            dataset_name="wikitext",
            tokenizer_name=self.tokenizer_name,
            output_dir=Path(self.output_dir) / f"{self.run_name}-pretrain",
            batch_size=self.pretrain_batch_size,
            micro_batch_size=self.pretrain_micro_batch_size,
            gradient_accumulation_steps=self._global_grad_accum_steps(
                self.pretrain_batch_size,
                self.pretrain_micro_batch_size,
            ),
            learning_rate=self.pretrain_lr,
            max_steps=self.pretrain_max_steps,
            warmup_steps=self.pretrain_warmup_steps,
            min_lr_ratio=self.pretrain_min_lr_ratio,
            weight_decay=0.1,
            eval_every=self.eval_every,
            save_every=self.save_every,
            log_every=self.log_every,
            seed=self.seed,
            mixed_precision=self.mixed_precision,  # type: ignore
            compile_model=self._resolve_compile_model(),
            quantization=self.quantization,
        )

    def get_sft_config(self) -> TrainingConfig:
        """Build TrainingConfig for SFT stage."""
        return TrainingConfig(
            run_name=f"{self.run_name}-sft",
            dataset_name=self.sft_dataset,
            tokenizer_name=self.tokenizer_name,
            output_dir=Path(self.output_dir) / f"{self.run_name}-sft",
            batch_size=self.sft_batch_size,
            micro_batch_size=self.sft_micro_batch_size,
            gradient_accumulation_steps=self._global_grad_accum_steps(
                self.sft_batch_size,
                self.sft_micro_batch_size,
            ),
            learning_rate=self.sft_lr,
            max_steps=self.sft_max_steps,
            warmup_steps=100,
            weight_decay=0.01,
            eval_every=self.eval_every,
            save_every=self.save_every,
            log_every=self.log_every,
            seed=self.seed,
            mixed_precision=self.mixed_precision,  # type: ignore
            compile_model=self._resolve_compile_model(),
            quantization=self.quantization,
        )

    def get_dpo_config(self) -> TrainingConfig:
        """Build TrainingConfig for DPO stage."""
        return TrainingConfig(
            run_name=f"{self.run_name}-dpo",
            dataset_name=self.dpo_dataset,
            tokenizer_name=self.tokenizer_name,
            output_dir=Path(self.output_dir) / f"{self.run_name}-dpo",
            batch_size=self.dpo_batch_size,
            micro_batch_size=self.dpo_micro_batch_size,
            gradient_accumulation_steps=self._global_grad_accum_steps(
                self.dpo_batch_size,
                self.dpo_micro_batch_size,
            ),
            learning_rate=self.dpo_lr,
            max_steps=self.dpo_max_steps,
            warmup_steps=50,
            weight_decay=0.0,
            eval_every=self.eval_every,
            save_every=self.save_every,
            log_every=self.log_every,
            seed=self.seed,
            mixed_precision=self.mixed_precision,  # type: ignore
            compile_model=self._resolve_compile_model(),
            quantization=self.quantization,
        )

    def get_verifier_config(self) -> TrainingConfig:
        """Build TrainingConfig for verifier training."""
        return TrainingConfig(
            run_name=f"{self.run_name}-verifier",
            dataset_name="gsm8k",
            tokenizer_name=self.tokenizer_name,
            output_dir=Path(self.output_dir) / f"{self.run_name}-verifier",
            batch_size=self.verifier_batch_size,
            micro_batch_size=self.verifier_micro_batch_size,
            gradient_accumulation_steps=self._global_grad_accum_steps(
                self.verifier_batch_size,
                self.verifier_micro_batch_size,
            ),
            learning_rate=self.verifier_lr,
            max_steps=self.verifier_max_steps,
            warmup_steps=100,
            weight_decay=0.01,
            eval_every=self.eval_every,
            save_every=self.save_every,
            log_every=self.log_every,
            seed=self.seed,
            mixed_precision=self.mixed_precision,  # type: ignore
            compile_model=self._resolve_compile_model(),
            quantization=self.quantization,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    def save(self, path: Path | str) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> PipelineConfig:
        """Load config from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        """Create config from dictionary."""
        return cls(**data)


# Preset pipeline configurations
def local_smoke_config() -> PipelineConfig:
    return PipelineConfig(
    )


def local_full_config() -> PipelineConfig:
    return PipelineConfig(
        d_model=768,
    )


def gpu_smoke_config() -> PipelineConfig:
    return PipelineConfig(
    )



def gpu_full_config() -> PipelineConfig:
    """Full config for high-end GPU training (A100/H100).
    
    Optimized for quality with diverse data:
    - Wikipedia + OpenWebText + WikiText-103 for factual/general knowledge
    - TinyStories downsampled (100K samples) to avoid story-mode collapse
    - 80K pretrain steps for thorough training
    - Multiple SFT datasets for diverse instruction following
    
    Estimated time on H100:
    - Pretrain: 80K steps * 1.5s = 33h
    - SFT: 10K steps = 5h  
    - DPO: 3K steps = 2h
    - Verifier: 3K steps = 2h
    - Total: ~42h (under 48h limit)
    """
    print("spdaopsaodpsaodpodspoapdosapdoapdoas")
    return PipelineConfig(
            d_model=512,
        n_layers=10,
        n_heads=4,
        ffn_hidden_size=1024,
        max_seq_len=1024,
        use_attention_sinks=False,  # Disable to enable Flash Attention
        hardware_preset="auto",
        data_preset="xl",
        pretrain_datasets=[
            "wikitext-103-raw-v1",
            "openwebtext",
            "wikipedia",
            "EleutherAI/fineweb-edu-dedup-10b",
           # "HuggingFaceTB/smollm-corpus"
            "roneneldan/TinyStories",  # Downsample to 100K
        ],
        pretrain_max_steps=120000,
        pretrain_batch_size=64,
        pretrain_micro_batch_size=16,  # H100 can handle much larger
        sft_batch_size=64,
        sft_micro_batch_size=8,
        dpo_batch_size=32,
        dpo_micro_batch_size=8,
        verifier_batch_size =64,
        verifier_micro_batch_size=16,
        sft_max_steps=30000,
        sft_datasets=["QuixiAI/dolphin",
            "tatsu-lab/alpaca",
            "databricks/databricks-dolly-15k",
            "Open-Orca/OpenOrca",  # Sample 50K from larger dataset
        ],
        dpo_max_steps=1000,
        verifier_max_steps=1000,
        eval_every=10000,  # Eval only 4 times during 80K pretrain (was 2000)
        save_every=10000,
    )


def get_pipeline_preset(name: str) -> PipelineConfig:
    """Get a pipeline preset by name.

    Pre: name is one of "local-smoke", "local", "gpu-smoke", "gpu".
    """
    presets = {
        "local-smoke": local_smoke_config,
        "local": local_full_config,
        "gpu-smoke": gpu_smoke_config,
        "gpu": gpu_full_config,
    }
    if name not in presets:
        raise ValueError(f"Unknown pipeline preset: {name}. Choose from {list(presets.keys())}")
    return presets[name]()
