"""Orchestrates Base -> SFT -> DPO -> Verifier pipeline (RLHF-inspired).

References:
    - SFT: Ouyang et al., 2022 (InstructGPT).
    - DPO: Rafailov et al., 2023.
    - Verifier reranking: Lightman et al., 2023.

This module provides the full pipeline orchestration:
1. Pretrain or load base model
2. SFT on instruction data
3. DPO on preference data
4. Train verifier for reranking
5. Evaluate all stages and generate comparison metrics
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from modern_llm.config import ModernLLMConfig, PipelineConfig
from modern_llm.data.instruction_datasets import InstructionDatasetConfig
from modern_llm.data.preference_datasets import PreferenceDatasetConfig
from modern_llm.models.transformer import ModernDecoderLM
from modern_llm.models.verifier import VerifierConfig, VerifierModel
from modern_llm.training.train_dpo import DPOConfig, run_dpo
from modern_llm.training.train_sft import run_sft
from modern_llm.training.train_verifier import VerifierDatasetConfig, run_verifier_training
from modern_llm.utils.checkpointing import load_checkpoint
from modern_llm.utils.logging_utils import create_logger


@dataclass
class PipelineState:
    """Tracks checkpoint paths and metrics across pipeline stages."""

    pretrain_checkpoint: Optional[Path] = None
    sft_checkpoint: Optional[Path] = None
    dpo_checkpoint: Optional[Path] = None
    verifier_checkpoint: Optional[Path] = None

    pretrain_metrics: Optional[dict] = None
    sft_metrics: Optional[dict] = None
    dpo_metrics: Optional[dict] = None
    verifier_metrics: Optional[dict] = None

    def to_dict(self) -> dict:
        """Serialize state to dictionary."""
        return {
            "pretrain_checkpoint": str(self.pretrain_checkpoint) if self.pretrain_checkpoint else None,
            "sft_checkpoint": str(self.sft_checkpoint) if self.sft_checkpoint else None,
            "dpo_checkpoint": str(self.dpo_checkpoint) if self.dpo_checkpoint else None,
            "verifier_checkpoint": str(self.verifier_checkpoint) if self.verifier_checkpoint else None,
            "pretrain_metrics": self.pretrain_metrics,
            "sft_metrics": self.sft_metrics,
            "dpo_metrics": self.dpo_metrics,
            "verifier_metrics": self.verifier_metrics,
        }

    def save(self, path: Path) -> None:
        """Save state to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> PipelineState:
        """Load state from JSON file."""
        with open(path) as f:
            data = json.load(f)

        return cls(
            pretrain_checkpoint=Path(data["pretrain_checkpoint"]) if data.get("pretrain_checkpoint") else None,
            sft_checkpoint=Path(data["sft_checkpoint"]) if data.get("sft_checkpoint") else None,
            dpo_checkpoint=Path(data["dpo_checkpoint"]) if data.get("dpo_checkpoint") else None,
            verifier_checkpoint=Path(data["verifier_checkpoint"]) if data.get("verifier_checkpoint") else None,
            pretrain_metrics=data.get("pretrain_metrics"),
            sft_metrics=data.get("sft_metrics"),
            dpo_metrics=data.get("dpo_metrics"),
            verifier_metrics=data.get("verifier_metrics"),
        )


class AlignmentPipeline:
    """Orchestrates the full alignment pipeline.

    Stages:
        1. Pretrain: Language model pretraining (or load existing)
        2. SFT: Supervised fine-tuning on instructions
        3. DPO: Direct Preference Optimization
        4. Verifier: Train answer correctness model
        5. Evaluation: Compare all stages
    """

    def __init__(
        self,
        config: PipelineConfig,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.config = config
        self.checkpoint_dir = checkpoint_dir or Path("experiments/runs") / config.run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.logger = create_logger(f"pipeline.{config.run_name}")
        self.state = PipelineState()
        self.state_path = self.checkpoint_dir / "pipeline_state.json"

        # Try to load existing state
        if self.state_path.exists():
            self.logger.info(f"Loading pipeline state from {self.state_path}")
            self.state = PipelineState.load(self.state_path)

    def run(
        self,
        skip_pretrain: bool = False,
        pretrain_checkpoint: Optional[Path] = None,
        skip_sft: bool = False,
        skip_dpo: bool = False,
        skip_verifier: bool = False,
    ) -> PipelineState:
        """Execute the full pipeline.

        Pre:
            - Config specifies valid hyperparameters for all stages.
            - If skip_pretrain=True, pretrain_checkpoint must be provided.
        Post:
            - All checkpoints saved to checkpoint_dir.
            - State file updated after each stage.
            - Returns final PipelineState with all paths and metrics.
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Starting alignment pipeline: {self.config.run_name}")
        self.logger.info("=" * 60)

        # Stage 1: Pretrain (or reuse existing)
        if skip_pretrain:
            if pretrain_checkpoint:
                self.state.pretrain_checkpoint = pretrain_checkpoint
                self.logger.info(f"[Stage 1/4] Using provided pretrain checkpoint: {pretrain_checkpoint}")
            elif self.state.pretrain_checkpoint:
                self.logger.info(f"[Stage 1/4] Using saved pretrain checkpoint: {self.state.pretrain_checkpoint}")
            else:
                raise ValueError("No pretrain checkpoint available. Either run pretrain or provide one.")
        else:
            if self.state.pretrain_checkpoint and not pretrain_checkpoint:
                self.logger.info(
                    f"[Stage 1/4] Pretraining already completed, reusing checkpoint: {self.state.pretrain_checkpoint}"
                )
            else:
                self.logger.info("\n[Stage 1/4] Pretraining...")
                self.state.pretrain_checkpoint = self._run_pretrain()

        self._save_state()

        # Stage 2: SFT
        if not skip_sft:
            self.logger.info("\n[Stage 2/4] Supervised Fine-Tuning...")
            self.state.sft_checkpoint = self._run_sft()
        elif self.state.sft_checkpoint:
            self.logger.info(f"[Stage 2/4] Using saved SFT checkpoint: {self.state.sft_checkpoint}")
        else:
            self.logger.warning("[Stage 2/4] Skipping SFT (no checkpoint)")
            self.state.sft_checkpoint = self.state.pretrain_checkpoint

        self._save_state()

        # Stage 3: DPO
        if not skip_dpo:
            self.logger.info("\n[Stage 3/4] Direct Preference Optimization...")
            self.state.dpo_checkpoint = self._run_dpo()
        elif self.state.dpo_checkpoint:
            self.logger.info(f"[Stage 3/4] Using saved DPO checkpoint: {self.state.dpo_checkpoint}")
        else:
            self.logger.warning("[Stage 3/4] Skipping DPO (no checkpoint)")
            self.state.dpo_checkpoint = self.state.sft_checkpoint

        self._save_state()

        # Stage 4: Verifier
        if not skip_verifier:
            self.logger.info("\n[Stage 4/4] Training Verifier...")
            self.state.verifier_checkpoint = self._run_verifier()
        elif self.state.verifier_checkpoint:
            self.logger.info(f"[Stage 4/4] Using saved verifier: {self.state.verifier_checkpoint}")
        else:
            self.logger.warning("[Stage 4/4] Skipping verifier training")

        self._save_state()

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Pipeline complete!")
        self.logger.info("=" * 60)
        self._log_summary()

        return self.state

    def _run_pretrain(self) -> Path:
        """Run pretraining stage."""
        # Import here to avoid circular imports
        from modern_llm.training.train_lm import run_training

        train_config = self.config.get_pretrain_config()
        # Override output_dir to use checkpoint_dir
        train_config.output_dir = self.checkpoint_dir / train_config.run_name
        train_config.output_dir.mkdir(parents=True, exist_ok=True)
        model_config = self.config.get_model_config()

        # Pass datasets from config (defaults to wikitext-2 if not specified)
        dataset_names = self.config.pretrain_datasets
        checkpoint = run_training(
            model_config,
            train_config,
            dataset_names=dataset_names,
            tokenizer_name=self.config.tokenizer_name,
            packed_shards_dir=self.config.pretrain_packed_shards,
            packed_eval_windows=self.config.pretrain_eval_windows,
        )
        return checkpoint

    def _run_sft(self) -> Path:
        """Run SFT stage.

        If `config.sft_datasets` is a non-empty list, build an interleaved
        mixture via `build_sft_mixture` with optional `sft_dataset_weights`
        (uniform if unset). Otherwise fall back to the single-dataset path
        defined by `config.sft_dataset`.
        """
        train_config = self.config.get_sft_config()
        # Override output_dir to use checkpoint_dir
        train_config.output_dir = self.checkpoint_dir / train_config.run_name
        train_config.output_dir.mkdir(parents=True, exist_ok=True)

        use_mixture = bool(self.config.sft_datasets)

        # dataset_config is still required by run_sft's signature and drives
        # the eval split when present. For the mixture path we point it at
        # the first listed dataset so eval has a concrete source.
        primary_name = (
            self.config.sft_datasets[0] if use_mixture else self.config.sft_dataset
        )
        dataset_config = InstructionDatasetConfig(
            dataset_name=primary_name,
            max_length=self.config.max_seq_len,
        )

        pre_built_dataset = None
        if use_mixture:
            from modern_llm.data.sft_mixture import build_sft_mixture

            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            self.logger.info(
                f"Building SFT mixture over {len(self.config.sft_datasets)} "
                f"datasets (weights={'uniform' if self.config.sft_dataset_weights is None else self.config.sft_dataset_weights})"
            )
            pre_built_dataset = build_sft_mixture(
                dataset_names=self.config.sft_datasets,
                weights=self.config.sft_dataset_weights,
                tokenizer=tokenizer,
                max_length=self.config.max_seq_len,
                seed=self.config.seed,
            )

        return run_sft(
            pretrain_checkpoint=self.state.pretrain_checkpoint,
            train_config=train_config,
            dataset_config=dataset_config,
            tokenizer_name=self.config.tokenizer_name,
            train_dataset=pre_built_dataset,
        )

    def _run_dpo(self) -> Path:
        """Run DPO stage."""
        train_config = self.config.get_dpo_config()
        # Override output_dir to use checkpoint_dir
        train_config.output_dir = self.checkpoint_dir / train_config.run_name
        train_config.output_dir.mkdir(parents=True, exist_ok=True)
        dpo_config = DPOConfig(
            beta=self.config.dpo_beta,
            max_length=self.config.max_seq_len,
        )
        preference_config = PreferenceDatasetConfig(
            dataset_name=self.config.dpo_dataset,
        )

        return run_dpo(
            sft_checkpoint=self.state.sft_checkpoint,
            train_config=train_config,
            dpo_config=dpo_config,
            preference_config=preference_config,
            tokenizer_name=self.config.tokenizer_name,
        )

    def _run_verifier(self) -> Path:
        """Run verifier training stage."""
        train_config = self.config.get_verifier_config()
        # Override output_dir to use checkpoint_dir
        train_config.output_dir = self.checkpoint_dir / train_config.run_name
        train_config.output_dir.mkdir(parents=True, exist_ok=True)
        verifier_config = VerifierConfig(
            vocab_size=50257,
            d_model=512,
            num_layers=4,
            n_heads=8,
            max_position_embeddings=self.config.max_seq_len,
        )
        dataset_config = VerifierDatasetConfig(
            max_length=self.config.max_seq_len,
        )

        return run_verifier_training(
            train_config=train_config,
            verifier_config=verifier_config,
            dataset_config=dataset_config,
            tokenizer_name=self.config.tokenizer_name,
        )

    def _save_state(self) -> None:
        """Save current pipeline state."""
        self.state.save(self.state_path)
        self.logger.info(f"State saved to {self.state_path}")

    def _log_summary(self) -> None:
        """Log pipeline summary."""
        self.logger.info("\nCheckpoints:")
        self.logger.info(f"  Pretrain: {self.state.pretrain_checkpoint}")
        self.logger.info(f"  SFT:      {self.state.sft_checkpoint}")
        self.logger.info(f"  DPO:      {self.state.dpo_checkpoint}")
        self.logger.info(f"  Verifier: {self.state.verifier_checkpoint}")

    def load_model(self, stage: str) -> ModernDecoderLM:
        """Load model from a specific stage.

        Pre: stage is one of "pretrain", "sft", "dpo".
        Post: Returns loaded model on appropriate device.
        """
        checkpoint_map = {
            "pretrain": self.state.pretrain_checkpoint,
            "sft": self.state.sft_checkpoint,
            "dpo": self.state.dpo_checkpoint,
        }

        if stage not in checkpoint_map:
            raise ValueError(f"Unknown stage: {stage}. Choose from {list(checkpoint_map.keys())}")

        checkpoint_path = checkpoint_map[stage]
        if not checkpoint_path or not checkpoint_path.exists():
            raise ValueError(f"No checkpoint for stage '{stage}'")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = load_checkpoint(checkpoint_path)

        config = ModernLLMConfig(**ckpt["config"])
        model = ModernDecoderLM(config)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)

        return model

    def load_verifier(self) -> VerifierModel:
        """Load trained verifier model.

        Pre: Verifier checkpoint exists.
        Post: Returns loaded verifier on appropriate device.
        """
        if not self.state.verifier_checkpoint or not self.state.verifier_checkpoint.exists():
            raise ValueError("No verifier checkpoint available")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = load_checkpoint(self.state.verifier_checkpoint)

        config = VerifierConfig(**ckpt["config"])
        model = VerifierModel(config)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)

        return model


def run_alignment_pipeline(
    config: PipelineConfig,
    checkpoint_dir: Optional[Path] = None,
    pretrain_checkpoint: Optional[Path] = None,
    skip_pretrain: bool = False,
    skip_sft: bool = False,
    skip_dpo: bool = False,
    skip_verifier: bool = False,
) -> PipelineState:
    """Execute Base -> SFT -> DPO -> Verifier with shared evaluation tables.

    Pre:
        - Config specifies valid hyperparameters.
        - If skip_pretrain, pretrain_checkpoint must be provided.
    Post:
        - Logs metric deltas after each stage and verifier impact.
        - Returns PipelineState with all checkpoint paths.
    """
    pipeline = AlignmentPipeline(config, checkpoint_dir)
    return pipeline.run(
        skip_pretrain=skip_pretrain,
        pretrain_checkpoint=pretrain_checkpoint,
        skip_sft=skip_sft,
        skip_dpo=skip_dpo,
        skip_verifier=skip_verifier,
    )
