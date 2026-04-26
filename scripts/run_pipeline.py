#!/usr/bin/env python3
"""Unified pipeline entry point for Modern LLM.

Single command to run any stage or the full pipeline.
Runs in the current Python environment.

Usage:
    # Smoke test (5 minutes)
    python scripts/run_pipeline.py --config local-smoke --stage all

    # Run just pretrain
    python scripts/run_pipeline.py --config local --stage pretrain

    # Resume SFT from existing pretrain checkpoint
    python scripts/run_pipeline.py --config local --stage sft --checkpoint experiments/runs/pretrain_final.pt

    # Full GPU pipeline
    python scripts/run_pipeline.py --config gpu --stage all --output-dir /path/to/checkpoints

    # Run with custom config file
    python scripts/run_pipeline.py --config configs/custom.json --stage all
"""

import argparse
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
# Apply HF cache redirects as soon as we know where the repo is. Done
# *before* any HF import so cache settings stick.
from modern_llm.utils.paths import apply_env_defaults  # noqa: E402

apply_env_defaults()

from modern_llm.config import PipelineConfig, get_pipeline_preset  # noqa: E402


VALID_STAGES = {"pretrain", "sft", "dpo", "verifier", "eval", "all"}


def _infer_sft_examples_per_dataset(config: PipelineConfig, dataset_count: int) -> int:
    """Infer a practical per-source cap for SFT mixtures.

    SFT only consumes roughly max_steps * global_batch examples. Loading full
    multi-million-row instruction sources before a 4k-step run can look like a
    hang, so cap each source to the amount needed for this run unless the config
    explicitly chooses a cap.
    """
    if dataset_count <= 0:
        raise ValueError("dataset_count must be positive")
    if config.sft_num_examples_per_dataset is not None:
        if config.sft_num_examples_per_dataset <= 0:
            raise ValueError("sft_num_examples_per_dataset must be positive when set")
        return config.sft_num_examples_per_dataset
    examples_needed = max(1, config.sft_max_steps * config.sft_batch_size)
    return max(1024, math.ceil(examples_needed / dataset_count))


def _maybe_self_spawn_under_torchrun(nproc_per_node: int) -> None:
    """If the user requested >1 process and we are not already a torchrun
    child, re-exec the current script under `torchrun --standalone
    --nproc_per_node=N`.

    Pre: nproc_per_node >= 1.
    Post: function returns only when the current process is a single-rank
          worker (either nproc_per_node==1, or we are already under torchrun).
          Otherwise it execvp's torchrun and never returns.

    This preserves the user's preferred invocation:
        python3 scripts/run_pipeline.py --config gpu --stage all --nproc-per-node 2
    while still using the canonical torchrun launcher under the hood.
    """
    if nproc_per_node <= 1:
        return
    # Already a torchrun child? torchrun sets all of these.
    if "TORCHELASTIC_RUN_ID" in os.environ or "WORLD_SIZE" in os.environ:
        return

    torchrun = shutil.which("torchrun") or shutil.which("torch.distributed.run")
    if torchrun is None:
        # Fall back to `python -m torch.distributed.run`.
        argv = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={nproc_per_node}",
            *sys.argv,
        ]
    else:
        # Drop --nproc-per-node from re-exec argv so torchrun children do not
        # try to recursively re-exec themselves.
        forwarded = [
            a for a in sys.argv
            if not a.startswith("--nproc-per-node") and not a.startswith("--nproc_per_node")
        ]
        # Also strip the value following the flag if it was passed separately.
        cleaned = []
        skip_next = False
        for a in forwarded:
            if skip_next:
                skip_next = False
                continue
            if a in ("--nproc-per-node", "--nproc_per_node"):
                skip_next = True
                continue
            cleaned.append(a)
        argv = [
            torchrun,
            "--standalone",
            f"--nproc_per_node={nproc_per_node}",
            *cleaned,
        ]
    print(f"[run_pipeline] self-spawning under torchrun: {' '.join(argv)}", flush=True)
    os.execvp(argv[0], argv)


def _report_exists(config: PipelineConfig) -> bool:
    """Check if a report already exists for this run."""
    report_dir = Path("report")
    report_path = report_dir / f"{config.run_name}_report.md"
    return report_path.exists()


def _protect_results(config: PipelineConfig, force: bool) -> None:
    """Check if results exist and abort if --force not set."""
    if _report_exists(config) and not force:
        report_path = Path("report") / f"{config.run_name}_report.md"
        raise FileExistsError(
            f"Report already exists at {report_path}. "
            f"Use --force to overwrite or change --run-name."
        )


def run_pretrain(config: PipelineConfig, output_dir: Path) -> Path:
    """Run pretraining stage."""
    from modern_llm.training.train_lm import run_training

    train_config = config.get_pretrain_config()
    train_config.output_dir = output_dir / train_config.run_name
    train_config.output_dir.mkdir(parents=True, exist_ok=True)

    model_config = config.get_model_config()
    dataset_names = config.pretrain_datasets

    return run_training(
        model_config,
        train_config,
        dataset_names=dataset_names,
        tokenizer_name=config.tokenizer_name,
        packed_shards_dir=config.pretrain_packed_shards,
    )


def run_sft(config: PipelineConfig, output_dir: Path, pretrain_checkpoint: Path) -> Path:
    """Run SFT stage."""
    from modern_llm.data.instruction_datasets import InstructionDatasetConfig
    from modern_llm.training.distributed import init_distributed, is_main_process, main_process_first
    from modern_llm.training.train_sft import run_sft as _run_sft
    from transformers import AutoTokenizer

    train_config = config.get_sft_config()
    train_config.output_dir = output_dir / train_config.run_name
    train_config.output_dir.mkdir(parents=True, exist_ok=True)

    use_mixture = bool(config.sft_datasets)
    primary_name = config.sft_datasets[0] if use_mixture else config.sft_dataset
    dataset_config = InstructionDatasetConfig(
        dataset_name=primary_name,
        max_length=config.max_seq_len,
    )

    train_dataset = None
    if use_mixture:
        from modern_llm.data.sft_mixture import build_sft_mixture

        init_distributed()
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        examples_per_dataset = _infer_sft_examples_per_dataset(config, len(config.sft_datasets))
        with main_process_first():
            if is_main_process():
                print(
                    f"Building SFT mixture over {len(config.sft_datasets)} datasets "
                    f"(weights={'uniform' if config.sft_dataset_weights is None else config.sft_dataset_weights}, "
                    f"cap_per_dataset={examples_per_dataset})",
                    flush=True,
                )
            train_dataset = build_sft_mixture(
                dataset_names=config.sft_datasets,
                weights=config.sft_dataset_weights,
                tokenizer=tokenizer,
                max_length=config.max_seq_len,
                seed=config.seed,
                num_examples_per_dataset=examples_per_dataset,
                log_fn=(lambda msg: print(msg, flush=True)) if is_main_process() else None,
            )

    return _run_sft(
        pretrain_checkpoint=pretrain_checkpoint,
        train_config=train_config,
        dataset_config=dataset_config,
        tokenizer_name=config.tokenizer_name,
        train_dataset=train_dataset,
    )


def run_dpo(config: PipelineConfig, output_dir: Path, sft_checkpoint: Path) -> Path:
    """Run DPO stage."""
    from modern_llm.data.preference_datasets import PreferenceDatasetConfig
    from modern_llm.training.train_dpo import DPOConfig, run_dpo as _run_dpo

    train_config = config.get_dpo_config()
    train_config.output_dir = output_dir / train_config.run_name
    train_config.output_dir.mkdir(parents=True, exist_ok=True)

    dpo_config = DPOConfig(
        beta=config.dpo_beta,
        max_length=config.max_seq_len,
    )
    preference_config = PreferenceDatasetConfig(
        dataset_name=config.dpo_dataset,
    )

    return _run_dpo(
        sft_checkpoint=sft_checkpoint,
        train_config=train_config,
        dpo_config=dpo_config,
        preference_config=preference_config,
        tokenizer_name=config.tokenizer_name,
        num_examples=config.dpo_num_examples,
    )


def run_verifier(config: PipelineConfig, output_dir: Path) -> Path:
    """Run verifier training stage."""
    from modern_llm.models.verifier import VerifierConfig
    from modern_llm.training.train_verifier import VerifierDatasetConfig, run_verifier_training

    train_config = config.get_verifier_config()
    train_config.output_dir = output_dir / train_config.run_name
    train_config.output_dir.mkdir(parents=True, exist_ok=True)

    verifier_config = VerifierConfig(
        vocab_size=50257,
        d_model=512,
        num_layers=4,
        n_heads=8,
        max_position_embeddings=config.max_seq_len,
    )
    dataset_config = VerifierDatasetConfig(
        max_length=config.max_seq_len,
    )

    return run_verifier_training(
        train_config=train_config,
        verifier_config=verifier_config,
        dataset_config=dataset_config,
        tokenizer_name=config.tokenizer_name,
    )


def run_eval(config: PipelineConfig, output_dir: Path) -> None:
    """Run evaluation on all available checkpoints."""
    from modern_llm.alignment.alignment_pipeline import PipelineState

    state_path = output_dir / "pipeline_state.json"
    if not state_path.exists():
        print(f"No pipeline state found at {state_path}, skipping evaluation.")
        return

    state = PipelineState.load(state_path)
    try:
        from modern_llm.evaluation.pipeline_eval import evaluate_pipeline_stages
        results = evaluate_pipeline_stages(state, config)
        print(f"Evaluation results saved to: {results}")
    except ImportError as e:
        print(f"Evaluation module not available: {e}")


def find_latest_checkpoint(output_dir: Path, stage: str) -> Optional[Path]:
    """Find the latest checkpoint for a given stage."""
    patterns = {
        "pretrain": ["*pretrain*final*.pt", "*pretrain*best*.pt", "*pretrain*step*.pt"],
        "sft": ["*sft*final*.pt", "*sft*best*.pt", "*sft*step*.pt"],
        "dpo": ["*dpo*final*.pt", "*dpo*best*.pt", "*dpo*step*.pt"],
        "verifier": ["*verifier*final*.pt", "*verifier*best*.pt"],
    }

    for pattern in patterns.get(stage, []):
        matches = sorted(output_dir.glob(f"**/{pattern}"), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return matches[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Modern LLM Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test
  python scripts/run_pipeline.py --config local-smoke --stage all

  # Run pretrain only
  python scripts/run_pipeline.py --config local --stage pretrain

  # Resume from existing pretrain checkpoint
  python scripts/run_pipeline.py --config local --stage sft \\
      --checkpoint experiments/runs/local-full/pretrain_final.pt

  # Full pipeline with custom output directory
  python scripts/run_pipeline.py --config gpu --stage all \\
      --output-dir /path/to/checkpoints

Config Presets:
  local-smoke  - Quick test (~5 min), tiny model
  local        - Full training for RTX 3060 (~24 hours)
  gpu-smoke    - Quick GPU test (~10 min)
  gpu          - Full high-end GPU training (~48 hours)

Stages:
  pretrain  - Pretrain language model on text corpora
  sft       - Supervised fine-tuning on instructions
  dpo       - Direct preference optimization
  verifier  - Train answer correctness model
  eval      - Run evaluation on existing checkpoints
  all       - Run full pipeline (pretrain -> sft -> dpo -> verifier)
""",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config preset (local-smoke, local, gpu-smoke, gpu) or path to JSON file",
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=sorted(VALID_STAGES),
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint for resuming (e.g., pretrain checkpoint for SFT)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for checkpoints (default: experiments/runs/<run_name>)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override run name from config",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override max training steps (useful for testing)",
    )
    parser.add_argument(
        "--pretrain-steps",
        type=int,
        default=None,
        help="Override pretrain max steps",
    )
    parser.add_argument(
        "--sft-steps",
        type=int,
        default=None,
        help="Override SFT max steps",
    )
    parser.add_argument(
        "--dpo-steps",
        type=int,
        default=None,
        help="Override DPO max steps",
    )
    parser.add_argument(
        "--verifier-steps",
        type=int,
        default=None,
        help="Override verifier max steps",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Override model/data max sequence length.",
    )
    parser.add_argument(
        "--pretrain-eval-windows",
        type=int,
        default=None,
        help="Override packed pretrain eval windows.",
    )
    parser.add_argument(
        "--sft-num-examples-per-dataset",
        type=int,
        default=None,
        help="Override per-source SFT mixture cap.",
    )
    parser.add_argument(
        "--dpo-num-examples",
        type=int,
        default=None,
        help="Override DPO preference pair cap.",
    )
    parser.add_argument(
        "--pretrain-datasets",
        type=str,
        default=None,
        help="Comma-separated list of pretrain datasets",
    )
    parser.add_argument(
        "--pretrain-packed-shards",
        type=str,
        default=None,
        help="Override config.pretrain_packed_shards (useful for packed-shard smoke subsets).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing reports/results",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help=(
            "Number of GPU processes to launch via torchrun. When >1, the "
            "script re-execs itself under `torchrun --standalone "
            "--nproc_per_node=N`. When 1 (default), runs in-process."
        ),
    )

    args = parser.parse_args()

    # Self-spawn under torchrun if multi-GPU was requested. Returns only on
    # rank workers (or in the single-process case).
    _maybe_self_spawn_under_torchrun(args.nproc_per_node)

    # Load config
    config_path = Path(args.config)

    if config_path.exists():
        print(f"Loading config from file: {args.config}")
        config = PipelineConfig.load(args.config)
    else:
        print(f"Using config preset: {args.config}")
        config = get_pipeline_preset(args.config)
    # Apply overrides
    from datetime import datetime

# Get current time
    now = datetime.now()
    if args.run_name:
        config.run_name = args.run_name+str(now)
    if args.max_steps:
        config.pretrain_max_steps = args.max_steps
        config.sft_max_steps = min(args.max_steps, config.sft_max_steps)
        config.dpo_max_steps = min(args.max_steps, config.dpo_max_steps)
        config.verifier_max_steps = min(args.max_steps, config.verifier_max_steps)
    if args.pretrain_steps:
        config.pretrain_max_steps = args.pretrain_steps
    if args.sft_steps:
        config.sft_max_steps = args.sft_steps
    if args.dpo_steps:
        config.dpo_max_steps = args.dpo_steps
    if args.verifier_steps:
        config.verifier_max_steps = args.verifier_steps
    if args.pretrain_datasets:
        config.pretrain_datasets = [d.strip() for d in args.pretrain_datasets.split(",")]
    if args.pretrain_packed_shards:
        config.pretrain_packed_shards = args.pretrain_packed_shards
    if args.max_seq_len:
        config.max_seq_len = args.max_seq_len
    if args.pretrain_eval_windows is not None:
        config.pretrain_eval_windows = args.pretrain_eval_windows
    if args.sft_num_examples_per_dataset is not None:
        config.sft_num_examples_per_dataset = args.sft_num_examples_per_dataset
    if args.dpo_num_examples is not None:
        config.dpo_num_examples = args.dpo_num_examples

    # Set output directory
    output_dir = args.output_dir or Path("experiments/runs") / config.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Banner on rank 0 only so multi-rank logs stay readable.
    rank0 = int(os.environ.get("RANK", "0")) == 0
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if rank0:
        print()
        print("=" * 60)
        print("Modern LLM Pipeline")
        print("=" * 60)
        print(f"Config:       {args.config}")
        print(f"Run name:     {config.run_name}")
        print(f"Stage:        {args.stage}")
        print(f"Output dir:   {output_dir}")
        print(f"Model:        d={config.d_model}, L={config.n_layers}, H={config.n_heads}")
        print(f"World size:   {world}")
        print(f"Start time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        print()

    # Check for existing results
    if args.stage in {"all", "eval"}:
        try:
            _protect_results(config, args.force)
        except FileExistsError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    # Execute stage(s)
    if args.stage == "pretrain":
        ckpt = run_pretrain(config, output_dir)
        print(f"\nPretrain complete: {ckpt}")

    elif args.stage == "sft":
        pretrain_ckpt = args.checkpoint or find_latest_checkpoint(output_dir, "pretrain")
        if not pretrain_ckpt or not pretrain_ckpt.exists():
            print("ERROR: No pretrain checkpoint found. Run pretrain first or provide --checkpoint.")
            sys.exit(1)
        print(f"Using pretrain checkpoint: {pretrain_ckpt}")
        ckpt = run_sft(config, output_dir, pretrain_ckpt)
        print(f"\nSFT complete: {ckpt}")

    elif args.stage == "dpo":
        sft_ckpt = args.checkpoint or find_latest_checkpoint(output_dir, "sft")
        if not sft_ckpt or not sft_ckpt.exists():
            print("ERROR: No SFT checkpoint found. Run SFT first or provide --checkpoint.")
            sys.exit(1)
        print(f"Using SFT checkpoint: {sft_ckpt}")
        ckpt = run_dpo(config, output_dir, sft_ckpt)
        print(f"\nDPO complete: {ckpt}")

    elif args.stage == "verifier":
        ckpt = run_verifier(config, output_dir)
        print(f"\nVerifier training complete: {ckpt}")

    elif args.stage == "eval":
        run_eval(config, output_dir)

    elif args.stage == "all":
        # Run full pipeline via AlignmentPipeline
        from modern_llm.alignment.alignment_pipeline import run_alignment_pipeline

        state = run_alignment_pipeline(
            config=config,
            checkpoint_dir=output_dir,
            pretrain_checkpoint=args.checkpoint,
            skip_pretrain=bool(args.checkpoint),
        )

        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print("=" * 60)
        print("Checkpoints:")
        print(f"  Pretrain: {state.pretrain_checkpoint}")
        print(f"  SFT:      {state.sft_checkpoint}")
        print(f"  DPO:      {state.dpo_checkpoint}")
        print(f"  Verifier: {state.verifier_checkpoint}")

    print(f"\nFinished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
