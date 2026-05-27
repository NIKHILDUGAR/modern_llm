# HyperReadme

This document is a deep map of the current `modern_llm` repository. It explains
how the launch scripts enter the code, how training/evaluation flows through the
package, which files own which responsibilities, and which top-level functions
and classes are connected to each other.

Scope note: this file documents source, scripts, configs, tests, reports, and
important artifacts that are present in the repo. It intentionally does not
enumerate Python bytecode cache files, large data directories, training logs, or
checkpoint/result payloads as primary code, because those are generated outputs.

## Executive Summary

The repo is a scratch LLM training and evaluation system. The primary model is
`ModernDecoderLM`, a GPT-style decoder with RoPE, RMSNorm, SwiGLU, GQA,
optional QK-norm, optional Gated DeltaNet layers, and optional low-bit training
wrappers. The primary orchestration path is:

```text
scripts/launch.sh or scripts/triple_launch*.sh
-> scripts/run_pipeline.py
-> PipelineConfig
-> AlignmentPipeline for stage=all, or direct stage functions for one stage
-> train_lm / train_sft / train_dpo / train_verifier
-> Trainer or DPOTrainer or VerifierTrainer
-> ModernDecoderLM / VerifierModel
-> checkpointing and evaluation scripts
```

Three major experiment families are currently configured:

- Dense regular model: `configs/lm_75m_2x4090.json`
- BitNet quantized model: `configs/lm_75m_2x4090_bitnet.json`
- Hybrid Gated DeltaNet model: `configs/lm_75m_2x4090_gated_deltanet.json`

The BitNet config has been scaled to a 125M-class architecture:

```text
run_name: lm-125m-2x4090-bitnet
d_model: 640
n_layers: 26
n_heads: 10
ffn_hidden_size: 1792
max_seq_len: 4096
dense/shadow params: 125.4M
BitNet-replaced modules: 132
micro_batch_size: 2 for pretrain/SFT/DPO/verifier
```

## Launch Script Flow

### `scripts/launch.sh`

`launch.sh` is the simple wrapper. It:

1. Computes `REPO_ROOT`.
2. Exports Hugging Face cache directories under the repo data volume.
3. Exports NCCL settings for consumer multi-GPU boxes.
4. Changes directory to repo root.
5. Runs:

```text
numactl --cpunodebind=1 python3 scripts/run_pipeline.py "$@"
```

6. After the pipeline command returns, it runs:

```text
numactl --cpunodebind=1 python3 scripts/evaluation/run_eval_sweep.py \
  --runs-dir experiments/runs \
  --output-root experiments/results/sweep \
  --gpus 0 1 2 3
```

The important detail is that `launch.sh` does not itself parse model config or
own training logic. It forwards all user arguments to `run_pipeline.py`.

### `scripts/triple_launch.sh`

`triple_launch.sh` is the reliability sweep launcher. It runs variants
sequentially:

1. Regular dense.
2. BitNet quantized.
3. Gated DeltaNet.
4. Evaluation sweep.

It defaults to:

```text
STAGE=all
NPROC_PER_NODE=4
VARIANTS="regular bitnet gated"
PRETRAIN_PACKED_SHARDS=data/tokenized/pretrain_mix_40b_balanced_10shard
TRAIN_STEPS=1000
PRETRAIN_STEPS=1000
SFT_STEPS=1000
DPO_STEPS=1000
VERIFIER_STEPS=1000
```

It also supports:

- `VARIANTS=gated` to run only Gated DeltaNet.
- `VARIANTS="bitnet gated"` to skip regular.
- `DRY_RUN=true` to print commands without running them.
- `RUN_ID=...` to resume or group outputs.
- `PIPELINE_FORCE=true` to pass `--force` to `run_pipeline.py`.

The internal function `maybe_run` prints every command with shell quoting and
executes it only when `DRY_RUN != true`. This makes the launcher safe to audit
before starting long training runs.

### `scripts/triple_launch_full.sh`

`triple_launch_full.sh` is the full-training counterpart. It follows the same
variant flow as `triple_launch.sh`, but defaults to:

```text
PRETRAIN_PACKED_SHARDS=data/tokenized/pretrain_mix_40b_balanced
no *_STEPS override
EVAL_FAST=false
EVAL_NO_BASELINES=false
```

Use this only after `data/tokenized/pretrain_mix_40b_balanced/index.json`
exists. It intentionally does not create a 10-shard subset.

## `run_pipeline.py` Control Flow

`scripts/run_pipeline.py` is the central CLI entry point.

High-level flow:

```text
main()
-> parse CLI args
-> _maybe_self_spawn_under_torchrun(args.nproc_per_node)
-> load PipelineConfig from JSON or preset
-> apply CLI overrides
-> create output dir
-> dispatch by stage
```

Stage dispatch:

```text
stage=pretrain -> run_pretrain()
stage=sft      -> find pretrain checkpoint -> run_sft()
stage=dpo      -> find SFT checkpoint -> run_dpo()
stage=verifier -> run_verifier()
stage=eval     -> run_eval()
stage=all      -> run_alignment_pipeline()
```

Important functions:

- `_maybe_self_spawn_under_torchrun`: if `--nproc-per-node > 1` and the process
  is not already under torchrun, it re-execs the same command under torchrun.
- `_infer_sft_examples_per_dataset`: caps SFT sources so short runs do not
  materialize huge datasets unnecessarily.
- `run_pretrain`: builds LM training config and calls `train_lm.run_training`.
- `run_sft`: optionally builds SFT mixture, then calls `train_sft.run_sft`.
- `run_dpo`: builds preference config and calls `train_dpo.run_dpo`.
- `run_verifier`: builds verifier config and calls `run_verifier_training`.
- `find_latest_checkpoint`: finds stage checkpoints for manual stage resumes.
- `main`: owns CLI, config loading, overrides, and stage dispatch.

## Stage Flow

### Stage 1: Pretraining

```text
run_pipeline.run_pretrain
-> PipelineConfig.get_pretrain_config
-> PipelineConfig.get_model_config
-> train_lm.run_training
-> ModernDecoderLM
-> optional prepare_model_for_quantization
-> load_packed_pretrain_train_eval_split or make_lm_dataloader
-> Trainer.train
-> ModernDecoderLM.forward
-> Trainer._save_checkpoint
```

Key files:

- `src/modern_llm/training/train_lm.py`
- `src/modern_llm/training/trainer_base.py`
- `src/modern_llm/models/transformer.py`
- `src/modern_llm/data/lm_datasets.py`
- `src/modern_llm/quantization/prepare.py`

### Stage 2: Supervised Fine-Tuning

```text
run_pipeline.run_sft or AlignmentPipeline._run_sft
-> build_sft_mixture when config.sft_datasets is set
-> train_sft.run_sft
-> load_pretrained_model
-> optional prepare_model_for_quantization
-> create_instruction_dataloader
-> Trainer.train
-> checkpoint
```

Important data adapters live in `src/modern_llm/data/sft_mixture.py`. They
normalize chat, ShareGPT, Alpaca, MetaMathQA, OpenMathInstruct, CoQA, and
prompt/response styles into a shared instruction/input/output shape.

### Stage 3: DPO

```text
run_pipeline.run_dpo or AlignmentPipeline._run_dpo
-> PreferenceDatasetConfig
-> train_dpo.run_dpo
-> load_model_from_checkpoint
-> optional prepare_model_for_quantization
-> PreferenceDataset
-> DPOTrainer.train
-> dpo_loss
-> checkpoint
```

DPO has a custom trainer because it compares chosen/rejected completions and
logs preference accuracy rather than ordinary LM perplexity.

### Stage 4: Verifier

```text
run_pipeline.run_verifier or AlignmentPipeline._run_verifier
-> VerifierConfig
-> VerifierDatasetConfig
-> run_verifier_training
-> VerifierDataset
-> VerifierTrainer.train
-> VerifierModel
-> checkpoint
```

Verifier training is a separate binary classification/regression style path for
answer correctness scoring.

### Evaluation Sweep

```text
run_eval_sweep.py
-> discover_checkpoints
-> build_jobs
-> worker_loop per GPU
-> eval_all.py per checkpoint/baseline
-> task scripts
-> summary JSON and CSV
```

`eval_all.py` is the per-model orchestrator. It calls task-specific scripts
such as `eval_hellaswag.py`, `eval_mmlu.py`, `eval_sst2.py`, and `eval_gsm8k.py`.
Shared model loading and scoring helpers live in `scripts/evaluation/_eval_common.py`.

## Model Flow

### `ModernDecoderLM`

`ModernDecoderLM` lives in `src/modern_llm/models/transformer.py`.

Forward path:

```text
input_ids
-> token_embed
-> optional embedding scale
-> dropout
-> attention mask -> causal attention bias
-> each DecoderBlock
   -> RMSNorm
   -> MultiHeadAttention or GatedDeltaNet
   -> residual
   -> RMSNorm
   -> SwiGLU or MoE FFN
   -> residual
-> final_norm
-> lm_head
-> optional cross_entropy loss
-> optional z_loss
-> {"logits", "loss"}
```

The model exposes `iter_quantizable_linear_layers`, which is the stable module
traversal hook used by quantization preparation.

### Dense Attention

`MultiHeadAttention` lives in `src/modern_llm/models/attention.py`.

It owns:

- Q/K/V/out projections.
- RoPE.
- optional attention sinks.
- optional QK RMSNorm.
- optional GQA.
- PyTorch SDPA/Flash Attention path.

### Gated DeltaNet

`GatedDeltaNet` lives in `src/modern_llm/models/gated_deltanet.py`.

It is an opt-in sequence mixer with:

- Q/K/V/G projections.
- learned retention/update gates.
- short causal depthwise conv on Q/K/V.
- recurrent per-head associative state.
- output RMSNorm and output projection.

The current implementation is a reference PyTorch loop. It is useful for
experiments and smoke tests, but it is not yet an optimized long-context kernel.

### Quantization

Quantization lives under `src/modern_llm/quantization/` and is fully opt-in.

Flow:

```text
PipelineConfig.quantization
-> TrainingConfig.quantization
-> train_lm/train_sft/train_dpo prepare_model_for_quantization
-> ModernDecoderLM.iter_quantizable_linear_layers
-> replace eligible nn.Linear modules with BitLinear or QAT wrapper
-> Trainer/DPOTrainer set_quantization_step every step
-> checkpoint saves dense-compatible state plus quantization metadata
```

Modes:

- `bitnet_b1_58`: ternary BitNet-style research path.
- `qat_8da4w`: fake quantized int8 activation/int4 weight fallback path.
- `none`: dense path unchanged.

## Checkpoint Flow

Training checkpoints are saved by `src/modern_llm/utils/checkpointing.py`.

Primary payload fields:

- `model_state`
- `optimizer_state`
- `step`
- `run_name`
- `config`
- optional `quantization`

The loading helpers normalize older key names and can reapply quantization
wrappers before loading quantized-training states.

## Data Flow

### Packed Pretrain Data

`scripts/data/tokenize_pretrain.py` streams HF datasets, tokenizes examples, and
writes flat uint32 shards plus `index.json`.

`src/modern_llm/data/lm_datasets.py` reads those shards through
`PackedShardDataset`, slices them into fixed `max_seq_len` windows, and creates
train/eval splits.

### SFT Data

`src/modern_llm/data/sft_mixture.py` loads multiple instruction datasets,
normalizes schema differences, interleaves with optional weights, tokenizes,
and masks labels so only response tokens contribute loss.

### DPO Data

`src/modern_llm/data/preference_datasets.py` resolves known dataset split quirks
such as UltraFeedback's `train_prefs` split. `train_dpo.PreferenceDataset`
then tokenizes chosen/rejected pairs.

## Config Files

- `configs/README.md`: explains available config recipes.
- `configs/lm_75m_2x4090.json`: dense 75M-class regular model recipe.
- `configs/lm_75m_2x4090_bitnet.json`: BitNet recipe, now 125.4M params.
- `configs/lm_75m_2x4090_gated_deltanet.json`: hybrid Gated DeltaNet recipe,
  currently 56.7M params and shorter context for memory safety.
- `configs/lm_75m_2x4090_smoke.json`: smoke-test variant.
- `configs/lm_75m_2x4090_smoketest.json`: alternate smoke-test variant.
- `configs/lm_75m_2x4090_v2.json`: alternate 75M recipe.
- `configs/lm_max_rtx3060.json`: config targeting a smaller local GPU.
- `configs/local_rtx3060.json`: local RTX 3060 configuration.
- `configs/smoke_test.json`: small generic smoke config.

## Source Package File Map

### Package Roots

- `src/modern_llm/__init__.py`: package marker.
- `src/modern_llm/report.py`: report-generation module, currently has a parse
  error around line 79 and should be fixed before importing directly.

### `src/modern_llm/alignment`

- `alignment/__init__.py`: package marker.
- `alignment/alignment_pipeline.py`: orchestrates full pretrain -> SFT -> DPO
  -> verifier pipeline.
  - `PipelineState.to_dict/save/load`: serialize stage checkpoint paths.
  - `AlignmentPipeline.__init__`: creates logger, state path, output dirs.
  - `_infer_sft_examples_per_dataset`: calculates SFT source caps.
  - `run`: full stage orchestration and resume from saved state.
  - `_run_pretrain`: calls `train_lm.run_training`.
  - `_run_sft`: builds optional SFT mixture and calls `train_sft.run_sft`.
  - `_run_dpo`: calls `train_dpo.run_dpo`.
  - `_run_verifier`: calls `train_verifier.run_verifier_training`.
  - `_save_state`: writes `pipeline_state.json`.
  - `_log_summary`: logs final checkpoint summary.
  - `load_model`: helper to load a `ModernDecoderLM` from a checkpoint.
  - `load_verifier`: helper to load verifier checkpoint.
  - `run_alignment_pipeline`: convenience function used by `run_pipeline.py`.
- `alignment/dpo_loss.py`: DPO objective.
  - `dpo_loss`: computes pairwise logistic preference loss from chosen and
    rejected sequence log-probs.

### `src/modern_llm/config`

- `config/__init__.py`: exports config dataclasses.
- `config/hardware_config.py`: hardware and data presets.
  - `HardwareConfig.__post_init__`: validates hardware fields.
  - `HardwareConfig.from_env`: infers CUDA/DDP state from environment.
  - `HardwareConfig.get_torch_device`: returns `torch.device`.
  - `DataConfig.__post_init__`: validates data preset fields.
  - `get_hardware_preset`: returns named hardware preset or auto.
  - `get_data_preset`: returns named data scale preset.
- `config/model_config.py`: model architecture dataclasses.
  - `MoEConfig.__post_init__`: validates MoE settings.
  - `ModernLLMConfig.__post_init__`: runs all architecture validation.
  - `_validate_dimensions`: validates vocab, hidden size, layers, heads, seq.
  - `_validate_attention_settings`: validates head/GQA/RoPE constraints.
  - `_validate_moe_settings`: validates MoE config consistency.
  - `_validate_sequence_mixer_settings`: validates attention/DeltaNet choice.
  - `uses_gated_deltanet_layer`: tells `DecoderBlock` which mixer to build.
- `config/pipeline_config.py`: end-to-end pipeline config.
  - `PipelineConfig.__post_init__`: converts quantization dicts to dataclass.
  - `get_model_config`: creates `ModernLLMConfig`.
  - `_resolve_compile_model`: chooses compile setting, disabled for quant.
  - `get_hardware_config`: resolves hardware preset.
  - `get_data_config`: resolves data preset.
  - `_global_grad_accum_steps`: interprets batch size as global batch.
  - `get_pretrain_config/get_sft_config/get_dpo_config/get_verifier_config`:
    create stage-specific `TrainingConfig`.
  - `to_dict/save/load/from_dict`: JSON serialization helpers.
  - `local_smoke_config/local_full_config/gpu_smoke_config/gpu_full_config`:
    preset builders.
  - `get_pipeline_preset`: name-to-preset dispatcher.
- `config/train_config.py`: shared trainer hyperparameters.
  - `TrainingConfig.__post_init__`: validates training fields and quant config.
  - `_validate_positive_int`: helper validation.
  - `_validate_non_negative_int`: helper validation.

### `src/modern_llm/data`

- `data/__init__.py`: package marker.
- `data/instruction_datasets.py`: single-source instruction datasets.
  - `InstructionDatasetConfig.__post_init__`: validates max length.
  - `format_instruction`: formats instruction/input/output into text.
  - `InstructionDataset.__init__`: loads and tokenizes examples.
  - `_load_and_process`: loads HF rows and formats them.
  - `_format_example`: converts one raw row into instruction text.
  - `_tokenize`: creates input ids, masks, labels.
  - `__len__/__getitem__`: PyTorch dataset API.
  - `load_instruction_dataset`: convenience dataset loader.
  - `create_instruction_dataloader`: DataLoader factory.
- `data/lm_datasets.py`: pretraining text and packed-shard datasets.
  - `LanguageModelingDatasetConfig.__post_init__`: validates seq length.
  - `_require_datasets`: import-gated HF datasets dependency.
  - `load_causal_lm_dataset`: loads and tokenizes one LM dataset.
  - `make_lm_dataloader`: wraps LM dataset into DataLoader.
  - `_parse_dataset_spec`: parses dataset:config specs.
  - `load_multi_dataset`: loads/interleaves multiple raw datasets.
  - `PackedShardDataset.__init__`: reads shard manifest.
  - `PackedShardDataset.__len__`: returns number of windows.
  - `_get_shard`: locates shard for global token offset.
  - `_read_flat`: reads raw uint32 tokens from shard.
  - `__getitem__`: returns `input_ids`, `attention_mask`, `labels`.
  - `load_packed_pretrain_dataset`: single dataset loader.
  - `load_packed_pretrain_train_eval_split`: train/eval split helper.
- `data/preference_datasets.py`: DPO preference datasets.
  - `PreferenceDatasetConfig.__post_init__`: validates max length.
  - `_require_datasets`: import-gated dependency check.
  - `_resolve_preference_load_args`: maps known split names.
  - `_extract_prompt_and_response_hh`: parses HH-style prompt/response.
  - `_process_hh_rlhf`: adapts Anthropic HH rows.
  - `load_preference_dataset`: loads and normalizes preference rows.
- `data/sft_mixture.py`: multi-source SFT mixture builder.
  - `_last_user_assistant_pair`: extracts last user/assistant messages.
  - `_adapt_messages/_adapt_sharegpt/_adapt_alpaca/_adapt_prompt_response`:
    schema adapters.
  - `_adapt_metamathqa/_adapt_openmathinstruct2/_adapt_coqa`: dataset-specific
    adapters.
  - `get_adapter`: chooses adapter for dataset name.
  - `_MixtureExample`: internal typed row container.
  - `SFTMixtureDataset.__init__/__len__/__getitem__`: tokenized SFT dataset.
  - `_tokenize_with_response_mask`: masks prompt labels.
  - `_normalize_weights`: validates/normalizes mixture weights.
  - `_resolve_dataset_load_args`: resolves HF config names.
  - `_load_raw`: calls HF `load_dataset`.
  - `build_sft_mixture`: loads, interleaves, adapts, and tokenizes SFT rows.
- `data/task_datasets.py`: generic supervised text datasets.
  - `TaskDatasetConfig.__post_init__`: validates dataset config.
  - `_require_datasets`: dependency check.
  - `load_supervised_text_dataset`: loads text/label tasks.

### `src/modern_llm/evaluation`

- `evaluation/__init__.py`: package marker.
- `evaluation/analysis.py`
  - `summarize_errors`: summarizes evaluation errors.
- `evaluation/attention_viz.py`
  - `visualize_attention`: creates attention visualization.
- `evaluation/metrics.py`
  - `EvaluationResult`: metric result container.
  - `compute_metrics`: computes task metrics.
  - `compute_f1`: token-level F1 helper.
- `evaluation/pipeline_eval.py`
  - `StageMetrics`: per-stage metric record.
  - `PipelineEvalResults.to_dict/save/to_csv`: save eval summaries.
  - `load_model_from_checkpoint`: load model for pipeline eval.
  - `compute_perplexity`: perplexity computation.
  - `evaluate_stage`: evaluate one checkpoint.
  - `evaluate_pipeline_stages`: evaluate pretrain/SFT/DPO stages.
- `evaluation/verifier_eval.py`
  - `evaluate_verifier`: evaluate verifier model quality.

### `src/modern_llm/hf`

- `hf/__init__.py`: package marker.
- `hf/lora_utils.py`
  - `LoraConfig.__post_init__`: validates LoRA config.
  - `prepare_lora_model`: wraps model with LoRA adapters.

### `src/modern_llm/models`

- `models/__init__.py`: exports model classes and loading helpers.
- `models/attention.py`
  - `AttentionConfig.__post_init__`: validates attention settings.
  - `MultiHeadAttention.__init__`: builds Q/K/V/out projections and RoPE state.
  - `forward`: applies attention, RoPE, QK-norm, GQA, masks, and SDPA.
  - `_shape_q/_shape_kv`: reshape projections into heads.
  - `_apply_rope/_get_rope_factors/_rotate_half`: RoPE helpers.
- `models/gated_deltanet.py`
  - `GatedDeltaNetConfig.__post_init__`: validates DeltaNet settings.
  - `ShortDepthwiseConv1d.__init__/forward`: causal depthwise conv.
  - `GatedDeltaNet.__init__`: builds projections, gates, convs, norm.
  - `forward`: recurrent gated delta-rule sequence mixing.
  - `_shape_heads`: reshapes tensors into heads.
- `models/layers.py`
  - `RMSNorm.__init__/forward`: RMS normalization layer.
  - `_swish`: swish activation helper.
  - `SwiGLU.__init__/forward`: feedforward block.
- `models/loading.py`
  - `normalize_model_config_dict`: maps legacy checkpoint config keys.
  - `resolve_quantization_config`: pulls quant config from checkpoint metadata.
  - `build_model_from_checkpoint_payload`: constructs model and loads state.
  - `load_model_from_checkpoint`: reads checkpoint then delegates builder.
- `models/moe.py`
  - `TopKRouter.__init__/forward`: routes tokens to top-k experts.
  - `MixtureOfExperts.__init__/forward`: applies expert FFNs.
- `models/transformer.py`
  - `QuantizableLinearRef`: stable reference for quantizable layers.
  - `DecoderBlock.__init__`: builds attention or Gated DeltaNet plus FFN.
  - `DecoderBlock.forward`: mixer + FFN residual block.
  - `ModernDecoderLM.__init__`: builds embeddings, blocks, final norm, head.
  - `_apply_residual_init_scale`: scales residual outputs by depth.
  - `forward`: full LM forward and loss.
  - `_build_attention_bias`: causal/padding mask builder.
  - `_init_weights`: module initialization.
  - `iter_quantizable_linear_layers`: quantization traversal API.
- `models/verifier.py`
  - `VerifierConfig.__post_init__`: validates verifier architecture.
  - `VerifierModel.__init__`: builds verifier transformer/classifier.
  - `_init_weights`: initializes verifier weights.
  - `forward`: returns correctness logits/loss.
  - `score`: returns scalar correctness scores.
  - `predict`: returns binary predictions.

### `src/modern_llm/quantization`

- `quantization/__init__.py`: exports quantization public API.
- `quantization/bitlinear.py`
  - `_LowBitLinearBase.__init__`: copies dense weight/bias shadow state.
  - `set_quantization_step`: sets current step.
  - `quantization_active`: checks warmup threshold.
  - `_ste_dequantized`: straight-through estimator helper.
  - `export_quantized_state`: abstract export hook.
  - `BitLinear`: BitNet ternary linear.
  - `BitLinear._weight_scale/_quantized_weight/forward/export_quantized_state`.
  - `Int8DynActInt4WeightQATLinear`: QAT fallback wrapper.
  - `_fake_quantize_activations/_weight_scale/_quantized_weight/forward/export`.
- `quantization/config.py`
  - `QuantizationConfig.__post_init__`: validates mode/targets/scales.
  - `enabled`: true when mode is not `none`.
  - `wants_target`: checks module target group.
  - `to_dict/from_dict`: serialization helpers.
- `quantization/prepare.py`
  - `QuantizationPreparationSummary.to_dict`: summary serialization.
  - `_replace_linear`: creates BitLinear or QAT wrapper.
  - `prepare_model_for_quantization`: replaces eligible linear modules.
  - `set_quantization_step`: updates all quantized modules.
  - `get_quantization_config/get_quantization_summary/get_quantization_payload`:
    metadata helpers.
  - `export_quantized_artifact`: writes optional low-bit artifact.

### `src/modern_llm/training`

- `training/__init__.py`: package marker.
- `training/distributed.py`
  - `world_size/rank/local_rank/is_distributed/is_main_process`: DDP state.
  - `_set_default_nccl_env`: safe NCCL defaults.
  - `init_distributed`: initializes process group and CUDA device.
  - `_dist_barrier/barrier`: safe barriers with device ids.
  - `cleanup_distributed`: destroy process group.
  - `seed_everything`: rank-aware seeding.
  - `get_device`: returns rank-local device.
  - `wrap_ddp/unwrap_model`: DDP wrappers.
  - `reduce_mean`: distributed scalar mean.
  - `main_process_first`: rank ordering context manager.
  - `scale_grad_accum_for_world_size`: global-batch semantics.
  - `maybe_distributed_sampler`: creates `DistributedSampler` when needed.
  - `split_iterable_by_rank`: shards iterable work by rank.
- `training/train_lm.py`
  - `_sample_next_token`: sampling helper for generation.
  - `generate_text`: scratch model generation.
  - `print_model_parameters`: logs model parameter table.
  - `run_training`: builds model/data/optimizer/scheduler/trainer.
  - `main`: standalone CLI.
- `training/train_sft.py`
  - `load_pretrained_model`: loads pretrain checkpoint.
  - `run_sft`: builds SFT dataloader and trainer.
  - `main`: standalone CLI.
- `training/train_dpo.py`
  - `DPOConfig`: DPO hyperparameter container.
  - `PreferenceDataset.__init__`: loads/tokenizes preference pairs.
  - `_chat_messages_to_text/_coerce_text/_process_item`: schema conversion.
  - `collate_preference_batch`: pads chosen/rejected tensors.
  - `compute_sequence_logprobs`: sums sequence log-probs.
  - `DPOTrainer.__init__/train/_training_step/_move_to_device/_save_checkpoint`.
  - `load_model_from_checkpoint`: loads SFT checkpoint.
  - `run_dpo`: full DPO training entry.
  - `main`: standalone CLI.
- `training/train_verifier.py`
  - `VerifierDatasetConfig`: verifier data config.
  - `VerifierDataset.__init__/_load_and_process/_extract_answer/_format_qa`:
    builds verifier examples.
  - `_generate_wrong_answer/_tokenize/__len__/__getitem__`: data helpers.
  - `collate_verifier_batch`: pads verifier batches.
  - `VerifierTrainer.__init__/train/_training_step/evaluate/_move_to_device/_save_checkpoint`.
  - `run_verifier_training`: full verifier training entry.
  - `main`: standalone CLI.
- `training/trainer_base.py`
  - `Trainer.__post_init__`: initializes DDP, device, AMP, optional compile.
  - `train`: generic causal LM training loop.
  - `_training_step`: forward/backward/optimizer step with accumulation.
  - `_forward_loss`: model forward under autocast.
  - `evaluate`: eval loop and perplexity.
  - `_move_batch_to_device`: device transfer.
  - `_set_sampler_epoch`: distributed sampler epoch hook.
  - `_save_checkpoint`: rank-0 checkpoint save with quant metadata.

### `src/modern_llm/utils`

- `utils/__init__.py`: package marker.
- `utils/checkpointing.py`
  - `_strip_orig_mod_prefix`: removes torch.compile prefix from keys.
  - `save_checkpoint`: writes checkpoint payload.
  - `load_checkpoint`: reads checkpoint payload.
- `utils/distributed_utils.py`
  - `init_distributed_mode`: older distributed init helper.
- `utils/logging_utils.py`
  - `create_logger`: standard logger setup.
- `utils/paths.py`
  - `repo_root/data_root/hf_home/hf_datasets_cache/hf_hub_cache`: path helpers.
  - `tokenized_root/tokenizers_root`: artifact path helpers.
  - `apply_env_defaults`: sets HF cache env vars.
  - `cache_dir_for_datasets`: returns datasets cache.

## Scripts File Map

### Top-level Training and Pipeline Scripts

- `scripts/run_pipeline.py`: central CLI described above.
- `scripts/launch.sh`: simple train then eval wrapper.
- `scripts/triple_launch.sh`: 1000-step three-variant reliability wrapper.
- `scripts/triple_launch_full.sh`: full three-variant wrapper.
- `scripts/pretrain.py`: standalone pretrain CLI wrapper.
- `scripts/sft.py`: standalone SFT CLI wrapper.
- `scripts/dpo.py`: standalone DPO CLI wrapper.
- `scripts/train_verifier.py`: standalone verifier CLI wrapper.

### Utility and Reporting Scripts

- `scripts/archive_old_runs.sh`: archive old experiment runs.
- `scripts/benchmark_gsm8k.py`
  - `extract_answer/load_model/load_verifier/generate_answer/generate_candidates`
    and `score_with_verifier/main`.
- `scripts/evaluate_and_compare.py`
  - `load_model/compute_perplexity/generate_text/main`.
- `scripts/evaluate_lm_checkpoints.py`
  - `_load_checkpoint/_evaluate_model/main`.
- `scripts/evaluate_pipeline.py`
  - `main`: wrapper around pipeline evaluation.
- `scripts/experiment_attention_sinks.py`
  - `_count_repetitions/_train_model_variant/main`.
- `scripts/generate_from_checkpoints.py`
  - `_load_checkpoint/main`.
- `scripts/generate_report.py`
  - `main`: report generation entry.
- `scripts/infer.py`
  - `load_model/main`: local inference CLI.
- `scripts/plot_training_loss.py`
  - `parse_log_file/smooth_curve/plot_losses/print_summary/main`.
- `scripts/setup_check.py`
  - `check_imports`: environment check.
- `scripts/verify_checkpoints.py`
  - `load_lm_checkpoint/load_verifier_checkpoint/generate_text/main`.
- `scripts/verify_datasets.py`
  - `DatasetStatus/test_dataset/main`.
- `scripts/visualize_attention.py`
  - `load_model_with_attention/extract_attention_weights/plot_attention_heatmap`
    and `compute_attention_summary/main`.

### Data Scripts

- `scripts/data/build_packed_subset.py`
  - `_shard_tokens`: counts uint32 tokens in a shard.
  - `main`: creates symlinked subset plus `index.json`.
- `scripts/data/download_dpo_mix.py`
  - `main`: downloads DPO datasets.
- `scripts/data/download_pretrain_mix.py`
  - `Source`: source descriptor.
  - `_ensure_rp_manifest`: resolves RedPajama manifests.
  - `main`: downloads pretrain sources.
- `scripts/data/download_reasoning_mix.py`
  - `main`: downloads reasoning datasets.
- `scripts/data/download_sft_mix.py`
  - `main`: downloads SFT datasets.
- `scripts/data/ensure_dataset.py`
  - `_cached_slug/_hub_slug/_already_cached/ensure_dataset/main`.
- `scripts/data/migrate_hf_cache.sh`: cache migration helper.
- `scripts/data/offload_safe_cache.sh`: cache offload helper.
- `scripts/data/tokenize_pretrain.py`
  - `Source`: pretrain source descriptor.
  - `_open_stream`: opens HF or RedPajama source stream.
  - `_ShardWriter.append/_flush_full/flush_partial/_write`: shard writer.
  - `_load_tokenizer`: tokenizer loader.
  - `main`: token quota scheduler and shard generation.
- `scripts/data/tokenize_sft.py`
  - `_normalize/_chatml_segments/_open`: SFT text helpers.
  - `_ShardWriter`: SFT shard writer.
  - `main`: SFT tokenization entry.
- `scripts/data/train_tokenizer.py`
  - `text_iter`: source text iterator.
  - `main`: tokenizer training entry.

### Evaluation Scripts

- `scripts/evaluation/_eval_common.py`
  - `_HFCausalLMAdapter`: wraps HF models to scratch-model output format.
  - `_looks_like_hf_id`: distinguishes local checkpoint from HF id.
  - `load_scratch_model`: loads scratch or HF model.
  - `score_completion`: log-prob score for completion.
  - `mc_argmax`: multiple-choice scoring.
  - `greedy_generate`: scratch-model greedy decode.
- `scripts/evaluation/eval_all.py`
  - `EvalTask`: task descriptor.
  - `build_cmd`: builds per-task subprocess.
  - `extract_metric`: reads metric from result JSON.
  - `run_task`: runs one task.
  - `main`: runs configured task set.
- `scripts/evaluation/run_eval_sweep.py`
  - `is_already_evaluated/slugify/discover_checkpoints/build_jobs`.
  - `run_job/worker_loop`: GPU worker execution.
  - `all_task_names/write_csv/main`: task naming and summary output.
- Task scripts:
  - `eval_anli.py`: `build_prompt/evaluate_anli_round/main`.
  - `eval_bbq.py`: `build_prompt/evaluate_bbq/main`.
  - `eval_commonsenseqa.py`: `build_prompt/evaluate_csqa/main`.
  - `eval_coqa.py`: `normalize_answer/f1_score/em_score/evaluate_coqa/main`.
  - `eval_glue.py`: GLUE metrics and per-task evaluators.
  - `eval_gpqa.py`: `deterministic_shuffle/build_prompt/evaluate_gpqa/main`.
  - `eval_gsm8k.py`: answer extraction, verifier reranking, generation, eval.
  - `eval_hellaswag.py`: `preprocess/evaluate_hellaswag/main`.
  - `eval_hle.py`: HLE prompt/evaluation helpers.
  - `eval_ifbench_test.py`: `_pull_field/evaluate_ifbench/main`.
  - `eval_ifeval.py`: instruction-following rule checkers and evaluator.
  - `eval_mixeval_easy.py`: normalization/F1/data loading/evaluation.
  - `eval_mmlu.py`: `format_example/build_prompt/evaluate_mmlu/main`.
  - `eval_mmlu_pro.py`: pro MMLU prompt/evaluation.
  - `eval_squad_v2.py`: SQuAD metrics and evaluator.
  - `eval_sst2.py`: sentiment prediction and evaluator.
  - `evaluate_tasks.py`: stage checkpoint discovery and comparison tables.

## Tests

- `tests/test_dpo_trainer_metrics.py`: verifies DPO logging/step metrics.
- `tests/test_gated_deltanet_model.py`: verifies dense default and hybrid
  Gated DeltaNet forward/backward.
- `tests/test_generation.py`: tests `generate_text`.
- `tests/test_global_batch_config.py`: tests world-size-aware accumulation.
- `tests/test_model_config.py`: tests model config validation.
- `tests/test_preference_datasets.py`: tests preference split and chat parsing.
- `tests/test_pretrain_mix_weights.py`: tests pretrain mix weights.
- `tests/test_run_pipeline_sft.py`: tests SFT cap inference.
- `tests/test_sft_mixture.py`: tests SFT adapters, weights, caps, and mixture.

## Non-Code Documentation and Artifacts

- `README.md`: main project overview.
- `configs/README.md`: config overview.
- `scripts/README.md`: script overview.
- `docs/TACC_MIGRATION.md`: TACC migration guidance.
- `notebooks/README.md`: notebook notes.
- `notebooks/visualize_lm_results.py`: notebook-style visualization script.
- `report/README.md`, `report/ACL_REPORT.md`, `report/final_tacc_report.md`,
  `report/gsm8k_benchmark_report.md`: reporting artifacts.
- `report/figures/*`: generated plots and attention visualizations.
- `tokenizers/cl_small_bpe_16k/*`: local 16k tokenizer files.
- `pyproject.toml`, `requirements.txt`, `uv.lock`, `pytest.ini`: environment
  and test configuration.

## Practical Debugging Map

If training hangs before data loading:

```text
launch script -> run_pipeline._maybe_self_spawn_under_torchrun
-> distributed.init_distributed
-> main_process_first barriers
```

Check NCCL env vars and whether each rank has the correct GPU.

If SFT loads zero examples:

```text
PipelineConfig.sft_datasets
-> run_pipeline.run_sft
-> build_sft_mixture
-> get_adapter
-> SFTMixtureDataset
```

Check dataset config names and schema adapters.

If DPO says unknown split:

```text
PreferenceDatasetConfig
-> _resolve_preference_load_args
-> load_preference_dataset
```

Add split mapping in `preference_datasets.py`.

If quantized checkpoint loading fails:

```text
checkpoint payload
-> models.loading.resolve_quantization_config
-> prepare_model_for_quantization
-> load_state_dict
```

The quantization wrapper must be applied before loading weights.

If Gated DeltaNet OOMs:

```text
ModernLLMConfig.sequence_mixer
-> DecoderBlock.gated_deltanet
-> GatedDeltaNet.forward recurrent loop
```

Reduce `max_seq_len`, number of Gated DeltaNet layers, or micro-batch size.
The current reference implementation is not an optimized long-context kernel.

## Current Main Run Commands

1000-step reliability run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/triple_launch.sh
```

Resume only Gated DeltaNet:

```bash
RUN_ID=triple_multigpu_1000step_20260426_124519 \
VARIANTS=gated \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/triple_launch.sh
```

Full run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/triple_launch_full.sh
```

Dry-run preview:

```bash
DRY_RUN=true CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/triple_launch.sh
```
