#!/usr/bin/env bash
# Sequentially train regular, BitNet-quantized, and Gated DeltaNet variants,
# then run one eval sweep across the resulting checkpoints.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Match launch.sh defaults so all three runs use the same cache/NCCL setup.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/data/raw/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/data/raw/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${REPO_ROOT}/data/raw/hf_home/hub}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_DISTRIBUTED_TIMEOUT_MINUTES="${TORCH_DISTRIBUTED_TIMEOUT_MINUTES:-60}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "$REPO_ROOT"

STAGE="${STAGE:-all}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

# Default profile: a bounded multi-GPU reliability run. It uses the first
# 10 completed 500M-token shards from the 40B mix and runs every training
# stage for 1000 optimizer steps unless an env var below overrides it.
RUN_ID="${RUN_ID:-triple_multigpu_1000step_$(date +%Y%m%d_%H%M%S)}"
FULL_PACKED_SHARDS="${FULL_PACKED_SHARDS:-data/tokenized/pretrain_mix_40b_balanced}"
TEST_PACKED_SHARDS="${TEST_PACKED_SHARDS:-data/tokenized/pretrain_mix_40b_balanced_10shard}"
TEST_SHARD_COUNT="${TEST_SHARD_COUNT:-10}"
TRAIN_STEPS="${TRAIN_STEPS:-1000}"

EVAL_RUNS_DIR="${EVAL_RUNS_DIR:-experiments/runs/${RUN_ID}}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-experiments/results/${RUN_ID}_sweep}"
EVAL_GPUS="${EVAL_GPUS:-0 1 2 3}"
EVAL_FAST="${EVAL_FAST:-true}"
EVAL_NO_BASELINES="${EVAL_NO_BASELINES:-true}"
EVAL_TASKS="${EVAL_TASKS:-}"
EVAL_SKIP="${EVAL_SKIP:-}"
PIPELINE_FORCE="${PIPELINE_FORCE:-false}"
DRY_RUN="${DRY_RUN:-false}"

PRETRAIN_PACKED_SHARDS="${PRETRAIN_PACKED_SHARDS:-$TEST_PACKED_SHARDS}"
MAX_STEPS="${MAX_STEPS:-}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-$TRAIN_STEPS}"
SFT_STEPS="${SFT_STEPS:-$TRAIN_STEPS}"
DPO_STEPS="${DPO_STEPS:-$TRAIN_STEPS}"
VERIFIER_STEPS="${VERIFIER_STEPS:-$TRAIN_STEPS}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-}"
PRETRAIN_EVAL_WINDOWS="${PRETRAIN_EVAL_WINDOWS:-32}"
SFT_NUM_EXAMPLES_PER_DATASET="${SFT_NUM_EXAMPLES_PER_DATASET:-8000}"
DPO_NUM_EXAMPLES="${DPO_NUM_EXAMPLES:-20000}"

REGULAR_OUTPUT_DIR="${REGULAR_OUTPUT_DIR:-experiments/runs/${RUN_ID}/lm-75m-2x4090_regular}"
QUANT_OUTPUT_DIR="${QUANT_OUTPUT_DIR:-experiments/runs/${RUN_ID}/lm-75m-2x4090_quantized}"
GATED_OUTPUT_DIR="${GATED_OUTPUT_DIR:-experiments/runs/${RUN_ID}/lm-75m-2x4090_gated_deltanet}"

REGULAR_CHECKPOINT="${REGULAR_CHECKPOINT:-}"
QUANT_CHECKPOINT="${QUANT_CHECKPOINT:-}"
GATED_CHECKPOINT="${GATED_CHECKPOINT:-}"

NUMACTL_PREFIX=()
if command -v numactl >/dev/null 2>&1; then
    NUMACTL_PREFIX=(numactl --cpunodebind=1)
fi

maybe_run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [[ "$DRY_RUN" != "true" ]]; then
        "$@"
    fi
}

if [[ "$PRETRAIN_PACKED_SHARDS" == "$TEST_PACKED_SHARDS" && ! -f "${PRETRAIN_PACKED_SHARDS}/index.json" ]]; then
    echo "[setup] building ${TEST_SHARD_COUNT}-shard packed subset at ${PRETRAIN_PACKED_SHARDS}"
    maybe_run python3 scripts/data/build_packed_subset.py \
        --source-dir "$FULL_PACKED_SHARDS" \
        --output-dir "$PRETRAIN_PACKED_SHARDS" \
        --max-shards "$TEST_SHARD_COUNT"
fi

if [[ ! -f "${PRETRAIN_PACKED_SHARDS}/index.json" && "$DRY_RUN" == "true" ]]; then
    echo "[dry-run] packed pretrain index is not present yet: ${PRETRAIN_PACKED_SHARDS}/index.json"
elif [[ ! -f "${PRETRAIN_PACKED_SHARDS}/index.json" ]]; then
    echo "ERROR: packed pretrain index not found: ${PRETRAIN_PACKED_SHARDS}/index.json" >&2
    exit 1
fi

echo
echo "============================================================"
echo "[triple] run_id=${RUN_ID}"
echo "[triple] stage=${STAGE} nproc_per_node=${NPROC_PER_NODE} cuda=${CUDA_VISIBLE_DEVICES}"
echo "[triple] packed_shards=${PRETRAIN_PACKED_SHARDS}"
echo "[triple] steps pretrain=${PRETRAIN_STEPS} sft=${SFT_STEPS} dpo=${DPO_STEPS} verifier=${VERIFIER_STEPS}"
echo "[triple] dry_run=${DRY_RUN}"
echo "============================================================"

run_variant() {
    local label="$1"
    local config="$2"
    local output_dir="$3"
    local checkpoint="${4:-}"
    local cmd=(
        "${NUMACTL_PREFIX[@]}" python3 scripts/run_pipeline.py
        --config "$config"
        --stage "$STAGE"
        --output-dir "$output_dir"
        --nproc-per-node="$NPROC_PER_NODE"
    )

    [[ -n "$PRETRAIN_PACKED_SHARDS" ]] && cmd+=(--pretrain-packed-shards "$PRETRAIN_PACKED_SHARDS")
    [[ -n "$MAX_STEPS" ]] && cmd+=(--max-steps "$MAX_STEPS")
    [[ -n "$PRETRAIN_STEPS" ]] && cmd+=(--pretrain-steps "$PRETRAIN_STEPS")
    [[ -n "$SFT_STEPS" ]] && cmd+=(--sft-steps "$SFT_STEPS")
    [[ -n "$DPO_STEPS" ]] && cmd+=(--dpo-steps "$DPO_STEPS")
    [[ -n "$VERIFIER_STEPS" ]] && cmd+=(--verifier-steps "$VERIFIER_STEPS")
    [[ -n "$MAX_SEQ_LEN" ]] && cmd+=(--max-seq-len "$MAX_SEQ_LEN")
    [[ -n "$PRETRAIN_EVAL_WINDOWS" ]] && cmd+=(--pretrain-eval-windows "$PRETRAIN_EVAL_WINDOWS")
    [[ -n "$SFT_NUM_EXAMPLES_PER_DATASET" ]] && cmd+=(--sft-num-examples-per-dataset "$SFT_NUM_EXAMPLES_PER_DATASET")
    [[ -n "$DPO_NUM_EXAMPLES" ]] && cmd+=(--dpo-num-examples "$DPO_NUM_EXAMPLES")
    [[ "$PIPELINE_FORCE" == "true" ]] && cmd+=(--force)

    if [[ -n "$checkpoint" ]]; then
        if [[ ! -f "$checkpoint" ]]; then
            echo "ERROR: checkpoint for ${label} does not exist: ${checkpoint}" >&2
            exit 1
        fi
        cmd+=(--checkpoint "$checkpoint")
    fi

    echo
    echo "============================================================"
    echo "[$label] stage=${STAGE} config=${config}"
    echo "[$label] output_dir=${output_dir}"
    if [[ -n "$checkpoint" ]]; then
        echo "[$label] checkpoint=${checkpoint}"
    fi
    echo "============================================================"

    maybe_run "${cmd[@]}"
}

run_variant "regular" \
    "configs/lm_75m_2x4090.json" \
    "$REGULAR_OUTPUT_DIR" \
    "$REGULAR_CHECKPOINT"

if [[ "$STAGE" == "sft" && -z "$QUANT_CHECKPOINT" ]]; then
    regular_pretrain="${REGULAR_OUTPUT_DIR}/lm-75m-2x4090-pretrain/lm-75m-2x4090-pretrain_final.pt"
    if [[ -f "$regular_pretrain" ]]; then
        QUANT_CHECKPOINT="$regular_pretrain"
    fi
fi

run_variant "bitnet-quantized" \
    "configs/lm_75m_2x4090_bitnet.json" \
    "$QUANT_OUTPUT_DIR" \
    "$QUANT_CHECKPOINT"

if [[ "$STAGE" == "sft" && -z "$GATED_CHECKPOINT" ]]; then
    echo
    echo "[gated-deltanet] No GATED_CHECKPOINT set. run_pipeline.py will look for a matching"
    echo "[gated-deltanet] pretrain checkpoint inside ${GATED_OUTPUT_DIR}."
fi

run_variant "gated-deltanet" \
    "configs/lm_75m_2x4090_gated_deltanet.json" \
    "$GATED_OUTPUT_DIR" \
    "$GATED_CHECKPOINT"

echo
echo "============================================================"
echo "[eval] running checkpoint sweep"
echo "============================================================"

# shellcheck disable=SC2086 # EVAL_GPUS is intentionally split into argv items.
eval_cmd=(
    "${NUMACTL_PREFIX[@]}" python3 scripts/evaluation/run_eval_sweep.py
    --runs-dir "$EVAL_RUNS_DIR"
    --output-root "$EVAL_OUTPUT_ROOT"
    --gpus
)
# shellcheck disable=SC2206 # EVAL_GPUS is intentionally split into argv items.
eval_gpus=($EVAL_GPUS)
eval_cmd+=("${eval_gpus[@]}")
if [[ "$EVAL_FAST" == "true" ]]; then
    eval_cmd+=(--fast)
fi
if [[ "$EVAL_NO_BASELINES" == "true" ]]; then
    eval_cmd+=(--no-baselines)
fi
if [[ -n "$EVAL_TASKS" ]]; then
    # shellcheck disable=SC2206 # EVAL_TASKS is intentionally split into argv items.
    eval_tasks=($EVAL_TASKS)
    eval_cmd+=(--tasks "${eval_tasks[@]}")
fi
if [[ -n "$EVAL_SKIP" ]]; then
    # shellcheck disable=SC2206 # EVAL_SKIP is intentionally split into argv items.
    eval_skip=($EVAL_SKIP)
    eval_cmd+=(--skip "${eval_skip[@]}")
fi

maybe_run "${eval_cmd[@]}"
