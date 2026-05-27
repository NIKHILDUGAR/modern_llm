#!/usr/bin/env bash
# Sequential full training run for general, LFM2, MatFormer, Gated
# DeltaNet, and BitNet variants, then a full eval sweep including baselines.

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
#VARIANTS="${VARIANTS:-regular lfm2 matformer gated bitnet}"
VARIANTS="${VARIANTS:-lfm2 matformer gated bitnet}"

# Full profile: use the complete packed shard directory and each config's
# native step counts. Override these env vars only when intentionally resuming
# or narrowing a run.
RUN_ID="${RUN_ID:-triple_full_$(date +%Y%m%d_%H%M%S)}"
FULL_PACKED_SHARDS="${FULL_PACKED_SHARDS:-data/tokenized/pretrain_multithread_300b_final}"

EVAL_RUNS_DIR="${EVAL_RUNS_DIR:-experiments/runs/${RUN_ID}}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-experiments/results/${RUN_ID}_sweep}"
EVAL_GPUS="${EVAL_GPUS:-0 1 2 3}"
EVAL_FAST="${EVAL_FAST:-false}"
EVAL_NO_BASELINES="${EVAL_NO_BASELINES:-false}"
EVAL_TASKS="${EVAL_TASKS:-}"
EVAL_SKIP="${EVAL_SKIP:-}"
PIPELINE_FORCE="${PIPELINE_FORCE:-false}"
DRY_RUN="${DRY_RUN:-false}"

PRETRAIN_PACKED_SHARDS="${PRETRAIN_PACKED_SHARDS:-$FULL_PACKED_SHARDS}"
MAX_STEPS="${MAX_STEPS:-}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-}"
SFT_STEPS="${SFT_STEPS:-}"
DPO_STEPS="${DPO_STEPS:-}"
VERIFIER_STEPS="${VERIFIER_STEPS:-}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-}"
PRETRAIN_EVAL_WINDOWS="${PRETRAIN_EVAL_WINDOWS:-}"
SFT_NUM_EXAMPLES_PER_DATASET="${SFT_NUM_EXAMPLES_PER_DATASET:-}"
DPO_NUM_EXAMPLES="${DPO_NUM_EXAMPLES:-}"

REGULAR_OUTPUT_DIR="${REGULAR_OUTPUT_DIR:-experiments/runs/${RUN_ID}/lm-75m-2x4090_regular}"
LFM2_OUTPUT_DIR="${LFM2_OUTPUT_DIR:-experiments/runs/${RUN_ID}/lm-75m-2x4090_lfm2}"
MATFORMER_OUTPUT_DIR="${MATFORMER_OUTPUT_DIR:-experiments/runs/${RUN_ID}/lm-75m-2x4090_matformer}"
GATED_OUTPUT_DIR="${GATED_OUTPUT_DIR:-experiments/runs/${RUN_ID}/lm-75m-2x4090_gated_deltanet}"
BITNET_OUTPUT_DIR="${BITNET_OUTPUT_DIR:-experiments/runs/${RUN_ID}/lm-75m-2x4090_bitnet}"

REGULAR_CHECKPOINT="${REGULAR_CHECKPOINT:-}"
LFM2_CHECKPOINT="${LFM2_CHECKPOINT:-}"
MATFORMER_CHECKPOINT="${MATFORMER_CHECKPOINT:-}"
GATED_CHECKPOINT="${GATED_CHECKPOINT:-}"
BITNET_CHECKPOINT="${BITNET_CHECKPOINT:-}"

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

if [[ ! -f "${PRETRAIN_PACKED_SHARDS}/index.json" && "$DRY_RUN" == "true" ]]; then
    echo "[dry-run] packed pretrain index is not present yet: ${PRETRAIN_PACKED_SHARDS}/index.json"
elif [[ ! -f "${PRETRAIN_PACKED_SHARDS}/index.json" ]]; then
    echo "ERROR: packed pretrain index not found: ${PRETRAIN_PACKED_SHARDS}/index.json" >&2
    echo "Wait for tokenization to finish or set PRETRAIN_PACKED_SHARDS to a complete packed dataset." >&2
    exit 1
fi

echo
echo "============================================================"
echo "[triple-full] run_id=${RUN_ID}"
echo "[triple-full] stage=${STAGE} nproc_per_node=${NPROC_PER_NODE} cuda=${CUDA_VISIBLE_DEVICES}"
echo "[triple-full] variants=${VARIANTS}"
echo "[triple-full] packed_shards=${PRETRAIN_PACKED_SHARDS}"
echo "[triple-full] steps=config defaults unless *_STEPS/MAX_STEPS env vars are set"
echo "[triple-full] eval_fast=${EVAL_FAST} eval_no_baselines=${EVAL_NO_BASELINES}"
echo "[triple-full] dry_run=${DRY_RUN}"
echo "============================================================"

variant_enabled() {
    local key="$1"
    local item
    for item in $VARIANTS; do
        case "$item:$key" in
            all:*) return 0 ;;
            regular:regular|dense:regular) return 0 ;;
            lfm2:lfm2|hybrid-lfm2:lfm2|hybrid_lfm2:lfm2) return 0 ;;
            matformer:matformer|mat-former:matformer) return 0 ;;
            gated:gated|deltanet:gated|gated-deltanet:gated) return 0 ;;
            bitnet:bitnet|quant:bitnet|quantized:bitnet) return 0 ;;
        esac
    done
    return 1
}

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

if variant_enabled regular; then
    run_variant "regular" \
        "configs/lm_75m_2x4090.json" \
        "$REGULAR_OUTPUT_DIR" \
        "$REGULAR_CHECKPOINT"
else
    echo
    echo "[regular] skipped by VARIANTS=${VARIANTS}"
fi

if [[ "$STAGE" == "sft" && -z "$LFM2_CHECKPOINT" ]]; then
    echo
    echo "[lfm2] No LFM2_CHECKPOINT set. run_pipeline.py will look for a matching"
    echo "[lfm2] pretrain checkpoint inside ${LFM2_OUTPUT_DIR}."
fi

if variant_enabled lfm2; then
    run_variant "lfm2" \
        "configs/lm_75m_2x4090_lfm2.json" \
        "$LFM2_OUTPUT_DIR" \
        "$LFM2_CHECKPOINT"
else
    echo
    echo "[lfm2] skipped by VARIANTS=${VARIANTS}"
fi

if [[ "$STAGE" == "sft" && -z "$MATFORMER_CHECKPOINT" ]]; then
    echo
    echo "[matformer] No MATFORMER_CHECKPOINT set. run_pipeline.py will look for a matching"
    echo "[matformer] pretrain checkpoint inside ${MATFORMER_OUTPUT_DIR}."
fi

if variant_enabled matformer; then
    run_variant "matformer" \
        "configs/lm_75m_2x4090_matformer.json" \
        "$MATFORMER_OUTPUT_DIR" \
        "$MATFORMER_CHECKPOINT"
else
    echo
    echo "[matformer] skipped by VARIANTS=${VARIANTS}"
fi

if [[ "$STAGE" == "sft" && -z "$GATED_CHECKPOINT" ]]; then
    echo
    echo "[gated-deltanet] No GATED_CHECKPOINT set. run_pipeline.py will look for a matching"
    echo "[gated-deltanet] pretrain checkpoint inside ${GATED_OUTPUT_DIR}."
fi

if variant_enabled gated; then
    run_variant "gated-deltanet" \
        "configs/lm_75m_2x4090_gated_deltanet.json" \
        "$GATED_OUTPUT_DIR" \
        "$GATED_CHECKPOINT"
else
    echo
    echo "[gated-deltanet] skipped by VARIANTS=${VARIANTS}"
fi

if [[ "$STAGE" == "sft" && -z "$BITNET_CHECKPOINT" ]]; then
    regular_pretrain="${REGULAR_OUTPUT_DIR}/lm-75m-2x4090-pretrain/lm-75m-2x4090-pretrain_final.pt"
    if [[ -f "$regular_pretrain" ]]; then
        BITNET_CHECKPOINT="$regular_pretrain"
    else
        echo
        echo "[bitnet] No BITNET_CHECKPOINT set and regular pretrain checkpoint was not found."
        echo "[bitnet] run_pipeline.py will look for a matching pretrain checkpoint inside ${BITNET_OUTPUT_DIR}."
    fi
fi

if variant_enabled bitnet; then
    run_variant "bitnet" \
        "configs/lm_75m_2x4090_bitnet.json" \
        "$BITNET_OUTPUT_DIR" \
        "$BITNET_CHECKPOINT"
else
    echo
    echo "[bitnet] skipped by VARIANTS=${VARIANTS}"
fi

echo
echo "============================================================"
echo "[eval] running full checkpoint sweep with baselines"
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
