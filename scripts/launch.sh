#!/usr/bin/env bash
# Launch wrapper for run_pipeline.py with NUMA pinning + HF cache redirects.
#
# Two supported invocation modes:
#
#   1) Single-GPU dev / smoke (matches the user's preferred shape):
#
#      CUDA_VISIBLE_DEVICES=1 ./scripts/launch.sh \
#          --config gpu --stage all
#
#   2) Multi-GPU DDP via torchrun self-spawn:
#
#      CUDA_VISIBLE_DEVICES=0,1 ./scripts/launch.sh \
#          --config gpu --stage all --nproc-per-node 2
#
# Both modes are GPU-count agnostic — set CUDA_VISIBLE_DEVICES to any subset
# of GPUs you want to use. The trainer reads WORLD_SIZE/LOCAL_RANK from env.
#
# NCCL defaults below are tuned for consumer PCIe boxes (RTX 3090/4090). Set
# them in your shell to override.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- HF cache redirects -----------------------------------------------------
# Datasets and model weights land inside the data volume (1.8 TB free) rather
# than the overlay FS (~470 GB free). The migrate_hf_cache.sh script populates
# these dirs on first use; data-loading code falls back to the live cache via
# `cache_dir=` if the env vars happen to be unset.
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


numactl --cpunodebind=1 python3 scripts/run_pipeline.py "$@"
numactl --cpunodebind=1 python3 scripts/evaluation/run_eval_sweep.py \
    --runs-dir experiments/runs \
    --output-root experiments/results/sweep \
    --gpus 0 1 2 3
