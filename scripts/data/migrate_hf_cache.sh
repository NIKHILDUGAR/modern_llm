#!/usr/bin/env bash
# Idempotent exposure of the HuggingFace caches under the repo's data/ tree via
# symlinks — no bytes are moved.
#
# What it does
# ------------
#  1. Creates the canonical cache dirs inside the repo:
#       data/raw/hf_cache    (HF_DATASETS_CACHE)
#       data/raw/hf_home     (HF_HOME)
#       data/raw/hf_home/hub (HF_HUB_CACHE)
#  2. For every per-dataset/per-model dir under ~/.cache/huggingface/{datasets,hub},
#     creates a symlink at the corresponding location under data/raw/ pointing
#     back at the original. The source files stay put in ~/.cache/huggingface.
#  3. Skips entries that are already correctly linked.
#  4. If the destination already exists as a real dir (from a previous move-style
#     migration), leaves it alone — doesn't clobber local data.
#
# Safe to re-run.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST_BASE="${REPO_ROOT}/data/raw"
DEST_DATASETS="${DEST_BASE}/hf_cache"
DEST_HUB="${DEST_BASE}/hf_home/hub"
DEST_HOME="${DEST_BASE}/hf_home"

SRC_HF_HOME="${HOME}/.cache/huggingface"
SRC_DATASETS="${SRC_HF_HOME}/datasets"
SRC_HUB="${SRC_HF_HOME}/hub"

mkdir -p "$DEST_DATASETS" "$DEST_HUB" "$DEST_HOME"

link_one_subdir() {
    local src="$1"
    local dest="$2"

    if [ ! -d "$src" ]; then
        echo "[migrate] source not present, skipping: $src"
        return
    fi

    shopt -s nullglob
    for entry in "$src"/*; do
        local name
        name="$(basename "$entry")"
        local target="$dest/$name"
        local src_abs
        src_abs="$(readlink -f "$entry")"

        if [ -L "$target" ]; then
            if [ "$(readlink -f "$target")" = "$src_abs" ]; then
                echo "[migrate] already linked: $target -> $src_abs"
                continue
            fi
            echo "[migrate] replacing stale link: $target"
            rm "$target"
        elif [ -e "$target" ]; then
            echo "[migrate] dest is a real path, leaving alone: $target"
            continue
        fi

        echo "[migrate] linking: $target -> $src_abs"
        ln -s "$src_abs" "$target"
    done
    shopt -u nullglob
}

echo "[migrate] datasets cache: $SRC_DATASETS  ->  $DEST_DATASETS"
link_one_subdir "$SRC_DATASETS" "$DEST_DATASETS"

echo "[migrate] hub cache:      $SRC_HUB  ->  $DEST_HUB"
link_one_subdir "$SRC_HUB" "$DEST_HUB"

cat <<EOF

[migrate] done (symlink-only, no bytes moved).
  HF_DATASETS_CACHE -> $DEST_DATASETS
  HF_HUB_CACHE      -> $DEST_HUB
  HF_HOME           -> $DEST_HOME
EOF
