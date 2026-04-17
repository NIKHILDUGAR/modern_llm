#!/usr/bin/env bash
# Idempotent migration of the HuggingFace caches off the overlay FS onto the
# 1.8 TB data volume.
#
# What it does
# ------------
#  1. Creates the canonical cache dirs inside the repo:
#       data/raw/hf_cache    (HF_DATASETS_CACHE)
#       data/raw/hf_home     (HF_HOME)
#       data/raw/hf_home/hub (HF_HUB_CACHE)
#  2. Moves every per-dataset/per-model dir from ~/.cache/huggingface/{datasets,hub}
#     into the corresponding location under data/raw/, and replaces the original
#     directory with a symlink so any code that still hard-codes ~/.cache/huggingface
#     keeps working.
#  3. Skips entries that are already symlinks pointing into data/raw/.
#  4. Skips entries that already exist at the destination (rsync the diff into
#     them and then delete the source) — this is what makes the script safe to
#     re-run after a partial migration.
#
# Run this BEFORE the first multi-GPU pretrain. It does not download any new
# data and is safe to interrupt.

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

migrate_one_subdir() {
    local src="$1"
    local dest="$2"

    if [ ! -d "$src" ]; then
        echo "[migrate] source not present, skipping: $src"
        return
    fi

    # Iterate top-level entries in src (each is a dataset or model).
    shopt -s nullglob
    for entry in "$src"/*; do
        local name
        name="$(basename "$entry")"
        local target="$dest/$name"

        # Already a symlink in the source pointing where we want? skip.
        if [ -L "$entry" ] && [ "$(readlink -f "$entry")" = "$(readlink -f "$target" 2>/dev/null || echo "$target")" ]; then
            echo "[migrate] already linked: $entry"
            continue
        fi

        if [ -e "$target" ] && [ ! -L "$target" ]; then
            echo "[migrate] dest exists, rsyncing diff: $entry -> $target"
            rsync -a --remove-source-files "$entry"/ "$target"/
            # Remove the now-empty source dir so we can replace it with a symlink.
            find "$entry" -type d -empty -delete || true
            if [ -e "$entry" ]; then
                rm -rf "$entry"
            fi
        else
            echo "[migrate] moving: $entry -> $target"
            mv "$entry" "$target"
        fi

        # Drop a back-symlink so legacy code paths still resolve.
        ln -s "$target" "$entry"
    done
    shopt -u nullglob
}

echo "[migrate] datasets cache: $SRC_DATASETS  ->  $DEST_DATASETS"
migrate_one_subdir "$SRC_DATASETS" "$DEST_DATASETS"

echo "[migrate] hub cache:      $SRC_HUB  ->  $DEST_HUB"
migrate_one_subdir "$SRC_HUB" "$DEST_HUB"

# Some HF tools also drop config files at ${HF_HOME} root (token, accelerate
# config etc). Do not move these silently — leave them in place under
# ~/.cache/huggingface; the new HF_HOME is empty by design.

cat <<EOF

[migrate] done.
  HF_DATASETS_CACHE -> $DEST_DATASETS
  HF_HUB_CACHE      -> $DEST_HUB
  HF_HOME           -> $DEST_HOME

Next: source scripts/launch.sh-style env vars (or just let scripts/launch.sh
do it) so all subsequent runs use the new locations.
EOF
