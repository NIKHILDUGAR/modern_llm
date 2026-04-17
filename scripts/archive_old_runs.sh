#!/usr/bin/env bash
# Archive pre-4090 runs out of the active experiments dir.
#
# Why: we are switching tokenizer and resizing the model, so the old
# experiments/runs/gpu-full* checkpoints are no longer compatible. We do not
# delete them — we move them under experiments/runs/gpu-full_archive_pre-4090/
# so a future bisection / re-eval is still possible.
#
# Idempotent: re-runs are no-ops if the archive already exists.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_DIR="${REPO_ROOT}/experiments/runs"
ARCHIVE_DIR="${RUNS_DIR}/gpu-full_archive_pre-4090"

mkdir -p "$ARCHIVE_DIR"

shopt -s nullglob
moved=0
for entry in "$RUNS_DIR"/2026*; do
    name="$(basename "$entry")"
    # Skip the archive dir itself.
    if [ "$name" = "gpu-full_archive_pre-4090" ]; then
        continue
    fi
    target="$ARCHIVE_DIR/$name"
    if [ -e "$target" ]; then
        echo "[archive] already archived: $entry"
        continue
    fi
    echo "[archive] $entry -> $target"
    mv "$entry" "$target"
    moved=$((moved + 1))
done
shopt -u nullglob

echo "[archive] done; moved $moved entries into $ARCHIVE_DIR"
