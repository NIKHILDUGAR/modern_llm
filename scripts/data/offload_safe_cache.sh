#!/usr/bin/env bash
# Move HF cache entries that the RUNNING eval sweep does NOT use from the
# overlay FS (~/.cache/huggingface) onto /dev/sda (data/raw/). Reverses the
# current symlink direction (dest → src) so that after this runs:
#   data/raw/hf_cache/<name>       is a real directory
#   ~/.cache/huggingface/datasets/<name>  is a symlink → data/raw/...
# Same pattern for the hub cache.
#
# Safety model
# ------------
#  * Whitelists (DATASETS_SAFE, HUB_DATASETS_SAFE, HUB_MODELS_SAFE) explicitly
#    list entries that are NOT referenced by any `scripts/evaluation/eval_*.py`.
#    Every unsafe/eval-touched entry is left in place.
#  * For each whitelisted entry we additionally `lsof +D` the source dir — if
#    any process holds an open fd inside it, we skip that entry.
#  * Move sequence per entry: (1) rm the dest symlink → (2) rsync -a src/ dst/
#    → (3) byte-count verify → (4) rm -rf src → (5) ln -s dst src. If (3)
#    fails we abort and leave the src intact.
#
# Flags
# -----
#   --dry-run   Print what would happen; don't touch anything.
#   --log FILE  Tee to FILE (default /tmp/offload_safe_cache.log).

set -euo pipefail

DRY=0
LOG=/tmp/offload_safe_cache.log
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --log) LOG="$2"; shift 2 ;;
        *) echo "usage: $0 [--dry-run] [--log FILE]"; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_HF_HOME="${HOME}/.cache/huggingface"
SRC_DATASETS="${SRC_HF_HOME}/datasets"
SRC_HUB="${SRC_HF_HOME}/hub"
DST_DATASETS="${REPO_ROOT}/data/raw/hf_cache"
DST_HUB="${REPO_ROOT}/data/raw/hf_home/hub"

mkdir -p "$DST_DATASETS" "$DST_HUB"

# ------------------------------------------------------------------
# Whitelists. Everything that scripts/evaluation/eval_*.py can reach is
# explicitly OMITTED here (cais/mmlu, Rowan/hellaswag, glue, nyu-mll/glue,
# stanfordnlp/coqa, rajpurkar/squad_v2, facebook/anli, TIGER-Lab/MMLU-Pro,
# Idavidrein/gpqa, tau/commonsense_qa, cais/hle, MixEval/MixEval, gsm8k,
# google/IFEval, allenai/IFBench_test, Elfsong/BBQ, heegyu/bbq, gpt2,
# Xenova/text-embedding-ada-002).
# ------------------------------------------------------------------

DATASETS_SAFE=(
    "Anthropic___hh-rlhf"
    "EleutherAI___fineweb-edu-dedup-10b"
    "Skylion007___openwebtext"
    "aps___super_glue"
    "lighteval___bbq_helm"
    "lighteval___mmlu"
    "lighteval___simple_qa"
    "lighteval___summarization"
    "roneneldan___tiny_stories"
    "tatsu-lab___alpaca"
    "tinyBenchmarks___tiny_ai2_arc"
    "tinyBenchmarks___tiny_gsm8k"
    "tinyBenchmarks___tiny_hellaswag"
    "tinyBenchmarks___tiny_mmlu"
    "tinyBenchmarks___tiny_truthful_qa"
    "tinyBenchmarks___tiny_winogrande"
    "wikimedia___wikipedia"
    "wikitext"
)

HUB_DATASETS_SAFE=(
    "datasets--Anthropic--hh-rlhf"
    "datasets--EleutherAI--fineweb-edu-dedup-10b"
    "datasets--Skylion007--openwebtext"
    "datasets--aps--super_glue"
    "datasets--facebook--babi_qa"
    "datasets--lighteval--SimpleQA"
    "datasets--lighteval--bbq_helm"
    "datasets--lighteval--logiqa_harness"
    "datasets--lighteval--mmlu"
    "datasets--lighteval--summarization"
    "datasets--roneneldan--TinyStories"
    "datasets--tatsu-lab--alpaca"
    "datasets--tinyBenchmarks--tinyAI2_arc"
    "datasets--tinyBenchmarks--tinyGSM8k"
    "datasets--tinyBenchmarks--tinyHellaswag"
    "datasets--tinyBenchmarks--tinyMMLU"
    "datasets--tinyBenchmarks--tinyTruthfulQA"
    "datasets--tinyBenchmarks--tinyWinogrande"
    "datasets--wikimedia--wikipedia"
    "datasets--wikitext"
)

HUB_MODELS_SAFE=(
    "models--distilgpt2"
)

# ------------------------------------------------------------------

log() { printf '%s\n' "$*" | tee -a "$LOG"; }

check_fds_clean() {
    # Return 0 iff no process holds a file descriptor inside $1.
    local dir="$1"
    if ! command -v lsof >/dev/null 2>&1; then
        # Fallback: /proc/*/fd scan.
        local abs
        abs="$(readlink -f "$dir")"
        shopt -s nullglob
        for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
            [ -d "/proc/$pid/fd" ] || continue
            for link in /proc/"$pid"/fd/*; do
                [ -L "$link" ] || continue
                case "$(readlink "$link" 2>/dev/null)" in
                    "$abs"/*|"$abs") shopt -u nullglob; return 1 ;;
                esac
            done
        done
        shopt -u nullglob
        return 0
    fi
    ! lsof +D "$dir" >/dev/null 2>&1
}

# Human-readable size of a path.
hsize() {
    du -sh "$1" 2>/dev/null | cut -f1
}

# Sum of file byte-sizes under $1 (directories excluded). We use file-only bytes
# for verification because `du -sb` includes directory inode sizes which differ
# across filesystems (overlayfs merged dirs can report different block counts
# than a plain ext4 copy of the same content).
bsize() {
    find "$1" -type f -printf '%s\n' 2>/dev/null | awk '{s+=$1}END{print s+0}'
}
# File count — secondary sanity check alongside bsize.
fcount() {
    find "$1" -type f 2>/dev/null | wc -l
}

migrate_one() {
    local src_root="$1" dst_root="$2" name="$3"
    local src="$src_root/$name"
    local dst="$dst_root/$name"

    if [ ! -e "$src" ] && [ ! -L "$src" ]; then
        log "[skip] missing source: $src"
        return 0
    fi
    # If src is already a symlink into dst_root, we've already migrated this one.
    if [ -L "$src" ]; then
        local tgt
        tgt="$(readlink -f "$src" 2>/dev/null || true)"
        if [ -n "$tgt" ] && [[ "$tgt" == "$(readlink -f "$dst_root")"/* ]]; then
            log "[skip already-migrated] $src -> $tgt"
            return 0
        fi
        log "[skip stale symlink (not ours)]: $src -> $tgt"
        return 0
    fi
    if [ ! -d "$src" ]; then
        log "[skip non-dir source]: $src"
        return 0
    fi

    # Safety: nothing may have an open fd in src.
    if ! check_fds_clean "$src"; then
        log "[skip open fds in $src]"
        return 0
    fi

    local sz
    sz="$(hsize "$src")"
    log "[move] $name  ($sz)  $src -> $dst"

    if [ "$DRY" = 1 ]; then
        return 0
    fi

    # (1) Clean dst: if a symlink (points back at src) OR a leftover dir from a
    #     previous aborted run, remove it so cp writes a fresh tree.
    if [ -L "$dst" ]; then
        rm "$dst"
    elif [ -d "$dst" ]; then
        log "[clean stale dst dir]: $dst"
        rm -rf "$dst"
    elif [ -e "$dst" ]; then
        log "[ABORT] dst exists and is not a dir or symlink: $dst"
        return 1
    fi
    mkdir -p "$dst_root"

    # (2) Copy. cp -a preserves mode/owner/timestamps and keeps symlinks as
    # symlinks. We intentionally copy first (no mv) so that if the copy fails
    # we still have the source intact.
    if ! cp -a "$src" "$dst"; then
        log "[ABORT] cp failed for $name"
        return 1
    fi

    # (3) Verify: file-byte sum AND file count must match.
    local sb db sc dc
    sb="$(bsize "$src")"; db="$(bsize "$dst")"
    sc="$(fcount "$src")"; dc="$(fcount "$dst")"
    if [ "$sb" != "$db" ] || [ "$sc" != "$dc" ]; then
        log "[ABORT] verify mismatch for $name: src_bytes=$sb dst_bytes=$db src_files=$sc dst_files=$dc"
        return 1
    fi

    # (4) Drop the source.
    rm -rf "$src"

    # (5) Put a symlink at the old source path so any legacy code that resolves
    #     via ~/.cache/huggingface/... still works.
    ln -s "$(readlink -f "$dst")" "$src"

    log "[ok] $name"
}

log "=========="
log "offload_safe_cache.sh  dry_run=$DRY  $(date -Is)"
log "SRC_DATASETS=$SRC_DATASETS"
log "DST_DATASETS=$DST_DATASETS"
log "SRC_HUB=$SRC_HUB"
log "DST_HUB=$DST_HUB"
log "=========="

for name in "${DATASETS_SAFE[@]}"; do
    migrate_one "$SRC_DATASETS" "$DST_DATASETS" "$name" || exit 1
done

for name in "${HUB_DATASETS_SAFE[@]}"; do
    migrate_one "$SRC_HUB" "$DST_HUB" "$name" || exit 1
done

for name in "${HUB_MODELS_SAFE[@]}"; do
    migrate_one "$SRC_HUB" "$DST_HUB" "$name" || exit 1
done

log "=========="
log "done  $(date -Is)"
if [ "$DRY" = 0 ]; then
    log "overlay free after:  $(df -h / | awk 'NR==2{print $4}')"
    log "sda free after:      $(df -h "$REPO_ROOT" | awk 'NR==2{print $4}')"
fi
