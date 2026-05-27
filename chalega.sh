set -euo pipefail

run_ckpt () {
  ckpt="$1"
  name="$2"
  out="experiments/results/full_5_model_2048_pleaselowlr/$name"
  mkdir -p "$out"

  CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python3 scripts/evaluation/eval_all.py \
    --checkpoint "$ckpt" --device cuda --output-dir "$out" \
    --summary "$out/eval_all_summary_gpu0.json" \
    --tasks mmlu mmlu_pro hellaswag anli &

  CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false python3 scripts/evaluation/eval_all.py \
    --checkpoint "$ckpt" --device cuda --output-dir "$out" \
    --summary "$out/eval_all_summary_gpu1.json" \
    --tasks commonsenseqa coqa glue gpqa &

  CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=false python3 scripts/evaluation/eval_all.py \
    --checkpoint "$ckpt" --device cuda --output-dir "$out" \
    --summary "$out/eval_all_summary_gpu2.json" \
    --tasks hle ifbench_test ifeval mixeval_easy &

  CUDA_VISIBLE_DEVICES=3 TOKENIZERS_PARALLELISM=false python3 scripts/evaluation/eval_all.py \
    --checkpoint "$ckpt" --device cuda --output-dir "$out" \
    --summary "$out/eval_all_summary_gpu3.json" \
    --tasks squad_v2 sst2 gsm8k &

  wait

  python3 - "$ckpt" "$out" <<'PY'
import json, sys
from pathlib import Path

ckpt = sys.argv[1]
out = Path(sys.argv[2])
order = [
    "mmlu", "mmlu_pro", "hellaswag", "anli",
    "commonsenseqa", "coqa", "glue", "gpqa",
    "hle", "ifbench_test", "ifeval", "mixeval_easy",
    "squad_v2", "sst2", "gsm8k",
]

tasks = []
elapsed = 0.0
for i in range(4):
    data = json.loads((out / f"eval_all_summary_gpu{i}.json").read_text())
    tasks.extend(data.get("tasks", []))
    elapsed += float(data.get("total_elapsed_s", 0) or 0)

by_name = {t["task"]: t for t in tasks}
merged = {
    "checkpoint": ckpt,
    "tokenizer": None,
    "device": "cuda",
    "fast": False,
    "_expected_tasks": order,
    "tasks": [by_name[name] for name in order if name in by_name],
    "total_elapsed_s": elapsed,
}
(out / "eval_all_summary.json").write_text(json.dumps(merged, indent=2))
PY
}

run_ckpt \
  experiments/runs/full_5_model_2048_pleaselowlr/lm-75m-2x4090_regular/lm-75m-2x4090-pretrain/lm-75m-2x4090-pretrain_final.pt \
  lm-75m-2x4090-pretrain_final

run_ckpt \
  experiments/runs/full_5_model_2048_pleaselowlr/lm-75m-2x4090_regular/lm-75m-2x4090-sft/lm-75m-2x4090-sft_final.pt \
  lm-75m-2x4090-sft_final

run_ckpt \
  experiments/runs/full_5_model_2048_pleaselowlr/lm-75m-2x4090_regular/lm-75m-2x4090-dpo/lm-75m-2x4090-dpo_final.pt \
  lm-75m-2x4090-dpo_final

