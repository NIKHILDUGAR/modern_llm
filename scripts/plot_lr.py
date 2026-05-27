#!/usr/bin/env python3
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def lr_mul(step: int, warm: int, total: int, min_ratio: float) -> float:
    warm = max(warm, 1)
    total = max(total, warm + 1)
    if step < warm:
        return float(step + 1) / float(warm)
    progress = (step - warm) / max(total - warm, 1)
    progress = min(max(progress * 1.3, 0.0), 1.0)
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))


def compute_lr_curve(total_steps: int, warmup_steps: int, base_lr: float, min_ratio: float):
    steps = np.arange(total_steps)
    # Vectorized computation
    warm = max(warmup_steps, 1)
    total = max(total_steps, warm + 1)
    lrs = np.empty_like(steps, dtype=float)
    # warmup
    warm_idxs = steps < warm
    lrs[warm_idxs] = (steps[warm_idxs] + 1).astype(float) / float(warm)
    # decay
    decay_idxs = ~warm_idxs
    progress = (steps[decay_idxs] - warm) / max(total - warm, 1)
    progress = np.clip(progress * 1.3, 0.0, 1.0)
    lrs[decay_idxs] = min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + np.cos(np.pi * progress))
    return steps, lrs * base_lr


if __name__ == "__main__":
    # ASSUMPTIONS: using pretrain defaults from configs/lm_75m_2x4090.json
    base_lr = 2e-3
    warmup_steps = 24000
    min_ratio = 0.01
    totals = [1013200, 4577381]

    plt.figure(figsize=(10, 4))
    for t in totals:
        steps, lrs = compute_lr_curve(t, warmup_steps, base_lr, min_ratio)
        plt.plot(steps, lrs, label=f"total={t:,}")
        # mark end of warmup and end of decay
        warm_point = warmup_steps - 1
        end_decay_step = int(math.ceil(warmup_steps + (max(t, warmup_steps + 1) - warmup_steps) / 1.3))
        plt.axvline(warm_point, color="gray", linestyle="--", alpha=0.4)
        plt.axvline(end_decay_step, color="gray", linestyle=":", alpha=0.4)
        plt.text(warm_point, base_lr * 0.9, "warmup end", rotation=90, va="center", ha="right", color="gray")
        plt.text(end_decay_step, base_lr * 0.9, "decay end", rotation=90, va="center", ha="left", color="gray")

    plt.xlabel("Training step")
    plt.ylabel("Learning rate")
    plt.title("LR schedule (warmup + cosine decay) — base_lr=2e-3, warmup=24k, min_ratio=0.01")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    outdir = "experiments/plots"
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "lr_schedule_compare.png")
    plt.savefig(outpath, dpi=150)
    print("Saved:", outpath)
