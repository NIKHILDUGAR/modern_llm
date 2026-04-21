#!/usr/bin/env python3
"""Resolve every dataset in the DPO mix (plan §3.4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from modern_llm.utils.paths import apply_env_defaults  # noqa: E402

apply_env_defaults()

from ensure_dataset import ensure_dataset  # noqa: E402

DPO_MIX = [
    ("ultrafeedback",  "HuggingFaceH4/ultrafeedback_binarized",            None,             "train_prefs", False),
    ("helpsteer2",     "nvidia/HelpSteer2",                                None,             "train",       False),
    ("skywork-reward", "Skywork/Skywork-Reward-Preference-80K-v0.2",       None,             "train",       False),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="+", default=None)
    args = p.parse_args()

    want = set(args.only) if args.only else None
    fails = []
    for key, name, cfg, split, streaming in DPO_MIX:
        if want and key not in want:
            continue
        try:
            ensure_dataset(name, cfg, split, streaming)
        except Exception as exc:
            print(f"[dpo-mix] FAIL {key}: {exc}", flush=True)
            fails.append((key, str(exc)))
    if fails:
        print(f"\n[dpo-mix] {len(fails)} failures", flush=True)
        return 1
    print("\n[dpo-mix] all ok", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
