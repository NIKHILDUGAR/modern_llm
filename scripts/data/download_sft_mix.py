#!/usr/bin/env python3
"""Resolve every dataset in the SFT mix (plan §3.3)."""

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

# (key, hf_name, hf_config, split, streaming)
SFT_MIX = [
    ("tulu-3",           "allenai/tulu-3-sft-mixture",      None, "train", False),
    ("smoltalk",         "HuggingFaceTB/smoltalk",          "all", "train", False),
    ("openhermes-2.5",   "teknium/OpenHermes-2.5",          None, "train", False),
    ("metamath",         "meta-math/MetaMathQA",            None, "train", False),
    ("openmathinstruct", "nvidia/OpenMathInstruct-2",       None, "train", False),
    ("ifeval-like",      "argilla/ifeval-like-data",        None, "train", False),
    ("no-robots",        "HuggingFaceH4/no_robots",         None, "train", False),
    ("coqa-train",       "stanfordnlp/coqa",                None, "train", False),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--only", nargs="+", default=None)
    args = p.parse_args()

    want = set(args.only) if args.only else None
    fails = []
    for key, name, cfg, split, streaming in SFT_MIX:
        if want and key not in want:
            continue
        try:
            ensure_dataset(name, cfg, split, streaming)
        except Exception as exc:
            print(f"[sft-mix] FAIL {key} ({name}): {exc}", flush=True)
            fails.append((key, str(exc)))
    if fails:
        print(f"\n[sft-mix] {len(fails)} failures", flush=True)
        return 1
    print("\n[sft-mix] all ok", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
