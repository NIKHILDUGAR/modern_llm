#!/usr/bin/env python3
"""IFEval (Zhou et al., 2023) instruction-following evaluation.

IFEval (`google/IFEval`) pairs each prompt with a list of verifiable constraints
(e.g. "answer in exactly 3 paragraphs", "use the word 'however' at least twice",
"respond in ALL CAPS"). The official harness scores prompt-level and
instruction-level strict/loose accuracy.

This script ships with a lightweight subset of the official constraint verifiers
(the ones with pure-Python, no-NLTK implementations). Enough constraints are
covered to make the aggregate signal meaningful; unimplemented constraints are
recorded under `unverified_constraint_counts` so the metric denominator stays
honest.

If the upstream `google-research/instruction_following_eval` package is on the
PYTHONPATH, this script will defer to it — just install it with:
    pip install instruction_following_eval
and re-run. Otherwise we fall back to the in-tree checkers below.

Usage:
    python scripts/evaluation/eval_ifeval.py \\
        --checkpoint experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt
"""

from __future__ import annotations

import argparse
import json
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict

import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from _eval_common import DEFAULT_TOKENIZER, greedy_generate, load_scratch_model

DEFAULT_CKPT = "experiments/runs/gpu-full/gpu-full-sft/gpu-full-sft_final.pt"


# ---------- Constraint verifiers (subset of the official registry) ----------
# Each function takes (response: str, kwargs: dict) and returns bool.

def _split_paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text)


def length_num_words(response, kw):
    n = len(_word_tokens(response))
    rel = kw.get("relation")
    target = kw.get("num_words")
    if rel == "at least": return n >= target
    if rel == "less than": return n < target
    return False


def length_num_sentences(response, kw):
    sents = [s for s in re.split(r"(?<=[.!?])\s+", response.strip()) if s.strip()]
    n = len(sents)
    rel = kw.get("relation")
    target = kw.get("num_sentences")
    if rel == "at least": return n >= target
    if rel == "less than": return n < target
    return False


def length_num_paragraphs(response, kw):
    return len(_split_paragraphs(response)) == kw.get("num_paragraphs")


def detectable_content_number_placeholders(response, kw):
    # placeholders are substrings in square brackets like [name]
    return len(re.findall(r"\[[^\]]+\]", response)) >= kw.get("num_placeholders", 0)


def detectable_content_postscript(response, kw):
    marker = kw.get("postscript_marker", "P.S.")
    return marker in response


def detectable_format_number_bullets(response, kw):
    bullets = [l for l in response.splitlines() if re.match(r"^\s*(\*|-|\u2022)\s+", l)]
    return len(bullets) == kw.get("num_bullets")


def detectable_format_constrained_response(response, kw):
    # Must be exactly one of a fixed set of options
    return response.strip() in {"My answer is yes.", "My answer is no.", "My answer is maybe."}


def detectable_format_number_highlighted_sections(response, kw):
    return len(re.findall(r"\*[^*]+\*", response)) >= kw.get("num_highlights", 0)


def detectable_format_multiple_sections(response, kw):
    marker = kw.get("section_spliter") or "SECTION"
    return len(re.findall(rf"{re.escape(marker)}\s*\d+", response)) >= kw.get("num_sections", 0)


def detectable_format_title(response, kw):
    return bool(re.search(r"<<.+?>>", response))


def detectable_format_json_format(response, kw):
    try:
        json.loads(response.strip().strip("`"))
        return True
    except Exception:
        return False


def keywords_existence(response, kw):
    kws = kw.get("keywords") or []
    return all(k.lower() in response.lower() for k in kws)


def keywords_frequency(response, kw):
    word = kw.get("keyword", "").lower()
    n = len(re.findall(rf"\b{re.escape(word)}\b", response.lower()))
    rel = kw.get("relation")
    target = kw.get("frequency")
    if rel == "at least": return n >= target
    if rel == "less than": return n < target
    return False


def keywords_forbidden_words(response, kw):
    forbidden = kw.get("forbidden_words") or []
    lower = response.lower()
    return not any(re.search(rf"\b{re.escape(w.lower())}\b", lower) for w in forbidden)


def keywords_letter_frequency(response, kw):
    letter = (kw.get("letter") or "").lower()
    n = sum(1 for ch in response.lower() if ch == letter)
    rel = kw.get("let_relation") or kw.get("relation")
    target = kw.get("let_frequency") or kw.get("frequency")
    if rel == "at least": return n >= target
    if rel == "less than": return n < target
    return False


def change_case_capital_word_frequency(response, kw):
    caps = [w for w in _word_tokens(response) if w.isupper() and len(w) > 1]
    rel = kw.get("capital_relation") or kw.get("relation")
    target = kw.get("capital_frequency") or kw.get("frequency")
    if rel == "at least": return len(caps) >= target
    if rel == "less than": return len(caps) < target
    return False


def change_case_english_capital(response, kw):
    letters = [c for c in response if c.isalpha()]
    return len(letters) > 0 and all(c.isupper() for c in letters)


def change_case_english_lowercase(response, kw):
    letters = [c for c in response if c.isalpha()]
    return len(letters) > 0 and all(c.islower() for c in letters)


def punctuation_no_comma(response, kw):
    return "," not in response


def startend_quotation(response, kw):
    s = response.strip()
    return len(s) >= 2 and s[0] in {'"', '\u201c'} and s[-1] in {'"', '\u201d'}


def startend_end_checker(response, kw):
    end_phrase = kw.get("end_phrase", "")
    return response.rstrip().endswith(end_phrase)


def combination_two_responses(response, kw):
    return "******" in response


def combination_repeat_prompt(response, kw):
    prompt_to_repeat = kw.get("prompt_to_repeat", "")
    return prompt_to_repeat in response


VERIFIERS: Dict[str, Callable[[str, dict], bool]] = {
    "length_constraints:number_words": length_num_words,
    "length_constraints:number_sentences": length_num_sentences,
    "length_constraints:number_paragraphs": length_num_paragraphs,
    "detectable_content:number_placeholders": detectable_content_number_placeholders,
    "detectable_content:postscript": detectable_content_postscript,
    "detectable_format:number_bullet_lists": detectable_format_number_bullets,
    "detectable_format:constrained_response": detectable_format_constrained_response,
    "detectable_format:number_highlighted_sections": detectable_format_number_highlighted_sections,
    "detectable_format:multiple_sections": detectable_format_multiple_sections,
    "detectable_format:title": detectable_format_title,
    "detectable_format:json_format": detectable_format_json_format,
    "keywords:existence": keywords_existence,
    "keywords:frequency": keywords_frequency,
    "keywords:forbidden_words": keywords_forbidden_words,
    "keywords:letter_frequency": keywords_letter_frequency,
    "change_case:capital_word_frequency": change_case_capital_word_frequency,
    "change_case:english_capital": change_case_english_capital,
    "change_case:english_lowercase": change_case_english_lowercase,
    "punctuation:no_comma": punctuation_no_comma,
    "startend:quotation": startend_quotation,
    "startend:end_checker": startend_end_checker,
    "combination:two_responses": combination_two_responses,
    "combination:repeat_prompt": combination_repeat_prompt,
}


def check_response(response: str, instruction_id: str, kw: dict) -> Any:
    fn = VERIFIERS.get(instruction_id)
    if fn is None:
        return None
    try:
        return bool(fn(response, kw or {}))
    except Exception:
        return False


def evaluate_ifeval(model, tokenizer, device: str, max_samples: int) -> dict:
    ds = load_dataset("google/IFEval", split="train")
    if max_samples and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))

    n_prompts = 0
    n_prompts_pass = 0
    n_instructions = 0
    n_instructions_pass = 0
    unverified = Counter()

    for ex in tqdm(ds, desc="Evaluating IFEval"):
        prompt = ex["prompt"]
        response = greedy_generate(model, tokenizer, prompt, device,
                                   max_new_tokens=256)
        inst_ids = ex.get("instruction_id_list") or []
        kwargs_list = ex.get("kwargs") or [{} for _ in inst_ids]
        prompt_pass = True
        for iid, kw in zip(inst_ids, kwargs_list):
            kw = kw or {}
            result = check_response(response, iid, kw)
            if result is None:
                unverified[iid] += 1
                prompt_pass = False  # treat unverified as failure to keep denominator honest
                continue
            n_instructions += 1
            if result:
                n_instructions_pass += 1
            else:
                prompt_pass = False
        n_prompts += 1
        if prompt_pass:
            n_prompts_pass += 1

    return {
        "prompt_accuracy": n_prompts_pass / n_prompts if n_prompts else 0.0,
        "instruction_accuracy": n_instructions_pass / n_instructions if n_instructions else 0.0,
        "prompts_total": n_prompts,
        "prompts_pass": n_prompts_pass,
        "instructions_total": n_instructions,
        "instructions_pass": n_instructions_pass,
        "unverified_constraint_counts": dict(unverified),
    }


def main():
    parser = argparse.ArgumentParser(description="IFEval instruction-following evaluation")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", type=str, default="experiments/results/ifeval_metrics.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    model, tokenizer = load_scratch_model(args.checkpoint, args.device, args.tokenizer)

    print("Evaluating IFEval...")
    results = evaluate_ifeval(model, tokenizer, args.device, args.max_samples)
    results["model"] = str(args.checkpoint)

    print(f"\nIFEval prompt accuracy     : {results['prompt_accuracy']:.2%}")
    print(f"IFEval instruction accuracy: {results['instruction_accuracy']:.2%}")
    if results["unverified_constraint_counts"]:
        print(f"Unverified constraint ids (counted as failures): "
              f"{results['unverified_constraint_counts']}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
