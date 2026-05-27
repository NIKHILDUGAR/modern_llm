from __future__ import annotations

import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1] / "scripts" / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import eval_gsm8k


class _Dataset(list):
    def select(self, indices):
        return _Dataset([self[i] for i in indices])


def test_evaluate_gsm8k_error_examples_keep_full_question(monkeypatch) -> None:
    question = " ".join(f"token{i}" for i in range(40))
    dataset = _Dataset([{"question": question, "answer": "#### 42"}])

    monkeypatch.setattr(eval_gsm8k, "load_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(eval_gsm8k, "generate_solutions", lambda *args, **kwargs: ["wrong 7"])

    result = eval_gsm8k.evaluate_gsm8k(
        model=object(),
        tokenizer=object(),
        device="cpu",
        max_samples=1,
    )

    saved_question = result["examples"]["errors"][0]["question"]
    assert len(question) > 100
    assert saved_question == question


def test_generate_solutions_passes_full_question_to_greedy_generate(monkeypatch) -> None:
    question = " ".join(f"word{i}" for i in range(40))
    calls = []

    def fake_greedy_generate(model, tokenizer, prompt, device, max_new_tokens):
        calls.append((prompt, device, max_new_tokens))
        return "solution"

    monkeypatch.setattr(eval_gsm8k, "greedy_generate", fake_greedy_generate)

    solutions = eval_gsm8k.generate_solutions(
        model=object(),
        tokenizer=object(),
        question=question,
        device="cpu",
        n_samples=2,
        max_length=17,
    )

    assert solutions == ["solution", "solution"]
    assert calls == [
        (f"Question: {question}\n\nLet me solve this step by step:\n", "cpu", 17),
        (f"Question: {question}\n\nLet me solve this step by step:\n", "cpu", 17),
    ]
