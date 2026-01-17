from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("tiktoken")

from iwc.analyze.read_jsonl import iter_requests_jsonl
from iwc.analyze.summary import build_summary
from iwc.diff.core import diff_summaries, diff_to_dict


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def test_diff_golden_session_vs_cumulative() -> None:
    root = _repo_root()
    examples = root / "examples"
    golden = root / "tests" / "golden" / "diff_session_vs_cumulative.golden.json"

    a = examples / "session_chat_5turns.jsonl"
    b = examples / "session_chat_5turns_cumulative.jsonl"

    a_sum = build_summary(list(iter_requests_jsonl(str(a))), tokenizer_prefer="tiktoken", tokenizer_model="gpt-4o-mini")
    b_sum = build_summary(list(iter_requests_jsonl(str(b))), tokenizer_prefer="tiktoken", tokenizer_model="gpt-4o-mini")

    d = diff_summaries(a_sum, b_sum)
    got = diff_to_dict(d, a_label=str(a), b_label=str(b))
    exp = _load_json(golden)

    # Labels are environment-specific (absolute paths differ on CI vs local),
# so don't snapshot-test them.
got.pop("a_label", None)
got.pop("b_label", None)
exp.pop("a_label", None)
exp.pop("b_label", None)

assert got == exp
