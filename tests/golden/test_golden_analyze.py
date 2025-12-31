from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("tiktoken")  # skip cleanly if tiktoken isn't installed

from iwc.analyze.read_jsonl import iter_requests_jsonl
from iwc.analyze.summary import build_summary
def _norm(x):
    if isinstance(x, float) and x != x:
        return None
    return x


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _summary_to_dict(s) -> dict:
    """Keep this stable: only store the metrics we care about for regressions."""
    def n(x):  # local alias
        return _norm(x)

    return {
        "requests": s.requests,
        "tokenizer_used": s.tokenizer_used,
        "prompt_tokens": {"p50": n(s.prompt_tokens.p50), "p90": n(s.prompt_tokens.p90), "p99": n(s.prompt_tokens.p99), "mean": n(s.prompt_tokens.mean)},
        "max_output_tokens": {"p90": n(s.max_output_tokens.p90), "mean": n(s.max_output_tokens.mean)},
        "prompt_output_ratio": {"mean": n(s.prompt_output_ratio.mean)},
        "prefill_dominance": {"p50": n(s.prefill_dominance.p50), "p90": n(s.prefill_dominance.p90)},
        "arrivals": {
            "duration_s": n(s.arrivals.duration_s),
            "mean_rps": n(s.arrivals.mean_rps),
            "peak_rps_1s": n(s.arrivals.peak_rps_1s),
            "burstiness_cv": n(s.arrivals.burstiness_cv),
            "pattern": s.arrivals.pattern,
            "interarrival_ms": {
                "p50": n(s.arrivals.interarrival_ms.p50),
                "p90": n(s.arrivals.interarrival_ms.p90),
                "p99": n(s.arrivals.interarrival_ms.p99),
            },
        },
        "sessions": {
            "sessions_detected": s.sessions.sessions_detected,
            "avg_turns_per_session": n(s.sessions.avg_turns_per_session),
            "turns_per_session": {"p90": n(s.sessions.turns_per_session.p90)},
            "prompt_reuse_ratio_tokens": n(s.sessions.prompt_reuse_ratio_tokens),
            "prompt_tokens_by_turn": {"p50": n(s.sessions.prompt_tokens_by_turn.p50), "p90": n(s.sessions.prompt_tokens_by_turn.p90)},
            "prompt_token_growth": {"p50": n(s.sessions.prompt_token_growth.p50), "p90": n(s.sessions.prompt_token_growth.p90)},
        },
    }



@pytest.mark.parametrize(
    "jsonl_name,golden_name",
    [
        ("bursty_10req.jsonl", "analyze_bursty_10req.jsonl.golden.json"),
        ("session_chat_5turns_cumulative.jsonl", "analyze_session_chat_5turns_cumulative.jsonl.golden.json"),
    ],
)
def test_analyze_golden(jsonl_name: str, golden_name: str) -> None:
    root = _repo_root()
    examples = root / "examples"
    golden = root / "tests" / "golden" / golden_name

    reqs = list(iter_requests_jsonl(str(examples / jsonl_name)))
    s = build_summary(reqs, tokenizer_prefer="tiktoken", tokenizer_model="gpt-4o-mini")

    got = _summary_to_dict(s)
    exp = _load_json(golden)

    assert got == exp
