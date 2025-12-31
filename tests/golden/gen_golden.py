from __future__ import annotations

import json
from pathlib import Path

from iwc.analyze.read_jsonl import iter_requests_jsonl
from iwc.analyze.summary import build_summary
from iwc.diff.core import diff_summaries, diff_to_dict
def _norm(x):
    # JSON cannot represent NaN reliably; also NaN != NaN in tests
    if isinstance(x, float) and x != x:
        return None
    return x


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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



def main() -> None:
    root = Path(__file__).resolve().parents[2]  # repo root
    examples = root / "examples"
    outdir = root / "tests" / "golden"

    tok = "tiktoken"
    model = "gpt-4o-mini"

    # ---- analyze goldens ----
    for name in ["bursty_10req.jsonl", "session_chat_5turns_cumulative.jsonl"]:
        p = examples / name
        reqs = list(iter_requests_jsonl(str(p)))
        s = build_summary(reqs, tokenizer_prefer=tok, tokenizer_model=model)
        _write_json(outdir / f"analyze_{name}.golden.json", _summary_to_dict(s))
        print(f"Wrote analyze golden: analyze_{name}.golden.json")

    # ---- diff golden ----
    a = examples / "session_chat_5turns.jsonl"
    b = examples / "session_chat_5turns_cumulative.jsonl"
    a_sum = build_summary(list(iter_requests_jsonl(str(a))), tokenizer_prefer=tok, tokenizer_model=model)
    b_sum = build_summary(list(iter_requests_jsonl(str(b))), tokenizer_prefer=tok, tokenizer_model=model)

    d = diff_summaries(a_sum, b_sum)
    _write_json(outdir / "diff_session_vs_cumulative.golden.json", diff_to_dict(d, a_label=str(a), b_label=str(b)))
    print("Wrote diff golden: diff_session_vs_cumulative.golden.json")


if __name__ == "__main__":
    main()
