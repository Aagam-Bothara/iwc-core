# iwc/analyze/summary.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .arrivals import ArrivalStats, analyze_arrivals
from .read_jsonl import Request
from .sessions import SessionStats, analyze_sessions
from .stats import DistSummary
from .tokenizer import get_tokenizer


@dataclass(frozen=True)
class WorkloadSummary:
    requests: int

    prompt_tokens: DistSummary
    max_output_tokens: DistSummary
    prompt_output_ratio: DistSummary
    prefill_dominance: DistSummary  # prompt / (prompt + max_output_tokens)

    arrivals: ArrivalStats
    sessions: SessionStats

    tokenizer_used: str


def build_summary(
    reqs: List[Request],
    tokenizer_prefer: str = "tiktoken",
    tokenizer_model: str = "gpt-4o-mini",
) -> WorkloadSummary:
    tok = get_tokenizer(prefer=tokenizer_prefer, model=tokenizer_model)
    tokenizer_used = f"{tokenizer_prefer}:{tokenizer_model}" if tokenizer_prefer == "tiktoken" else "simple"

    prompt_lens: List[float] = []
    out_caps: List[float] = []
    ratios: List[float] = []
    prefill_scores: List[float] = []
    arrivals_ms: List[int] = []

    for r in reqs:
        p = r.prompt or ""
        pt = len(tok.encode(p))
        prompt_lens.append(float(pt))

        out_cap = int(r.max_output_tokens)
        out_caps.append(float(out_cap))

        if out_cap > 0:
            ratios.append(float(pt) / float(out_cap))

        den = float(pt + max(0, out_cap))
        if den > 0:
            prefill_scores.append(float(pt) / den)

        arrivals_ms.append(int(r.arrival_time_ms))

    prompt_sum = DistSummary.from_list(prompt_lens)
    out_sum = DistSummary.from_list(out_caps)
    ratio_sum = DistSummary.from_list(ratios)
    prefill_sum = DistSummary.from_list(prefill_scores)

    arrival_stats = analyze_arrivals(arrivals_ms)
    session_stats = analyze_sessions(reqs, tokenizer_prefer=tokenizer_prefer, tokenizer_model=tokenizer_model)

    return WorkloadSummary(
        requests=len(reqs),
        prompt_tokens=prompt_sum,
        max_output_tokens=out_sum,
        prompt_output_ratio=ratio_sum,
        prefill_dominance=prefill_sum,
        arrivals=arrival_stats,
        sessions=session_stats,
        tokenizer_used=tokenizer_used,
    )


def fmt(x: float, nd: int = 2) -> str:
    if x != x:  # NaN
        return "n/a"
    return f"{x:.{nd}f}"


def _is_nan(x: float) -> bool:
    return x != x


def render_summary(s: WorkloadSummary) -> str:
    lines: List[str] = []
    lines.append("WORKLOAD SUMMARY")
    lines.append("----------------")
    lines.append(f"Requests           : {s.requests}")
    lines.append(f"Tokenizer          : {s.tokenizer_used}")

    # ---- Workload tags (coarse) ----
    tags: List[str] = []
    if not _is_nan(s.arrivals.burstiness_cv):
        tags.append("bursty" if s.arrivals.burstiness_cv > 1.5 else "smooth")

    # Multi-turn tag: sessions + turns>=2
    turns_p90 = s.sessions.turns_per_session.p90
    is_multiturn = (
        s.sessions.sessions_detected
        and not _is_nan(turns_p90)
        and turns_p90 >= 2
    )
    if is_multiturn:
        tags.append("multi-turn")

    if not _is_nan(s.prefill_dominance.p50) and s.prefill_dominance.p50 > 0.65:
        tags.append("prefill-heavy")

    if (
        s.sessions.sessions_detected
        and not _is_nan(s.sessions.prompt_reuse_ratio_tokens)
        and s.sessions.prompt_reuse_ratio_tokens > 0.5
    ):
        tags.append("high-reuse")

    lines.append(f"WORKLOAD TYPE       : {', '.join(tags) if tags else 'unknown'}")

    # ---- Primary class (single label) ----
    primary = "unknown"
    if is_multiturn:
        if not _is_nan(s.prefill_dominance.p50) and s.prefill_dominance.p50 > 0.65:
            primary = "interactive-chat (prefill-heavy)"
        else:
            primary = "interactive-chat"
    elif not _is_nan(s.arrivals.burstiness_cv) and s.arrivals.burstiness_cv > 1.5:
        primary = "bursty-api"
    else:
        primary = "batch/offline"
    lines.append(f"PRIMARY CLASS       : {primary}")

    lines.append("")
    lines.append("TOKENS")
    lines.append("-----")
    lines.append(f"Avg prompt tokens  : {fmt(s.prompt_tokens.mean, 2)}")
    lines.append(f"P50 prompt tokens  : {fmt(s.prompt_tokens.p50, 0)}")
    lines.append(f"P90 prompt tokens  : {fmt(s.prompt_tokens.p90, 0)}")
    lines.append(f"P99 prompt tokens  : {fmt(s.prompt_tokens.p99, 0)}")
    lines.append("")
    lines.append(f"Avg max_output_tokens : {fmt(s.max_output_tokens.mean, 2)}")
    lines.append(f"P90 max_output_tokens : {fmt(s.max_output_tokens.p90, 0)}")
    lines.append("")
    lines.append(f"Avg prompt/output cap ratio : {fmt(s.prompt_output_ratio.mean, 3)}")
    lines.append(f"Prefill dominance P50/P90   : {fmt(s.prefill_dominance.p50, 3)}/{fmt(s.prefill_dominance.p90, 3)}")
    lines.append("")

    lines.append("ARRIVAL PROFILE")
    lines.append("---------------")
    lines.append(f"Duration (s)       : {fmt(s.arrivals.duration_s, 2)}")
    lines.append(f"Mean RPS           : {fmt(s.arrivals.mean_rps, 2)}")
    lines.append(f"Peak reqs (1s bin) : {fmt(s.arrivals.peak_rps_1s, 0)}")
    lines.append(
        "Inter-arrival ms P50/P90/P99 : "
        f"{fmt(s.arrivals.interarrival_ms.p50,0)}/"
        f"{fmt(s.arrivals.interarrival_ms.p90,0)}/"
        f"{fmt(s.arrivals.interarrival_ms.p99,0)}"
    )
    lines.append(f"Burstiness (CV)    : {fmt(s.arrivals.burstiness_cv, 2)}")
    lines.append(f"Pattern            : {s.arrivals.pattern}")
    lines.append("")

    lines.append("SESSION ANALYSIS")
    lines.append("---------------")
    lines.append(f"Sessions detected  : {s.sessions.sessions_detected}")
    lines.append(f"Avg turns/session  : {fmt(s.sessions.avg_turns_per_session, 2)}")
    lines.append(f"P90 turns/session  : {fmt(s.sessions.turns_per_session.p90, 0)}")
    lines.append(f"Prompt reuse ratio (tokens) : {fmt(s.sessions.prompt_reuse_ratio_tokens, 3)}")

    if s.sessions.sessions_detected and s.sessions.sessions_detected > 0:
        lines.append("")
        lines.append("SESSION CONTEXT")
        lines.append("--------------")
        lines.append(
            "Prompt tokens/turn P50/P90 : "
            f"{fmt(s.sessions.prompt_tokens_by_turn.p50,0)}/"
            f"{fmt(s.sessions.prompt_tokens_by_turn.p90,0)}"
        )
        lines.append(
            "Prompt growth Δtokens P50/P90 : "
            f"{fmt(s.sessions.prompt_token_growth.p50,0)}/"
            f"{fmt(s.sessions.prompt_token_growth.p90,0)}"
        )

    return "\n".join(lines)
