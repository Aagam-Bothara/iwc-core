from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from iwc.analyze.summary import WorkloadSummary


def _is_nan(x: float) -> bool:
    return x != x


def _fmt_num(x: float, nd: int = 2) -> str:
    if _is_nan(x):
        return "n/a"
    return f"{x:.{nd}f}"


def _fmt_int(x: float) -> str:
    if _is_nan(x):
        return "n/a"
    return f"{int(round(x))}"


def _fmt_delta(a: float, b: float, nd: int = 2, signed: bool = True) -> str:
    if _is_nan(a) or _is_nan(b):
        return "n/a"
    d = b - a
    if signed:
        return f"{d:+.{nd}f}"
    return f"{d:.{nd}f}"


def _delta_is_zero(delta: str) -> bool:
    """
    Treat these as "no change" for text output filtering:
      "", "+0", "+0.0", "+0.00", "+0.000", "0", "0.00", etc.
    """
    if not delta:
        return True
    d = delta.strip()
    if d == "n/a":
        return False
    if d.startswith("+"):
        d = d[1:]
    if d.startswith("-"):
        d = d[1:]
    try:
        return float(d) == 0.0
    except ValueError:
        return False


@dataclass(frozen=True)
class FieldDiff:
    label: str
    a: str
    b: str
    delta: str


@dataclass(frozen=True)
class SummaryDiff:
    a: WorkloadSummary
    b: WorkloadSummary
    rows: List[FieldDiff]


def diff_summaries(a: WorkloadSummary, b: WorkloadSummary) -> SummaryDiff:
    rows: List[FieldDiff] = []

    rows.append(FieldDiff("Tokenizer", a.tokenizer_used, b.tokenizer_used, ""))
    rows.append(FieldDiff("Requests", str(a.requests), str(b.requests), _fmt_delta(float(a.requests), float(b.requests), 0)))

    rows.append(FieldDiff("Prompt tokens P50", _fmt_int(a.prompt_tokens.p50), _fmt_int(b.prompt_tokens.p50), _fmt_delta(a.prompt_tokens.p50, b.prompt_tokens.p50, 0)))
    rows.append(FieldDiff("Prompt tokens P90", _fmt_int(a.prompt_tokens.p90), _fmt_int(b.prompt_tokens.p90), _fmt_delta(a.prompt_tokens.p90, b.prompt_tokens.p90, 0)))
    rows.append(FieldDiff("Prompt tokens P99", _fmt_int(a.prompt_tokens.p99), _fmt_int(b.prompt_tokens.p99), _fmt_delta(a.prompt_tokens.p99, b.prompt_tokens.p99, 0)))

    rows.append(FieldDiff("Max output cap P90", _fmt_int(a.max_output_tokens.p90), _fmt_int(b.max_output_tokens.p90), _fmt_delta(a.max_output_tokens.p90, b.max_output_tokens.p90, 0)))

    rows.append(FieldDiff("Prefill dominance P50", _fmt_num(a.prefill_dominance.p50, 3), _fmt_num(b.prefill_dominance.p50, 3), _fmt_delta(a.prefill_dominance.p50, b.prefill_dominance.p50, 3)))
    rows.append(FieldDiff("Prefill dominance P90", _fmt_num(a.prefill_dominance.p90, 3), _fmt_num(b.prefill_dominance.p90, 3), _fmt_delta(a.prefill_dominance.p90, b.prefill_dominance.p90, 3)))

    rows.append(FieldDiff("Duration (s)", _fmt_num(a.arrivals.duration_s, 2), _fmt_num(b.arrivals.duration_s, 2), _fmt_delta(a.arrivals.duration_s, b.arrivals.duration_s, 2)))
    rows.append(FieldDiff("Mean RPS", _fmt_num(a.arrivals.mean_rps, 2), _fmt_num(b.arrivals.mean_rps, 2), _fmt_delta(a.arrivals.mean_rps, b.arrivals.mean_rps, 2)))
    rows.append(FieldDiff("Peak reqs (1s bin)", _fmt_int(a.arrivals.peak_rps_1s), _fmt_int(b.arrivals.peak_rps_1s), _fmt_delta(a.arrivals.peak_rps_1s, b.arrivals.peak_rps_1s, 0)))
    rows.append(FieldDiff("Inter-arrival ms P50", _fmt_int(a.arrivals.interarrival_ms.p50), _fmt_int(b.arrivals.interarrival_ms.p50), _fmt_delta(a.arrivals.interarrival_ms.p50, b.arrivals.interarrival_ms.p50, 0)))
    rows.append(FieldDiff("Inter-arrival ms P90", _fmt_int(a.arrivals.interarrival_ms.p90), _fmt_int(b.arrivals.interarrival_ms.p90), _fmt_delta(a.arrivals.interarrival_ms.p90, b.arrivals.interarrival_ms.p90, 0)))
    rows.append(FieldDiff("Burstiness (CV)", _fmt_num(a.arrivals.burstiness_cv, 2), _fmt_num(b.arrivals.burstiness_cv, 2), _fmt_delta(a.arrivals.burstiness_cv, b.arrivals.burstiness_cv, 2)))

    rows.append(FieldDiff("Sessions detected", str(a.sessions.sessions_detected), str(b.sessions.sessions_detected), _fmt_delta(float(a.sessions.sessions_detected), float(b.sessions.sessions_detected), 0)))
    rows.append(FieldDiff("Turns/session P90", _fmt_int(a.sessions.turns_per_session.p90), _fmt_int(b.sessions.turns_per_session.p90), _fmt_delta(a.sessions.turns_per_session.p90, b.sessions.turns_per_session.p90, 0)))
    rows.append(FieldDiff("Prompt reuse (tokens)", _fmt_num(a.sessions.prompt_reuse_ratio_tokens, 3), _fmt_num(b.sessions.prompt_reuse_ratio_tokens, 3), _fmt_delta(a.sessions.prompt_reuse_ratio_tokens, b.sessions.prompt_reuse_ratio_tokens, 3)))
    rows.append(FieldDiff("Prompt tokens/turn P50", _fmt_int(a.sessions.prompt_tokens_by_turn.p50), _fmt_int(b.sessions.prompt_tokens_by_turn.p50), _fmt_delta(a.sessions.prompt_tokens_by_turn.p50, b.sessions.prompt_tokens_by_turn.p50, 0)))
    rows.append(FieldDiff("Prompt tokens/turn P90", _fmt_int(a.sessions.prompt_tokens_by_turn.p90), _fmt_int(b.sessions.prompt_tokens_by_turn.p90), _fmt_delta(a.sessions.prompt_tokens_by_turn.p90, b.sessions.prompt_tokens_by_turn.p90, 0)))
    rows.append(FieldDiff("Δtokens/turn P50", _fmt_int(a.sessions.prompt_token_growth.p50), _fmt_int(b.sessions.prompt_token_growth.p50), _fmt_delta(a.sessions.prompt_token_growth.p50, b.sessions.prompt_token_growth.p50, 0)))
    rows.append(FieldDiff("Δtokens/turn P90", _fmt_int(a.sessions.prompt_token_growth.p90), _fmt_int(b.sessions.prompt_token_growth.p90), _fmt_delta(a.sessions.prompt_token_growth.p90, b.sessions.prompt_token_growth.p90, 0)))

    return SummaryDiff(a=a, b=b, rows=rows)


def _infer_primary(s: WorkloadSummary) -> str:
    if s.sessions.sessions_detected and not _is_nan(s.sessions.prompt_reuse_ratio_tokens) and s.sessions.prompt_reuse_ratio_tokens > 0.5:
        if not _is_nan(s.prefill_dominance.p50) and s.prefill_dominance.p50 > 0.65:
            return "interactive-chat (prefill-heavy)"
        return "interactive-chat"
    if not _is_nan(s.arrivals.burstiness_cv) and s.arrivals.burstiness_cv > 1.5:
        return "bursty-api"
    return "batch/offline"


def _direction_hint(a: WorkloadSummary, b: WorkloadSummary) -> str:
    hints: List[str] = []

    if not _is_nan(a.arrivals.burstiness_cv) and not _is_nan(b.arrivals.burstiness_cv):
        if b.arrivals.burstiness_cv - a.arrivals.burstiness_cv > 0.5:
            hints.append("more bursty")
        elif a.arrivals.burstiness_cv - b.arrivals.burstiness_cv > 0.5:
            hints.append("less bursty")

    if not _is_nan(a.prefill_dominance.p50) and not _is_nan(b.prefill_dominance.p50):
        if b.prefill_dominance.p50 - a.prefill_dominance.p50 > 0.05:
            hints.append("more prefill-heavy")
        elif a.prefill_dominance.p50 - b.prefill_dominance.p50 > 0.05:
            hints.append("less prefill-heavy")

    if not _is_nan(a.sessions.prompt_reuse_ratio_tokens) and not _is_nan(b.sessions.prompt_reuse_ratio_tokens):
        if b.sessions.prompt_reuse_ratio_tokens - a.sessions.prompt_reuse_ratio_tokens > 0.05:
            hints.append("higher reuse")
        elif a.sessions.prompt_reuse_ratio_tokens - b.sessions.prompt_reuse_ratio_tokens > 0.05:
            hints.append("lower reuse")

    return ", ".join(hints) if hints else "no major shift detected"


def render_diff(d: SummaryDiff, a_label: str = "A", b_label: str = "B", only_changed: bool = False) -> str:
    lines: List[str] = []
    lines.append("WORKLOAD DIFF")
    lines.append("-------------")
    lines.append(f"A (baseline) : {a_label}")
    lines.append(f"B (candidate): {b_label}")
    lines.append("")
    lines.append(f"Primary class A : {_infer_primary(d.a)}")
    lines.append(f"Primary class B : {_infer_primary(d.b)}")
    lines.append(f"Shift           : {_direction_hint(d.a, d.b)}")
    lines.append("")

    rows = d.rows
    if only_changed:
        rows = [r for r in rows if not _delta_is_zero(r.delta)]

    col1 = max(len(r.label) for r in rows) if rows else 10
    col2 = max(len(r.a) for r in rows) if rows else 10
    col3 = max(len(r.b) for r in rows) if rows else 10

    header = f"{'Metric'.ljust(col1)}  {'A'.ljust(col2)}  {'B'.ljust(col3)}  Δ(B-A)"
    lines.append(header)
    lines.append("-" * len(header))

    for r in rows:
        lines.append(f"{r.label.ljust(col1)}  {r.a.ljust(col2)}  {r.b.ljust(col3)}  {r.delta}")

    return "\n".join(lines)


def diff_to_dict(d: SummaryDiff, a_label: str = "A", b_label: str = "B") -> Dict[str, Any]:
    return {
        "a_label": a_label,
        "b_label": b_label,
        "primary_class_a": _infer_primary(d.a),
        "primary_class_b": _infer_primary(d.b),
        "shift": _direction_hint(d.a, d.b),
        "metrics": [
            {"metric": r.label, "a": r.a, "b": r.b, "delta": r.delta}
            for r in d.rows
        ],
    }
def check_regressions(
    d: SummaryDiff,
    burstiness_delta: float | None = None,
    prefill_p50_delta: float | None = None,
    reuse_delta: float | None = None,
    prompt_p50_delta: float | None = None,
    prompt_p90_delta: float | None = None,
) -> List[str]:
    """
    Returns a list of human-readable regression strings.
    If empty -> OK.
    All checks are absolute deltas |B-A|.
    """
    msgs: List[str] = []

    a = d.a
    b = d.b

    def abs_delta(xa: float, xb: float) -> float | None:
        if _is_nan(xa) or _is_nan(xb):
            return None
        return abs(xb - xa)

    # Burstiness CV
    if burstiness_delta is not None:
        v = abs_delta(a.arrivals.burstiness_cv, b.arrivals.burstiness_cv)
        if v is not None and v > burstiness_delta:
            msgs.append(f"Burstiness CV changed by {v:.3f} (> {burstiness_delta:.3f})")

    # Prefill dominance P50
    if prefill_p50_delta is not None:
        v = abs_delta(a.prefill_dominance.p50, b.prefill_dominance.p50)
        if v is not None and v > prefill_p50_delta:
            msgs.append(f"Prefill dominance P50 changed by {v:.3f} (> {prefill_p50_delta:.3f})")

    # Reuse (tokens)
    if reuse_delta is not None:
        v = abs_delta(a.sessions.prompt_reuse_ratio_tokens, b.sessions.prompt_reuse_ratio_tokens)
        if v is not None and v > reuse_delta:
            msgs.append(f"Prompt reuse (tokens) changed by {v:.3f} (> {reuse_delta:.3f})")

    # Prompt tokens P50/P90
    if prompt_p50_delta is not None:
        v = abs_delta(a.prompt_tokens.p50, b.prompt_tokens.p50)
        if v is not None and v > prompt_p50_delta:
            msgs.append(f"Prompt tokens P50 changed by {v:.1f} (> {prompt_p50_delta:.1f})")

    if prompt_p90_delta is not None:
        v = abs_delta(a.prompt_tokens.p90, b.prompt_tokens.p90)
        if v is not None and v > prompt_p90_delta:
            msgs.append(f"Prompt tokens P90 changed by {v:.1f} (> {prompt_p90_delta:.1f})")

    return msgs
