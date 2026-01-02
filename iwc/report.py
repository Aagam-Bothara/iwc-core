from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

# NEW: prompt token counting helper (you must implement this in iwc/analyze/tokenizer.py)
from iwc.analyze.tokenizer import count_tokens_for_prompt


def report_to_dict(r: "WorkloadReport", *, top_k_tags: int = 10) -> dict[str, Any]:
    return {
        "num_requests": r.num_requests,
        "arrival_time_ms": {
            "min": r.min_arrival_ms,
            "max": r.max_arrival_ms,
            "span": r.arrival_span_ms,
        },
        "coverage": {
            "semantic_present": r.semantic_present,
            "semantic_present_pct": _pct(r.semantic_present, r.num_requests),
            "streaming_true": r.streaming_true,
            "streaming_false": r.streaming_false,
            "session_id_present": r.session_id_present,
            "session_id_present_pct": _pct(r.session_id_present, r.num_requests),
            "turn_id_present": r.turn_id_present,
            "turn_id_present_pct": _pct(r.turn_id_present, r.num_requests),
            "prompt_format_missing": r.prompt_format_missing,
            "prompt_format_missing_pct": _pct(r.prompt_format_missing, r.num_requests),
            "task_present": r.task_present,
            "task_present_pct": _pct(r.task_present, r.num_requests),
            "difficulty_present": r.difficulty_present,
            "difficulty_present_pct": _pct(r.difficulty_present, r.num_requests),
            "tags_present": r.tags_present,
            "tags_present_pct": _pct(r.tags_present, r.num_requests),
        },
        "max_output_tokens": {
            "min": r.mot_min,
            "avg": r.mot_avg,
            "p50": r.mot_p50,
            "p90": r.mot_p90,
            "p99": r.mot_p99,
            "max": r.mot_max,
        },
        "prompt_chars": {
            "min": r.prompt_chars_min,
            "avg": r.prompt_chars_avg,
            "p50": r.prompt_chars_p50,
            "p90": r.prompt_chars_p90,
            "p99": r.prompt_chars_p99,
            "max": r.prompt_chars_max,
        },
        # NEW
        "prompt_tokens": {
            "min": r.prompt_tokens_min,
            "avg": r.prompt_tokens_avg,
            "p50": r.prompt_tokens_p50,
            "p90": r.prompt_tokens_p90,
            "p99": r.prompt_tokens_p99,
            "max": r.prompt_tokens_max,
        },
        "semantic": {
            "task_counts": dict(r.task_counts),
            "difficulty_counts": dict(r.difficulty_counts),
            "top_tags": [{"tag": k, "count": v} for k, v in r.tag_counts.most_common(top_k_tags)],
        },
        "prompt_format_counts": dict(r.prompt_format_counts),
        "warnings": list(r.warnings),
    }


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object per line")
            yield obj


def _pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def _percentile(sorted_vals: list[int], p: float) -> Optional[int]:
    """Nearest-rank-ish percentile. p in [0,100]."""
    if not sorted_vals:
        return None
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = int((p / 100.0) * (len(sorted_vals) - 1))
    return sorted_vals[k]


@dataclass
class WorkloadReport:
    num_requests: int
    min_arrival_ms: Optional[int]
    max_arrival_ms: Optional[int]
    arrival_span_ms: int

    mot_min: Optional[int]
    mot_avg: Optional[float]
    mot_p50: Optional[int]
    mot_p90: Optional[int]
    mot_p99: Optional[int]
    mot_max: Optional[int]

    prompt_chars_min: Optional[int]
    prompt_chars_avg: Optional[float]
    prompt_chars_p50: Optional[int]
    prompt_chars_p90: Optional[int]
    prompt_chars_p99: Optional[int]
    prompt_chars_max: Optional[int]

    # NEW
    prompt_tokens_min: Optional[int]
    prompt_tokens_avg: Optional[float]
    prompt_tokens_p50: Optional[int]
    prompt_tokens_p90: Optional[int]
    prompt_tokens_p99: Optional[int]
    prompt_tokens_max: Optional[int]

    task_counts: Counter[str]
    difficulty_counts: Counter[str]
    tag_counts: Counter[str]

    semantic_present: int
    streaming_true: int
    streaming_false: int

    prompt_format_counts: Counter[str]
    prompt_format_missing: int

    session_id_present: int
    turn_id_present: int

    # per-request presence (more robust than summing counters)
    task_present: int
    difficulty_present: int
    tags_present: int

    warnings: Tuple[str, ...]


def build_report(path: Path) -> WorkloadReport:
    arrivals: list[int] = []
    max_out: list[int] = []
    prompt_chars: list[int] = []
    prompt_tokens: list[int] = []  # NEW

    task_counts: Counter[str] = Counter()
    difficulty_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()

    semantic_present = 0
    streaming_true = 0
    streaming_false = 0

    prompt_format_counts: Counter[str] = Counter()
    prompt_format_missing = 0

    session_id_present = 0
    turn_id_present = 0

    task_present = 0
    difficulty_present = 0
    tags_present = 0

    n = 0
    for req in read_jsonl(path):
        n += 1

        # streaming
        streaming = req.get("streaming")
        if streaming is True:
            streaming_true += 1
        elif streaming is False:
            streaming_false += 1

        # prompt_format
        pf = req.get("prompt_format")
        if isinstance(pf, str) and pf.strip():
            prompt_format_counts[pf] += 1
        else:
            prompt_format_counts["(missing)"] += 1
            prompt_format_missing += 1

        # session_id / turn_id
        sid = req.get("session_id")
        if isinstance(sid, str) and sid.strip():
            session_id_present += 1

        tid = req.get("turn_id")
        if isinstance(tid, int):
            turn_id_present += 1

        # arrival_time_ms (required by schema)
        at = req.get("arrival_time_ms")
        if isinstance(at, int):
            arrivals.append(at)

        # max_output_tokens (required by schema)
        mot = req.get("max_output_tokens")
        if isinstance(mot, int):
            max_out.append(mot)

        # prompt (required by schema)
        prompt = req.get("prompt")
        if isinstance(prompt, str):
            prompt_chars.append(len(prompt))

            # Token count: use prompt_format if present, else assume raw
            pf2 = pf if (isinstance(pf, str) and pf.strip()) else "raw"
            try:
                pt = count_tokens_for_prompt(prompt, prompt_format=str(pf2))
                prompt_tokens.append(int(pt))
            except Exception:
                pass


            # NEW: token count (assume raw if prompt_format missing)
            pf2 = pf if (isinstance(pf, str) and pf.strip()) else "raw"
            try:
                pt = count_tokens_for_prompt(prompt, prompt_format=str(pf2))
                if isinstance(pt, int):
                    prompt_tokens.append(pt)
            except Exception:
                # robust: keep report working even if tokenizer fails on some prompt formats
                pass

        # semantic (optional)
        sem = req.get("semantic")
        if isinstance(sem, dict):
            semantic_present += 1

            # task
            task = sem.get("task")
            if isinstance(task, str) and task.strip():
                task_present += 1
                task_counts[task] += 1

            # difficulty
            diff = sem.get("difficulty")
            if isinstance(diff, str) and diff.strip():
                difficulty_present += 1
                difficulty_counts[diff] += 1

            # tags
            tags = sem.get("tags")
            if isinstance(tags, list):
                any_tag = False
                for t in tags:
                    if isinstance(t, str) and t.strip():
                        any_tag = True
                        tag_counts[t] += 1
                if any_tag:
                    tags_present += 1

    min_at = min(arrivals) if arrivals else None
    max_at = max(arrivals) if arrivals else None
    span = int(max_at - min_at) if (min_at is not None and max_at is not None) else 0

    def stats(
        vals: list[int],
    ) -> Tuple[Optional[int], Optional[float], Optional[int], Optional[int], Optional[int], Optional[int]]:
        if not vals:
            return (None, None, None, None, None, None)
        s = sorted(vals)
        avg = sum(s) / len(s)
        return (
            s[0],
            avg,
            _percentile(s, 50),
            _percentile(s, 90),
            _percentile(s, 99),
            s[-1],
        )

    mot_min, mot_avg, mot_p50, mot_p90, mot_p99, mot_max = stats(max_out)
    pc_min, pc_avg, pc_p50, pc_p90, pc_p99, pc_max = stats(prompt_chars)
    pt_min, pt_avg, pt_p50, pt_p90, pt_p99, pt_max = stats(prompt_tokens)  # NEW

    warnings: list[str] = []
    if n > 0:
        if prompt_format_missing > 0:
            warnings.append(
                f"prompt_format missing in {prompt_format_missing}/{n} ({round(_pct(prompt_format_missing, n), 2)}%)"
            )
        missing_sem = n - semantic_present
        if missing_sem > 0:
            warnings.append(f"semantic missing in {missing_sem}/{n} ({round(_pct(missing_sem, n), 2)}%)")
        missing_task = n - task_present
        if missing_task > 0:
            warnings.append(f"semantic.task missing in {missing_task}/{n} ({round(_pct(missing_task, n), 2)}%)")
        missing_diff = n - difficulty_present
        if missing_diff > 0:
            warnings.append(
                f"semantic.difficulty missing in {missing_diff}/{n} ({round(_pct(missing_diff, n), 2)}%)"
            )

        # NEW: token stat missing warning (helps later predict correctness)
        if prompt_chars and not prompt_tokens:
            warnings.append("prompt_tokens could not be computed (tokenizer unavailable or failed)")

    return WorkloadReport(
        num_requests=n,
        min_arrival_ms=min_at,
        max_arrival_ms=max_at,
        arrival_span_ms=span,
        mot_min=mot_min,
        mot_avg=mot_avg,
        mot_p50=mot_p50,
        mot_p90=mot_p90,
        mot_p99=mot_p99,
        mot_max=mot_max,
        prompt_chars_min=pc_min,
        prompt_chars_avg=pc_avg,
        prompt_chars_p50=pc_p50,
        prompt_chars_p90=pc_p90,
        prompt_chars_p99=pc_p99,
        prompt_chars_max=pc_max,
        prompt_tokens_min=pt_min,
        prompt_tokens_avg=pt_avg,
        prompt_tokens_p50=pt_p50,
        prompt_tokens_p90=pt_p90,
        prompt_tokens_p99=pt_p99,
        prompt_tokens_max=pt_max,
        task_counts=task_counts,
        difficulty_counts=difficulty_counts,
        tag_counts=tag_counts,
        semantic_present=semantic_present,
        streaming_true=streaming_true,
        streaming_false=streaming_false,
        prompt_format_counts=prompt_format_counts,
        prompt_format_missing=prompt_format_missing,
        session_id_present=session_id_present,
        turn_id_present=turn_id_present,
        task_present=task_present,
        difficulty_present=difficulty_present,
        tags_present=tags_present,
        warnings=tuple(warnings),
    )


def format_report(r: WorkloadReport, *, top_k_tags: int = 10) -> str:
    lines: list[str] = []

    lines.append("IWC Workload Report")
    lines.append("-" * 60)
    lines.append(f"requests: {r.num_requests}")
    lines.append(f"arrival_time_ms: min={r.min_arrival_ms}  max={r.max_arrival_ms}  span={r.arrival_span_ms}")

    if r.warnings:
        lines.append("")
        lines.append("WARNINGS:")
        for w in r.warnings:
            lines.append(f"  - {w}")

    lines.append("")
    lines.append("coverage:")
    lines.append(
        f"  semantic present: {r.semantic_present}/{r.num_requests} ({round(_pct(r.semantic_present, r.num_requests), 2)}%)"
    )
    lines.append(f"  streaming: true={r.streaming_true}  false={r.streaming_false}")
    lines.append(
        f"  session_id present: {r.session_id_present}/{r.num_requests} ({round(_pct(r.session_id_present, r.num_requests), 2)}%)"
    )
    lines.append(
        f"  turn_id present: {r.turn_id_present}/{r.num_requests} ({round(_pct(r.turn_id_present, r.num_requests), 2)}%)"
    )
    lines.append(f"  tags present: {r.tags_present}/{r.num_requests} ({round(_pct(r.tags_present, r.num_requests), 2)}%)")

    lines.append("")
    lines.append("max_output_tokens:")
    lines.append(
        f"  min={r.mot_min}  avg={None if r.mot_avg is None else round(r.mot_avg, 2)}  "
        f"p50={r.mot_p50}  p90={r.mot_p90}  p99={r.mot_p99}  max={r.mot_max}"
    )

    lines.append("")
    lines.append("prompt length (chars):")
    lines.append(
        f"  min={r.prompt_chars_min}  avg={None if r.prompt_chars_avg is None else round(r.prompt_chars_avg, 2)}  "
        f"p50={r.prompt_chars_p50}  p90={r.prompt_chars_p90}  p99={r.prompt_chars_p99}  max={r.prompt_chars_max}"
    )

    # NEW
    lines.append("")
    lines.append("prompt length (tokens):")
    lines.append(
        f"  min={r.prompt_tokens_min}  avg={None if r.prompt_tokens_avg is None else round(r.prompt_tokens_avg, 2)}  "
        f"p50={r.prompt_tokens_p50}  p90={r.prompt_tokens_p90}  p99={r.prompt_tokens_p99}  max={r.prompt_tokens_max}"
    )

    # semantic.task distribution
    lines.append("")
    lines.append("semantic.task distribution:")
    if r.task_counts:
        for task, cnt in r.task_counts.most_common():
            lines.append(f"  - {task}: {cnt} ({round(_pct(cnt, r.num_requests), 2)}%)")
        missing = r.num_requests - r.task_present
        if missing > 0:
            lines.append(f"  - (missing): {missing} ({round(_pct(missing, r.num_requests), 2)}%)")
    else:
        lines.append("  (none)")

    # semantic.difficulty distribution
    lines.append("")
    lines.append("semantic.difficulty distribution:")
    if r.difficulty_counts:
        for diff, cnt in r.difficulty_counts.most_common():
            lines.append(f"  - {diff}: {cnt} ({round(_pct(cnt, r.num_requests), 2)}%)")
        missing = r.num_requests - r.difficulty_present
        if missing > 0:
            lines.append(f"  - (missing): {missing} ({round(_pct(missing, r.num_requests), 2)}%)")
    else:
        lines.append("  (none)")

    # prompt_format distribution
    lines.append("")
    lines.append("prompt_format distribution:")
    if r.prompt_format_counts:
        for k, v in r.prompt_format_counts.most_common():
            lines.append(f"  - {k}: {v} ({round(_pct(v, r.num_requests), 2)}%)")
    else:
        lines.append("  (none)")

    # top tags
    lines.append("")
    lines.append(f"top tags (top {top_k_tags}):")
    if r.tag_counts:
        for tag, cnt in r.tag_counts.most_common(top_k_tags):
            lines.append(f"  - {tag}: {cnt}")
    else:
        lines.append("  (none)")

    return "\n".join(lines)
