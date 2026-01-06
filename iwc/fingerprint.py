# iwc/fingerprint.py
"""
Workload Fingerprint (v0.2)

Goals:
- Deterministic, stable fingerprint for a canonical workload JSONL.
- No circular imports (fingerprint is standalone).
- Fields match what predict.py expects:
  - workload.num_requests
  - arrival.span_ms
  - token.prompt_tokens.{p50,p90,p99}
  - token.max_output_tokens.{p50,p90,p99}
  - semantic.task_counts, difficulty_counts, top_tags
- workload_hash is stable across runs.

This file intentionally computes fingerprint directly from JSONL (source of truth).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Helpers
# -----------------------------
def _canonical_json(obj: Any) -> str:
    """Deterministic JSON string (for hashing)."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _percentile_int(xs: List[int], p: float) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    # nearest-rank-ish (deterministic, simple)
    idx = int(p * (len(s) - 1) + 0.5)
    if idx < 0:
        idx = 0
    if idx >= len(s):
        idx = len(s) - 1
    return int(s[idx])


def _safe_int(v: Any, default: int = 0) -> int:
    if isinstance(v, bool):
        return default
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    return default


def _extract_prompt(req: Dict[str, Any]) -> str:
    p = req.get("prompt")
    if isinstance(p, str):
        return p
    # if openai_messages etc
    try:
        return _canonical_json(p)
    except Exception:
        return ""


def _extract_prompt_tokens(req: Dict[str, Any]) -> int:
    """
    If your workload already stores token counts, honor them.
    Otherwise fallback to a rough proxy (word-ish count).
    We keep it deterministic; accuracy isn't the goal here.
    """
    t = req.get("prompt_tokens")
    if isinstance(t, int) and t >= 0:
        return t

    # Some formats store token counts inside "token" or "usage"
    tok = req.get("token", {})
    if isinstance(tok, dict):
        pt = tok.get("prompt_tokens")
        if isinstance(pt, int) and pt >= 0:
            return pt

    usage = req.get("usage", {})
    if isinstance(usage, dict):
        pt = usage.get("prompt_tokens")
        if isinstance(pt, int) and pt >= 0:
            return pt

    # Deterministic fallback: whitespace split length
    prompt = _extract_prompt(req)
    if not prompt:
        return 0
    return max(1, len(prompt.split()))


def _extract_max_output_tokens(req: Dict[str, Any]) -> int:
    v = req.get("max_output_tokens")
    if isinstance(v, int) and v > 0:
        return v

    # Some formats use max_tokens
    v2 = req.get("max_tokens")
    if isinstance(v2, int) and v2 > 0:
        return v2

    return 0


def _extract_arrival_ms(req: Dict[str, Any]) -> int:
    v = req.get("arrival_time_ms", 0)
    if isinstance(v, (int, float)) and v >= 0:
        return int(v)
    return 0


def _extract_semantic(req: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], List[str]]:
    sem = req.get("semantic")
    if not isinstance(sem, dict):
        return None, None, []
    task = sem.get("task")
    diff = sem.get("difficulty")
    tags = sem.get("tags")

    task_s = task if isinstance(task, str) and task.strip() else None
    diff_s = diff if isinstance(diff, str) and diff.strip() else None

    tag_list: List[str] = []
    if isinstance(tags, list):
        for t in tags:
            if isinstance(t, str) and t.strip():
                tag_list.append(t.strip())
    return task_s, diff_s, tag_list


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise SystemExit(f"{path}:{line_no}: expected JSON object")
            out.append(obj)
    return out


def _stable_request_view(req: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hash input is a stable, reduced projection.
    We avoid including volatile fields (timestamps, measurements).
    """
    task, diff, tags = _extract_semantic(req)
    view = {
        "prompt": req.get("prompt"),
        "prompt_format": req.get("prompt_format"),
        "max_output_tokens": _extract_max_output_tokens(req),
        "arrival_time_ms": _extract_arrival_ms(req),
        "semantic": {
            "task": task,
            "difficulty": diff,
            "tags": sorted(tags),
        },
        "session_id": req.get("session_id"),
        "turn_id": req.get("turn_id"),
        "streaming": req.get("streaming"),
    }
    return view


# -----------------------------
# Public API
# -----------------------------
def build_fingerprint(workload_jsonl: Path) -> Dict[str, Any]:
    """
    Primary fingerprint used by predict.py.

    Returns dict INCLUDING workload_hash (stable).
    """
    reqs = _read_jsonl(workload_jsonl)

    num_requests = len(reqs)

    arrivals = [_extract_arrival_ms(r) for r in reqs]
    min_a = min(arrivals) if arrivals else 0
    max_a = max(arrivals) if arrivals else 0
    span_ms = max(0, max_a - min_a)

    prompt_tokens = [_extract_prompt_tokens(r) for r in reqs]
    max_out = [_extract_max_output_tokens(r) for r in reqs]

    # semantic counts
    task_counts: Dict[str, int] = {}
    difficulty_counts: Dict[str, int] = {}
    tag_counts: Dict[str, int] = {}

    for r in reqs:
        task, diff, tags = _extract_semantic(r)
        if task:
            task_counts[task] = task_counts.get(task, 0) + 1
        if diff:
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    top_tags = sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    top_tags_list = [t for (t, _) in top_tags]

    # deterministic hash based on stable projection of each request
    stable_views = [_stable_request_view(r) for r in reqs]
    hash_input = _canonical_json(stable_views)
    workload_hash = _sha256_hex(hash_input)

    fp = {
        "fingerprint_version": "0.2",
        "workload_hash": workload_hash,
        "workload": {"num_requests": num_requests},
        "arrival": {"span_ms": span_ms},
        "token": {
            "prompt_tokens": {
                "p50": _percentile_int(prompt_tokens, 0.50),
                "p90": _percentile_int(prompt_tokens, 0.90),
                "p99": _percentile_int(prompt_tokens, 0.99),
            },
            "max_output_tokens": {
                "p50": _percentile_int(max_out, 0.50),
                "p90": _percentile_int(max_out, 0.90),
                "p99": _percentile_int(max_out, 0.99),
            },
        },
        "semantic": {
            "task_counts": task_counts,
            "difficulty_counts": difficulty_counts,
            "top_tags": top_tags_list,
        },
    }
    return fp


def build_fingerprint_extended(workload_jsonl: Path, include_distributions: bool = True) -> Dict[str, Any]:
    """
    Extended fingerprint: adds simple distributions (still deterministic).
    """
    fp = build_fingerprint(workload_jsonl)
    if not include_distributions:
        return fp

    reqs = _read_jsonl(workload_jsonl)
    prompt_tokens = [_extract_prompt_tokens(r) for r in reqs]
    max_out = [_extract_max_output_tokens(r) for r in reqs]
    arrivals = [_extract_arrival_ms(r) for r in reqs]

    fp["distributions"] = {
        "prompt_tokens": {
            "min": min(prompt_tokens) if prompt_tokens else 0,
            "max": max(prompt_tokens) if prompt_tokens else 0,
            "values": sorted(prompt_tokens) if len(prompt_tokens) <= 2000 else None,
        },
        "max_output_tokens": {
            "min": min(max_out) if max_out else 0,
            "max": max(max_out) if max_out else 0,
            "values": sorted(max_out) if len(max_out) <= 2000 else None,
        },
        "arrival_time_ms": {
            "min": min(arrivals) if arrivals else 0,
            "max": max(arrivals) if arrivals else 0,
            "values": sorted(arrivals) if len(arrivals) <= 2000 else None,
        },
    }
    return fp


def build_fingerprint_from_report_json(report_json: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Compatibility shim for older code paths that attempted to fingerprint from report JSON.

    This does NOT try to reinterpret unknown report schemas.
    Instead, it creates a deterministic hash of the provided report_json
    and returns a minimal fingerprint shell + that hash.

    Prefer build_fingerprint(<jsonl>) instead.
    """
    if not isinstance(report_json, dict):
        raise SystemExit("build_fingerprint_from_report_json: expected dict")

    workload_hash = _sha256_hex(_canonical_json(report_json))

    fp = {
        "fingerprint_version": "0.2",
        "workload_hash": workload_hash,
        "workload": {"num_requests": _safe_int(report_json.get("requests"), 0)},
        "arrival": {"span_ms": _safe_int(report_json.get("arrival", {}).get("span_ms"), 0) if isinstance(report_json.get("arrival"), dict) else 0},
        "token": {
            "prompt_tokens": {"p50": 0, "p90": 0, "p99": 0},
            "max_output_tokens": {"p50": 0, "p90": 0, "p99": 0},
        },
        "semantic": {"task_counts": {}, "difficulty_counts": {}, "top_tags": []},
        "note": "This fingerprint was derived from report_json; prefer build_fingerprint(jsonl) for full fidelity.",
    }
    return fp, workload_hash
