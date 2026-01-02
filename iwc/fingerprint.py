from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Tuple

FINGERPRINT_VERSION = "0.2"


def _canonical_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_canonical_json(obj: Any) -> str:
    s = _canonical_dumps(obj).encode("utf-8")
    h = hashlib.sha256(s).hexdigest()
    return f"sha256:{h}"


def _require(d: Dict[str, Any], key: str) -> Any:
    if key not in d:
        raise KeyError(f"fingerprint: missing required field in report json: {key}")
    return d[key]


def _require_stats(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = _require(d, key)
    if not isinstance(v, dict):
        raise TypeError(f"fingerprint: {key} must be an object, got {type(v)}")
    if "p50" not in v or "p90" not in v:
        raise KeyError(f"fingerprint: {key} must include at least p50 and p90")
    out = {"p50": v["p50"], "p90": v["p90"]}
    if "p99" in v:
        out["p99"] = v["p99"]
    return out


def build_fingerprint_from_report_json(rj: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    # core token stats (now real tokens)
    prompt_tokens = _require_stats(rj, "prompt_tokens")
    max_output_tokens = _require_stats(rj, "max_output_tokens")

    # keep prompt_chars as auxiliary (helpful for debugging)
    prompt_chars = _require_stats(rj, "prompt_chars")

    arrival = _require(rj, "arrival_time_ms")
    if not isinstance(arrival, dict) or "span" not in arrival:
        raise KeyError("fingerprint: arrival_time_ms must be an object with key 'span'")

    coverage = _require(rj, "coverage")
    if not isinstance(coverage, dict):
        raise TypeError("fingerprint: coverage must be an object")

    has_sessions = float(coverage.get("session_id_present_pct", 0.0)) > 0.0
    sessions_mismatch_flag = False  # future: compute real mismatch

    core_fp: Dict[str, Any] = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "workload": {
            "num_requests": int(_require(rj, "num_requests")),
        },
        "token": {
            "prompt_tokens": prompt_tokens,
            "max_output_tokens": max_output_tokens,
            "prompt_chars": prompt_chars,
        },
        "arrival": {
            "span_ms": int(arrival["span"]),
            "min_ms": int(arrival.get("min", 0)),
            "max_ms": int(arrival.get("max", arrival["span"])),
        },
        "session": {
            "has_sessions": bool(has_sessions),
            "sessions_mismatch_flag": bool(sessions_mismatch_flag),
        },
    }

    workload_hash = sha256_canonical_json(core_fp)
    fp = dict(core_fp)
    fp["workload_hash"] = workload_hash
    return fp, workload_hash
