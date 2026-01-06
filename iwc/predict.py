# iwc/predict.py
"""
Enhanced Prediction Module for LLM Inference Workload Predictor (v2.1)

v2.1.3 FINAL fixes:
1) TTFT definition alignment:
   - predicted.ttft_p90_ms now represents TTFT_PROXY (no external queueing),
     matching meas_ttftP in eval output.
   - predicted.ttft_e2e_p90_ms added for human/diagnostic "client-perceived TTFT" (includes queueing).

2) Overload queueing for percentiles:
   - When rho >= 1, steady-state queue formulas diverge.
   - Use finite-horizon fluid transient with a p90 wait approximation:
       backlog_end = (lambda - mu_total) * T
       wait_end    = backlog_end / mu_total
       wait_p90    = 0.9 * wait_end
   - We feed wait_p90 into the p90 E2E pipeline.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from iwc.decision import decide, decision_to_dict
from iwc.profile import TargetProfile


# -----------------------------
# Calibration Data Structures
# -----------------------------
@dataclass(frozen=True)
class Calibration:
    cal_version: str
    engine: str
    model: str

    prefill_fixed_overhead_ms: float
    prefill_ms_per_token: float
    decode_fixed_overhead_ms: float
    decode_ms_per_token: float

    request_overhead_ms: float = 0.0
    request_overhead_std: float = 0.0

    prefill_fixed_ci: Tuple[float, float] = (0.0, 0.0)
    prefill_slope_ci: Tuple[float, float] = (0.0, 0.0)
    decode_fixed_ci: Tuple[float, float] = (0.0, 0.0)
    decode_slope_ci: Tuple[float, float] = (0.0, 0.0)

    kv_cache_pressure_ms_per_1k: float = 0.0
    kv_cache_threshold_tokens: int = 0

    decode_variance_coefficient: float = 0.0

    batch_overhead_ms_per_concurrent: float = 0.0
    max_efficient_batch: int = 1

    prefill_r_squared: float = 0.0
    decode_r_squared: float = 0.0

    warnings: List[str] = field(default_factory=list)
    quality: Dict[str, Any] = field(default_factory=dict)

    health: Optional[Dict[str, Any]] = None


def load_calibration(path: Path) -> Calibration:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit(f"{path}: calibration must be a JSON object")

    def get_str(k: str, default: str = "") -> str:
        v = obj.get(k)
        return v if isinstance(v, str) and v.strip() else default

    def get_float(k: str, default: float = 0.0) -> float:
        v = obj.get(k)
        if isinstance(v, (int, float)):
            return float(v)
        return default

    def get_tuple(k: str, default: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
        v = obj.get(k)
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return (float(v[0]), float(v[1]))
        return default

    def get_list(k: str) -> List[str]:
        v = obj.get(k)
        return [str(x) for x in v] if isinstance(v, list) else []

    def get_dict(k: str) -> Dict[str, Any]:
        v = obj.get(k)
        return v if isinstance(v, dict) else {}

    prefill_fixed = get_float("prefill_fixed_overhead_ms")
    prefill_slope = get_float("prefill_ms_per_token")
    decode_fixed = get_float("decode_fixed_overhead_ms")
    decode_slope = get_float("decode_ms_per_token")

    # Legacy fallback
    if prefill_fixed == 0.0 and prefill_slope == 0.0:
        prefill_tps = get_float("prefill_tps")
        decode_tps = get_float("decode_tps")
        overhead_ms = get_float("overhead_ms")
        if prefill_tps > 0 and decode_tps > 0:
            prefill_fixed = overhead_ms
            prefill_slope = 1000.0 / max(1e-9, prefill_tps)
            decode_fixed = overhead_ms
            decode_slope = 1000.0 / max(1e-9, decode_tps)

    if prefill_slope == 0.0 and decode_slope == 0.0:
        raise SystemExit(f"{path}: missing required calibration fields")

    health: Optional[Dict[str, Any]] = None
    dbg = obj.get("debug")
    if isinstance(dbg, dict):
        h = dbg.get("health")
        if isinstance(h, dict):
            health = h

    return Calibration(
        cal_version=get_str("cal_version", "unknown"),
        engine=get_str("engine", "unknown"),
        model=get_str("model", "unknown"),
        prefill_fixed_overhead_ms=prefill_fixed,
        prefill_ms_per_token=prefill_slope,
        decode_fixed_overhead_ms=decode_fixed,
        decode_ms_per_token=decode_slope,
        request_overhead_ms=get_float("request_overhead_ms"),
        request_overhead_std=get_float("request_overhead_std"),
        prefill_fixed_ci=get_tuple("prefill_fixed_ci"),
        prefill_slope_ci=get_tuple("prefill_slope_ci"),
        decode_fixed_ci=get_tuple("decode_fixed_ci"),
        decode_slope_ci=get_tuple("decode_slope_ci"),
        kv_cache_pressure_ms_per_1k=get_float("kv_cache_pressure_ms_per_1k"),
        kv_cache_threshold_tokens=int(get_float("kv_cache_threshold_tokens")),
        decode_variance_coefficient=get_float("decode_variance_coefficient"),
        batch_overhead_ms_per_concurrent=get_float("batch_overhead_ms_per_concurrent"),
        max_efficient_batch=int(get_float("max_efficient_batch", 1)),
        prefill_r_squared=get_float("prefill_r_squared"),
        decode_r_squared=get_float("decode_r_squared"),
        warnings=get_list("warnings"),
        quality=get_dict("quality"),
        health=health,
    )


# -----------------------------
# Queueing Theory Helpers
# -----------------------------
def _factorial(n: int) -> float:
    if n <= 1:
        return 1.0
    r = 1.0
    for i in range(2, n + 1):
        r *= i
    return r


def _erlang_c(c: int, rho: float) -> float:
    if c <= 0:
        return 0.0
    total_rho = c * rho
    if total_rho >= c:
        return 1.0
    if rho <= 0:
        return 0.0
    sum_term = sum((total_rho**k) / _factorial(k) for k in range(c))
    last_term = ((total_rho**c) / _factorial(c)) * (1 / (1 - rho))
    p0 = 1.0 / (sum_term + last_term)
    ec = last_term * p0
    return min(1.0, max(0.0, ec))


def _mmc_wait_time(arrival_rate: float, service_rate: float, c: int) -> Tuple[float, float, float]:
    if c <= 0 or service_rate <= 0:
        return 0.0, 0.0, 0.0
    rho = arrival_rate / (c * service_rate)
    if rho >= 1.0:
        return float("inf"), float("inf"), 1.0
    if rho <= 0:
        return 0.0, 0.0, 0.0
    p_queue = _erlang_c(c, rho)
    wq = (p_queue / (c * service_rate * (1 - rho)))
    lq = arrival_rate * wq
    return max(0.0, wq), max(0.0, lq), rho


def _kingman_approximation(
    arrival_rate: float,
    service_rate: float,
    c: int,
    ca_squared: float = 1.0,
    cs_squared: float = 0.5,
) -> float:
    if c <= 0 or service_rate <= 0:
        return 0.0
    rho = arrival_rate / (c * service_rate)
    if rho >= 1.0:
        return float("inf")
    if rho <= 0:
        return 0.0
    variability = (ca_squared + cs_squared) / 2
    wait = (
        (rho ** math.sqrt(2 * (c + 1)))
        / (c * (1 - rho))
        * variability
        * (1 / service_rate)
    )
    return max(0.0, wait)


def _fluid_overload_wait_p90(arrival_rate: float, mu_total: float, horizon_s: float) -> Tuple[float, float, float]:
    """
    Finite-horizon overload approximation for p90 waiting time.

    backlog_end = (lambda - mu_total) * T
    wait_end    = backlog_end / mu_total
    wait_p90    = 0.9 * wait_end  (linear backlog growth -> wait approx uniform over [0, wait_end])

    Returns: (Wq_p90_s, avg_backlog_reqs, backlog_end_reqs)
    """
    T = horizon_s if horizon_s and horizon_s > 0 else 10.0
    growth = max(0.0, arrival_rate - mu_total)
    backlog_end = growth * T
    avg_backlog = backlog_end / 2.0
    wait_end = backlog_end / max(1e-9, mu_total)
    wq_p90 = 0.9 * wait_end
    return max(0.0, wq_p90), max(0.0, avg_backlog), max(0.0, backlog_end)


# -----------------------------
# Latency Prediction
# -----------------------------
@dataclass
class LatencyBreakdown:
    request_overhead_ms: float
    prefill_ms: float
    decode_ms: float
    kv_cache_pressure_ms: float
    batch_overhead_ms: float
    queueing_delay_ms: float  # external queue delay (p90-ish when overloaded)

    @property
    def service_ms(self) -> float:
        return (
            self.request_overhead_ms
            + self.prefill_ms
            + self.decode_ms
            + self.kv_cache_pressure_ms
            + self.batch_overhead_ms
        )

    @property
    def ttft_proxy_ms(self) -> float:
        # Match meas_ttftP: prefill + kv pressure + batch/scheduler effect under concurrency
        return self.prefill_ms + self.kv_cache_pressure_ms + self.batch_overhead_ms


    @property
    def ttft_e2e_ms(self) -> float:
        """Client-perceived TTFT (kept as diagnostic): includes request overhead + queueing."""
        return self.request_overhead_ms + self.prefill_ms + self.kv_cache_pressure_ms + self.queueing_delay_ms

    @property
    def completion_ms(self) -> float:
        return self.service_ms + self.queueing_delay_ms


def _compute_percentile_latency(base_ms: float, variance_cv: float, percentile: float) -> float:
    if variance_cv <= 0 or percentile <= 0.5:
        return base_ms
    sigma2 = math.log(1 + variance_cv**2)
    sigma = math.sqrt(sigma2)
    mu = math.log(max(1e-9, base_ms))
    z_scores = {0.5: 0.0, 0.9: 1.282, 0.95: 1.645, 0.99: 2.326}
    z = z_scores.get(percentile, 1.282)
    return math.exp(mu + z * sigma)


def _predict_latency_breakdown(
    prompt_tokens: int,
    output_tokens: int,
    cal: Calibration,
    concurrency: int = 1,
    arrival_rate_rps: float = 0.0,
    horizon_s: float = 0.0,
) -> Tuple[LatencyBreakdown, Dict[str, float]]:
    prefill_ms = cal.prefill_fixed_overhead_ms + cal.prefill_ms_per_token * prompt_tokens
    decode_ms = cal.decode_fixed_overhead_ms + cal.decode_ms_per_token * output_tokens

    kv_pressure_ms = 0.0
    if cal.kv_cache_threshold_tokens > 0 and prompt_tokens > cal.kv_cache_threshold_tokens:
        excess = prompt_tokens - cal.kv_cache_threshold_tokens
        kv_pressure_ms = cal.kv_cache_pressure_ms_per_1k * (excess / 1000.0)

    batch_overhead_ms = 0.0
    if concurrency > 1:
        batch_overhead_ms = cal.batch_overhead_ms_per_concurrent * (concurrency - 1)

    service_time_s = (cal.request_overhead_ms + prefill_ms + decode_ms + kv_pressure_ms + batch_overhead_ms) / 1000.0
    service_rate = 1.0 / max(1e-9, service_time_s)
    mu_total = max(1e-9, concurrency * service_rate)

    queueing_delay_ms = 0.0
    qm: Dict[str, float] = {
        "service_time_s": service_time_s,
        "service_rate_rps": service_rate,
        "utilization": 0.0,
        "queue_probability": 0.0,
        "expected_queue_length": 0.0,
        "queueing_delay_ms": 0.0,
        "rho": 0.0,
        "overloaded": 0.0,
        "horizon_s": float(horizon_s if horizon_s and horizon_s > 0 else 0.0),
        "overload_backlog_end": 0.0,
    }

    if arrival_rate_rps > 0 and concurrency > 0:
        rho = arrival_rate_rps / max(1e-9, mu_total)
        qm["rho"] = float(rho)

        if rho >= 1.0:
            qm["overloaded"] = 1.0
            wq_p90_s, avg_backlog, backlog_end = _fluid_overload_wait_p90(
                arrival_rate=arrival_rate_rps,
                mu_total=mu_total,
                horizon_s=horizon_s if horizon_s and horizon_s > 0 else 10.0,
            )
            queueing_delay_ms = wq_p90_s * 1000.0
            qm["queue_probability"] = 1.0
            qm["expected_queue_length"] = float(avg_backlog)
            qm["queueing_delay_ms"] = float(queueing_delay_ms)
            qm["utilization"] = 1.0
            qm["overload_backlog_end"] = float(backlog_end)
        else:
            cs_sq = cal.decode_variance_coefficient**2 if cal.decode_variance_coefficient > 0 else 0.5
            qdelay_s = _kingman_approximation(
                arrival_rate=arrival_rate_rps,
                service_rate=service_rate,
                c=concurrency,
                ca_squared=1.0,
                cs_squared=cs_sq,
            )
            queueing_delay_ms = qdelay_s * 1000.0

            _, qlen, util = _mmc_wait_time(arrival_rate=arrival_rate_rps, service_rate=service_rate, c=concurrency)
            qprob = _erlang_c(concurrency, util) if util < 1 else 1.0

            qm.update(
                {
                    "utilization": float(util),
                    "queue_probability": float(qprob),
                    "expected_queue_length": float(qlen),
                    "queueing_delay_ms": float(queueing_delay_ms),
                }
            )

    bd = LatencyBreakdown(
        request_overhead_ms=cal.request_overhead_ms,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        kv_cache_pressure_ms=kv_pressure_ms,
        batch_overhead_ms=batch_overhead_ms,
        queueing_delay_ms=queueing_delay_ms,
    )
    return bd, qm


def _compute_confidence_interval(
    base_ms: float,
    cal: Calibration,
    prompt_tokens: int,
    output_tokens: int,
    kind: str,
) -> Tuple[float, float]:
    if kind == "ttft_proxy":
        fixed_ci = cal.prefill_fixed_ci
        slope_ci = cal.prefill_slope_ci
        tokens = prompt_tokens
        extra = 0.0  # ttft_proxy excludes request overhead
    else:
        fixed_ci = (
            cal.prefill_fixed_ci[0] + cal.decode_fixed_ci[0],
            cal.prefill_fixed_ci[1] + cal.decode_fixed_ci[1],
        )
        slope_ci = (
            cal.prefill_slope_ci[0] + cal.decode_slope_ci[0],
            cal.prefill_slope_ci[1] + cal.decode_slope_ci[1],
        )
        tokens = prompt_tokens + output_tokens
        extra = cal.request_overhead_ms

    if fixed_ci == (0.0, 0.0) and slope_ci == (0.0, 0.0):
        return (base_ms * 0.8, base_ms * 1.2)

    low = max(0.0, fixed_ci[0] + slope_ci[0] * tokens + extra)
    high = fixed_ci[1] + slope_ci[1] * tokens + extra

    if kind != "ttft_proxy" and cal.request_overhead_std > 0:
        low = max(0.0, low - 2 * cal.request_overhead_std)
        high = high + 2 * cal.request_overhead_std

    return (low, high)


def _estimate_sla_compliance(base_ms: float, variance_cv: float, thresholds_ms: List[int]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for thr in thresholds_ms:
        if thr <= 0:
            out[thr] = 0.0
            continue
        if variance_cv <= 0 or base_ms <= 0:
            out[thr] = 1.0 if base_ms <= thr else 0.0
            continue
        sigma2 = math.log(1 + variance_cv**2)
        sigma = math.sqrt(sigma2)
        mu = math.log(max(1e-9, base_ms))
        z = (math.log(thr) - mu) / sigma
        cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        out[thr] = max(0.0, min(1.0, cdf))
    return out


def _normalize_fingerprint(fp: Any) -> Tuple[Dict[str, Any], str]:
    if isinstance(fp, dict):
        wh = fp.get("workload_hash")
        if not isinstance(wh, str) or not wh.strip():
            raise SystemExit("predict: fingerprint missing workload_hash")
        return fp, wh
    if isinstance(fp, (list, tuple)) and len(fp) == 2:
        a, b = fp[0], fp[1]
        if not isinstance(a, dict):
            raise SystemExit("predict: fingerprint[0] must be dict")
        if not isinstance(b, str) or not b.strip():
            raise SystemExit("predict: fingerprint[1] must be hash string")
        a.setdefault("workload_hash", b)
        return a, b
    raise SystemExit("predict: unsupported fingerprint format")


def _infer_calibration_token_envelope(profile: TargetProfile, cal: Calibration) -> Tuple[Optional[int], Optional[int]]:
    max_ctx = getattr(profile, "max_context", None)
    max_prompt = int(max_ctx) if isinstance(max_ctx, int) and max_ctx > 0 else None
    max_output = 512
    return max_prompt, max_output


def predict_workload(
    *,
    workload_jsonl: Path,
    profile: TargetProfile,
    cal: Calibration,
    concurrency: int = 1,
    target_utilization: float = 0.7,
    sla_thresholds_ms: Optional[List[int]] = None,
    allow_mismatch: bool = False,
) -> Dict[str, Any]:
    from iwc.fingerprint import build_fingerprint

    if sla_thresholds_ms is None:
        sla_thresholds_ms = [100, 200, 500, 1000, 2000, 5000]

    fp_raw = build_fingerprint(workload_jsonl)
    fp, workload_hash = _normalize_fingerprint(fp_raw)

    prof_model = getattr(profile, "model", None)
    model_match = (prof_model == cal.model)
    if not model_match and not allow_mismatch:
        raise SystemExit(f"predict: model mismatch (profile={prof_model!r} != cal={cal.model!r})")

    prompt_p50 = int(fp["token"]["prompt_tokens"]["p50"])
    prompt_p90 = int(fp["token"]["prompt_tokens"]["p90"])
    output_p50 = int(fp["token"]["max_output_tokens"]["p50"])
    output_p90 = int(fp["token"]["max_output_tokens"]["p90"])
    num_requests = int(fp["workload"]["num_requests"])
    span_ms = int(fp["arrival"]["span_ms"])
    span_s = max(0.0, span_ms / 1000.0)

    offered_rps = 0.0
    if span_ms > 0 and num_requests > 1:
        offered_rps = num_requests / (span_ms / 1000.0)

    breakdown, queue_metrics = _predict_latency_breakdown(
        prompt_tokens=prompt_p90,
        output_tokens=output_p90,
        cal=cal,
        concurrency=concurrency,
        arrival_rate_rps=offered_rps,
        horizon_s=span_s if span_s > 0 else 0.0,
    )

    raw_cv = cal.decode_variance_coefficient if cal.decode_variance_coefficient > 0 else 0.15
    unclamped = raw_cv
    variance_cv = min(0.35, max(0.08, raw_cv))
    tail_cv_clamped = (abs(variance_cv - unclamped) > 1e-12)

    max_prompt_cal, max_output_cal = _infer_calibration_token_envelope(profile, cal)
    extrap_prompt = bool(max_prompt_cal is not None and prompt_p90 > max_prompt_cal)
    extrap_out = bool(max_output_cal is not None and output_p90 > max_output_cal)

    # ---- TTFT_PROXY (no queue) ----
    ttft_proxy_base = breakdown.ttft_proxy_ms
    ttft_p50 = ttft_proxy_base * 0.85
    ttft_p90 = ttft_proxy_base
    ttft_p95 = _compute_percentile_latency(ttft_proxy_base, variance_cv * 0.5, 0.95)
    ttft_p99 = _compute_percentile_latency(ttft_proxy_base, variance_cv * 0.5, 0.99)
    ttft_ci = _compute_confidence_interval(ttft_p90, cal, prompt_p90, 0, kind="ttft_proxy")

    # ---- Service (no external queue) ----
    svc_base = breakdown.service_ms
    svc_p50 = svc_base * 0.85
    svc_p90 = svc_base
    svc_p95 = _compute_percentile_latency(svc_base, variance_cv, 0.95)
    svc_p99 = _compute_percentile_latency(svc_base, variance_cv, 0.99)
    svc_ci = _compute_confidence_interval(svc_p90, cal, prompt_p90, output_p90, kind="service")

    # ---- E2E (includes external queue) ----
    e2e_base = breakdown.completion_ms
    e2e_p50 = e2e_base * 0.85
    e2e_p90 = e2e_base
    e2e_p95 = _compute_percentile_latency(e2e_base, variance_cv, 0.95)
    e2e_p99 = _compute_percentile_latency(e2e_base, variance_cv, 0.99)
    e2e_ci = _compute_confidence_interval(e2e_p90, cal, prompt_p90, output_p90, kind="e2e")

    # Diagnostic TTFT that includes queue (human-only)
    ttft_e2e_p90 = breakdown.ttft_e2e_ms

    service_time_s = float(queue_metrics.get("service_time_s", 0.0))
    max_throughput = concurrency / max(1e-9, service_time_s)
    sustainable_throughput = max_throughput * target_utilization

    sla_compliance = _estimate_sla_compliance(e2e_base, variance_cv, sla_thresholds_ms)

    warnings: List[str] = []
    warnings.extend([f"CAL: {w}" for w in cal.warnings])

    max_ctx = getattr(profile, "max_context", None)
    if isinstance(max_ctx, int) and max_ctx > 0 and prompt_p90 > int(0.9 * max_ctx):
        warnings.append(f"prompt_tokens_p90={prompt_p90} near context limit {max_ctx}")

    util = float(queue_metrics.get("utilization", 0.0) or 0.0)
    if util > 0.9:
        warnings.append(f"High utilization ({util:.1%}) may cause queueing delays")

    if float(queue_metrics.get("overloaded", 0.0) or 0.0) > 0.5:
        hs = float(queue_metrics.get("horizon_s", 0.0) or 0.0)
        rho = float(queue_metrics.get("rho", 0.0) or 0.0)
        warnings.append(
            f"Queue overload detected (rho={rho:.2f} >= 1). Using finite-horizon transient (p90) queue estimate over {hs:.2f}s."
        )

    if cal.prefill_r_squared < 0.8 and cal.prefill_ms_per_token > 0:
        warnings.append("Prefill calibration R² < 0.8; predictions may be less accurate")
    if cal.decode_r_squared < 0.9:
        warnings.append("Decode calibration R² < 0.9; predictions may be less accurate")

    out: Dict[str, Any] = {
        "predict_version": "2.1.3",
        "engine": cal.engine,
        "model": cal.model,
        "model_match": model_match,
        "workload_hash": workload_hash,
        "warnings": warnings,
        "inputs": {
            "prompt_tokens_p50": prompt_p50,
            "prompt_tokens_p90": prompt_p90,
            "max_output_tokens_p50": output_p50,
            "max_output_tokens_p90": output_p90,
            "num_requests": num_requests,
            "arrival_span_ms": span_ms,
            "offered_rps": offered_rps,
            "concurrency": concurrency,
            "target_utilization": target_utilization,
        },
        "calibration": {
            "prefill_fixed_overhead_ms": cal.prefill_fixed_overhead_ms,
            "prefill_ms_per_token": cal.prefill_ms_per_token,
            "decode_fixed_overhead_ms": cal.decode_fixed_overhead_ms,
            "decode_ms_per_token": cal.decode_ms_per_token,
            "request_overhead_ms": cal.request_overhead_ms,
            "decode_variance_cv": variance_cv,
            "prefill_r_squared": cal.prefill_r_squared,
            "decode_r_squared": cal.decode_r_squared,
            "health": cal.health,
        },
        "predicted": {
            # TTFT_PROXY (matches meas_ttftP)
            "ttft_p50_ms": ttft_p50,
            "ttft_p90_ms": ttft_p90,
            "ttft_p95_ms": ttft_p95,
            "ttft_p99_ms": ttft_p99,
            "ttft_p90_ci": list(ttft_ci),

            # Diagnostic TTFT including queue
            "ttft_e2e_p90_ms": float(ttft_e2e_p90),

            # Service (no queue)
            "service_time_p50_ms": svc_p50,
            "service_time_p90_ms": svc_p90,
            "service_time_p95_ms": svc_p95,
            "service_time_p99_ms": svc_p99,
            "service_time_p90_ci": list(svc_ci),

            # E2E (includes queue)
            "e2e_time_p50_ms": e2e_p50,
            "e2e_time_p90_ms": e2e_p90,
            "e2e_time_p95_ms": e2e_p95,
            "e2e_time_p99_ms": e2e_p99,
            "e2e_time_p90_ci": list(e2e_ci),

            "tail_cv_clamped": bool(tail_cv_clamped),
            "extrapolating_prompt_tokens": bool(extrap_prompt),
            "extrapolating_output_tokens": bool(extrap_out),
            "prefill_r_squared": float(cal.prefill_r_squared),
            "decode_r_squared": float(cal.decode_r_squared),
        },
        "breakdown": {
            "request_overhead_ms": breakdown.request_overhead_ms,
            "prefill_ms": breakdown.prefill_ms,
            "decode_ms": breakdown.decode_ms,
            "kv_cache_pressure_ms": breakdown.kv_cache_pressure_ms,
            "batch_overhead_ms": breakdown.batch_overhead_ms,
            "queueing_delay_ms": breakdown.queueing_delay_ms,
        },
        "throughput": {
            "max_throughput_rps": max_throughput,
            "sustainable_throughput_rps": sustainable_throughput,
            "service_time_ms": service_time_s * 1000.0,
        },
        "queueing": {
            "utilization": float(queue_metrics.get("utilization", 0.0) or 0.0),
            "queue_probability": float(queue_metrics.get("queue_probability", 0.0) or 0.0),
            "expected_queue_length": float(queue_metrics.get("expected_queue_length", 0.0) or 0.0),
            "rho": float(queue_metrics.get("rho", 0.0) or 0.0),
            "overloaded": bool(float(queue_metrics.get("overloaded", 0.0) or 0.0) > 0.5),
            "horizon_s": float(queue_metrics.get("horizon_s", 0.0) or 0.0),
            "overload_backlog_end": float(queue_metrics.get("overload_backlog_end", 0.0) or 0.0),
        },
        "sla_compliance": sla_compliance,
        "confidence": {
            "raw_decode_cv": float(unclamped),
            "used_decode_cv": float(variance_cv),
            "tail_cv_clamped": bool(tail_cv_clamped),
            "extrapolating_prompt_tokens": bool(extrap_prompt),
            "extrapolating_output_tokens": bool(extrap_out),
            "calibration_token_envelope": {
                "max_prompt_tokens": max_prompt_cal,
                "max_output_tokens": max_output_cal,
            },
        },
    }

    try:
        sla_ms = float(os.environ.get("IWC_SLA_MS", "300"))
    except Exception:
        sla_ms = 300.0

    dec = decide(
        predicted=out["predicted"],
        breakdown=out.get("breakdown", {}),
        throughput=out.get("throughput", {}),
        queueing=out.get("queueing", {}),
        calibration_health=cal.health,
        sla_ms=sla_ms,
        concurrency=int(concurrency),
        replicas=None,
    )
    out["decision"] = decision_to_dict(dec)

    return out
