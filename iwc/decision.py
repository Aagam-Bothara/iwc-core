# iwc/decision.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DecisionResult:
    verdict: str  # SAFE | BORDERLINE | RISKY
    confidence: str  # HIGH | MEDIUM | LOW
    sla_ms: float
    sla_breach_prob: float
    max_safe_concurrency_est: Optional[int]
    recommended_replicas: Optional[int]
    recommended_max_concurrency: Optional[int]
    reasons: List[str]
    notes: List[str]


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_tail_prob(z: float) -> float:
    return 1.0 - _normal_cdf(z)


def estimate_sla_breach_probability(
    *,
    pred_p90_ms: float,
    pred_p99_ms: Optional[float],
    sla_ms: float,
) -> float:
    """
    Turn predicted p90/p99 into a rough distribution proxy and estimate P(latency > SLA).

    Heuristic:
    - If p99 exists and is meaningfully larger than p90, infer a lognormal-ish sigma from ratio.
    - Else fallback to a conservative sigma.
    """
    if pred_p90_ms <= 0.0:
        return 1.0 if sla_ms <= 0.0 else 0.5

    if pred_p99_ms is not None and pred_p99_ms > pred_p90_ms * 1.01:
        z99 = 2.326
        z90 = 1.282
        sigma = math.log(pred_p99_ms / pred_p90_ms) / max(1e-6, (z99 - z90))
    else:
        sigma = 0.35

    z90 = 1.282
    median = pred_p90_ms / math.exp(sigma * z90)

    mu = math.log(max(1e-9, median))
    ln_sla = math.log(max(1e-9, sla_ms))
    z = (ln_sla - mu) / max(1e-9, sigma)
    return _clamp01(_normal_tail_prob(z))


def decide(
    *,
    predicted: Dict[str, Any],
    breakdown: Optional[Dict[str, Any]] = None,
    throughput: Optional[Dict[str, Any]] = None,
    queueing: Optional[Dict[str, Any]] = None,
    calibration_health: Optional[Dict[str, Any]] = None,
    sla_ms: float = 300.0,
    concurrency: int = 1,
    replicas: Optional[int] = None,
) -> DecisionResult:
    """
    Inputs are whatever predict_workload returns.

    This function makes a human-readable decision:
    - SAFE/BORDERLINE/RISKY based on P(breach) derived from p90/p99.
    - Confidence depends on calibration health + tail clamp + extrapolation + fit quality + utilization.
    """
    reasons: List[str] = []
    notes: List[str] = []

    e2e_p90 = float(predicted.get("e2e_time_p90_ms", 0.0) or 0.0)
    e2e_p99 = predicted.get("e2e_time_p99_ms", None)
    e2e_p99_f = float(e2e_p99) if isinstance(e2e_p99, (int, float)) else None

    util = None
    qprob = None
    if isinstance(queueing, dict):
        util = queueing.get("utilization", None)
        qprob = queueing.get("queue_probability", None)

    breach = estimate_sla_breach_probability(
        pred_p90_ms=e2e_p90,
        pred_p99_ms=e2e_p99_f,
        sla_ms=sla_ms,
    )

    # -------------------------
    # Confidence
    # -------------------------
    conf = "HIGH"

    # Calibration health gate
    if isinstance(calibration_health, dict):
        st = str(calibration_health.get("status", "")).lower()
        if st == "warn":
            conf = "MEDIUM"
            notes.append("Calibration health=warn (predictions usable but add safety margin).")
        elif st == "fail":
            conf = "LOW"
            notes.append("Calibration health=fail (treat as diagnostic only).")

    # Tail clamp / extrapolation / fit quality (expects predict.py to pass these keys)
    tail_clamped = bool(predicted.get("tail_cv_clamped", False))
    ex_prompt = bool(predicted.get("extrapolating_prompt_tokens", False))
    ex_out = bool(predicted.get("extrapolating_output_tokens", False))

    pref_r2 = predicted.get("prefill_r_squared", None)
    dec_r2 = predicted.get("decode_r_squared", None)

    if tail_clamped:
        if conf == "HIGH":
            conf = "MEDIUM"
        notes.append("Tail variance was clamped (p99 less trustworthy).")

    if ex_prompt or ex_out:
        if conf == "HIGH":
            conf = "MEDIUM"
        notes.append("Prediction extrapolates beyond calibration range.")

    if isinstance(pref_r2, (int, float)) and float(pref_r2) < 0.85:
        conf = "MEDIUM" if conf == "HIGH" else conf
        notes.append(f"Prefill fit weak (R²={float(pref_r2):.2f}).")
    if isinstance(dec_r2, (int, float)) and float(dec_r2) < 0.95:
        conf = "MEDIUM" if conf == "HIGH" else conf
        notes.append(f"Decode fit weak (R²={float(dec_r2):.2f}).")

    # Utilization warning: queueing dominates tails
    if isinstance(util, (int, float)) and util >= 0.9:
        conf = "MEDIUM" if conf == "HIGH" else conf
        notes.append("High utilization: queueing effects likely dominate tail latency.")

    # -------------------------
    # Verdict thresholds
    # -------------------------
    if breach <= 0.10:
        verdict = "SAFE"
    elif breach <= 0.35:
        verdict = "BORDERLINE"
    else:
        verdict = "RISKY"

    if e2e_p90 <= 0.0:
        verdict = "RISKY"
        reasons.append("Invalid predicted E2E p90 (0ms). Check measurement/eval pipeline.")
    else:
        if e2e_p90 > sla_ms:
            reasons.append(f"Predicted E2E p90 ({e2e_p90:.1f}ms) exceeds SLA ({sla_ms:.1f}ms).")
        else:
            reasons.append(f"Predicted E2E p90 ({e2e_p90:.1f}ms) vs SLA ({sla_ms:.1f}ms).")

    if isinstance(util, (int, float)):
        reasons.append(f"Estimated utilization: {float(util):.0%}.")
    if isinstance(qprob, (int, float)):
        reasons.append(f"Estimated queue probability: {float(qprob):.0%}.")

    # -------------------------
    # Recommendations
    # -------------------------
    if verdict == "SAFE":
        rec_max_conc = concurrency
    elif verdict == "BORDERLINE":
        rec_max_conc = max(1, int(math.floor(concurrency * 0.75)))
    else:
        rec_max_conc = max(1, int(math.floor(concurrency * 0.5)))

    rec_replicas: Optional[int] = None
    if replicas is not None:
        if verdict == "SAFE":
            rec_replicas = replicas
        elif verdict == "BORDERLINE":
            rec_replicas = replicas + 1
        else:
            rec_replicas = replicas + 2

    max_safe_conc_est: Optional[int] = None
    if isinstance(util, (int, float)) and util > 1e-6:
        target = 0.70
        max_safe_conc_est = max(1, int(math.floor(concurrency * (target / float(util)))))

    return DecisionResult(
        verdict=verdict,
        confidence=conf,
        sla_ms=float(sla_ms),
        sla_breach_prob=float(breach),
        max_safe_concurrency_est=max_safe_conc_est,
        recommended_replicas=rec_replicas,
        recommended_max_concurrency=rec_max_conc,
        reasons=reasons,
        notes=notes,
    )


def decision_to_dict(d: DecisionResult) -> Dict[str, Any]:
    return {
        "verdict": d.verdict,
        "confidence": d.confidence,
        "sla_ms": d.sla_ms,
        "sla_breach_probability": d.sla_breach_prob,
        "max_safe_concurrency_est": d.max_safe_concurrency_est,
        "recommended_replicas": d.recommended_replicas,
        "recommended_max_concurrency": d.recommended_max_concurrency,
        "reasons": d.reasons,
        "notes": d.notes,
    }
