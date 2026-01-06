# iwc/eval.py
"""
Enhanced Evaluation Module for LLM Inference Workload Predictor (v2.2)

Key fixes in v2.2 (fixes your "meas_svc=0, perfect MAPE" bug):
- Warmup exclusion is applied over SUCCESSFUL requests only (per repeat).
- Warmup exclusion is capped so it never deletes all samples.
- Very small workloads: warmup exclusion automatically disabled (by default threshold).
- Errors no longer get masked when measured==0:
  - _abs_pct_error/_signed_error return NaN for invalid measurements
  - summary stats (MAPE/RMSE/R²/Bias) skip NaNs
- Extra debug fields per experiment:
  - n_success_total, n_service_samples, warmup_excluded_success, samples_dropped_reason
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from iwc.profile import TargetProfile
from iwc.predict import load_calibration, predict_workload


# -----------------------------
# Statistical Utilities
# -----------------------------
def _now() -> float:
    return time.perf_counter()


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = max(0, min(len(s) - 1, int(p * (len(s) - 1) + 0.5)))
    return float(s[idx])


def _mean(xs: List[float]) -> float:
    xs2 = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(xs2) / len(xs2) if xs2 else float("nan")


def _std(xs: List[float]) -> float:
    xs2 = [x for x in xs if x is not None and math.isfinite(x)]
    if len(xs2) < 2:
        return float("nan")
    return statistics.stdev(xs2)


def _cv(xs: List[float]) -> float:
    xs2 = [x for x in xs if x is not None and math.isfinite(x)]
    if len(xs2) < 2:
        return float("nan")
    m = _mean(xs2)
    sd = _std(xs2)
    return (sd / m) if (math.isfinite(m) and m > 0 and math.isfinite(sd)) else float("nan")


def _bootstrap_ci(
    vals: List[float],
    stat_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    vals2 = [v for v in vals if v is not None and math.isfinite(v)]
    if not vals2:
        return float("nan"), float("nan"), float("nan")

    point_est = stat_fn(vals2)
    if len(vals2) < 3:
        return point_est, point_est, point_est

    boot_stats = []
    for _ in range(n_bootstrap):
        sample = [random.choice(vals2) for _ in range(len(vals2))]
        boot_stats.append(stat_fn(sample))

    boot_stats = [b for b in boot_stats if math.isfinite(b)]
    if not boot_stats:
        return point_est, point_est, point_est

    boot_stats.sort()
    alpha = 1.0 - confidence
    lo = boot_stats[int(alpha / 2 * len(boot_stats))]
    hi = boot_stats[int((1 - alpha / 2) * len(boot_stats))]
    return point_est, lo, hi


def _abs_pct_error(pred: float, meas: float) -> float:
    # IMPORTANT: never mask invalid measurements as 0% error
    if meas is None or not math.isfinite(meas) or meas <= 1e-9:
        return float("nan")
    if pred is None or not math.isfinite(pred):
        return float("nan")
    return abs(pred - meas) / meas


def _signed_error(pred: float, meas: float) -> float:
    if meas is None or not math.isfinite(meas) or meas <= 1e-9:
        return float("nan")
    if pred is None or not math.isfinite(pred):
        return float("nan")
    return (pred - meas) / meas


def _mape(preds: List[float], meas: List[float]) -> float:
    if not preds or len(preds) != len(meas):
        return float("nan")
    errs = []
    for p, m in zip(preds, meas):
        e = _abs_pct_error(p, m)
        if math.isfinite(e):
            errs.append(e)
    return _mean(errs)


def _rmse(preds: List[float], meas: List[float]) -> float:
    if not preds or len(preds) != len(meas):
        return float("nan")
    diffs = []
    for p, m in zip(preds, meas):
        if p is None or m is None:
            continue
        if not (math.isfinite(p) and math.isfinite(m)):
            continue
        diffs.append((p - m) ** 2)
    return math.sqrt(_mean(diffs)) if diffs else float("nan")


def _r_squared(preds: List[float], meas: List[float]) -> float:
    if not preds or len(preds) != len(meas):
        return float("nan")

    pairs = [(p, m) for p, m in zip(preds, meas) if math.isfinite(p) and math.isfinite(m)]
    if len(pairs) < 2:
        return float("nan")

    meas2 = [m for _, m in pairs]
    preds2 = [p for p, _ in pairs]

    mbar = _mean(meas2)
    if not math.isfinite(mbar):
        return float("nan")

    ss_tot = sum((m - mbar) ** 2 for m in meas2)
    ss_res = sum((m - p) ** 2 for (p, m) in pairs)
    if ss_tot <= 1e-12:
        return float("nan")
    return max(0.0, 1.0 - ss_res / ss_tot)


def _log_normal_test(xs: List[float]) -> Dict[str, Any]:
    xs2 = [x for x in xs if x is not None and math.isfinite(x)]
    if len(xs2) < 10:
        return {"valid": False, "reason": "insufficient data"}

    pos = [x for x in xs2 if x > 0]
    if len(pos) < 10:
        return {"valid": False, "reason": "insufficient positive values"}

    log_xs = [math.log(x) for x in pos]
    n = len(log_xs)
    mu = _mean(log_xs)
    sd = _std(log_xs)
    if not math.isfinite(sd) or sd <= 1e-9:
        return {"valid": False, "reason": "zero variance"}

    m3 = sum((x - mu) ** 3 for x in log_xs) / n
    m4 = sum((x - mu) ** 4 for x in log_xs) / n
    skew = m3 / (sd**3)
    kurt = m4 / (sd**4) - 3

    is_normal = abs(skew) < 1.0 and abs(kurt) < 2.0
    return {
        "valid": True,
        "is_log_normal": is_normal,
        "log_mean": mu,
        "log_std": sd,
        "skewness": skew,
        "kurtosis": kurt,
        "original_cv": _cv(pos),
    }


# -----------------------------
# HTTP Client
# -----------------------------
def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(obj, dict):
                raise SystemExit(f"{path}:{line_no}: expected JSON object")
            out.append(obj)
    return out


def _post_openai_nonstream(
    *,
    host: str,
    api_key: Optional[str],
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: int = 120,
) -> Dict[str, Any]:
    url = f"http://{host}/v1/completions"
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            pass
        raise urllib.error.HTTPError(
            e.url, e.code, f"{e.msg} | body={err_body[:800]}", e.hdrs, e.fp
        ) from None

    return json.loads(body)


def _extract_prompt(req: Dict[str, Any]) -> str:
    p = req.get("prompt")
    if isinstance(p, str):
        return p
    return json.dumps(p, ensure_ascii=False)


def _extract_max_output_tokens(req: Dict[str, Any]) -> int:
    v = req.get("max_output_tokens")
    if isinstance(v, int) and v > 0:
        return v
    return 16


def _extract_arrival_time_ms(req: Dict[str, Any]) -> float:
    v = req.get("arrival_time_ms", 0)
    if isinstance(v, (int, float)) and v >= 0:
        return float(v)
    return 0.0


# -----------------------------
# Evaluation Configuration
# -----------------------------
@dataclass
class EvalConfig:
    host: str
    api_key: Optional[str]
    concurrency: int
    ttft_sample_n: int
    timeout_s: int
    debug_sched: bool = False

    # warmup handling
    warmup_requests: int = 3
    min_success_for_warmup: int = 10  # if fewer SUCCESS samples than this, warmup exclusion disabled

    bootstrap_samples: int = 1000
    confidence_level: float = 0.95


@dataclass
class RequestResult:
    idx: int
    arrival_s: float
    send_start_s: float
    done_s: float
    qdelay_ms: float
    service_ms: float
    e2e_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class RepeatResult:
    repeat_idx: int
    request_results: List[RequestResult]
    ttft_proxy_samples_ms: List[float]
    warmup_excluded_success: int = 0  # how many SUCCESSFUL requests excluded
    duration_s: float = 0.0

    def _success_in_order(self) -> List[RequestResult]:
        return [r for r in self.request_results if r.success]

    def _exclude_warmup_success(self, rs_success: List[RequestResult]) -> List[RequestResult]:
        k = max(0, min(self.warmup_excluded_success, len(rs_success)))
        return rs_success[k:]

    @property
    def qdelay_ms_list(self) -> List[float]:
        rs = self._exclude_warmup_success(self._success_in_order())
        return [r.qdelay_ms for r in rs]

    @property
    def service_ms_list(self) -> List[float]:
        rs = self._exclude_warmup_success(self._success_in_order())
        return [r.service_ms for r in rs]

    @property
    def e2e_ms_list(self) -> List[float]:
        rs = self._exclude_warmup_success(self._success_in_order())
        return [r.e2e_ms for r in rs]


# -----------------------------
# Measurement Functions
# -----------------------------
def _build_arrival_schedule(requests: List[Dict[str, Any]]) -> List[Tuple[float, int]]:
    sched = []
    for i, r in enumerate(requests):
        a_ms = _extract_arrival_time_ms(r)
        sched.append((a_ms / 1000.0, i))
    sched.sort(key=lambda x: (x[0], x[1]))
    return sched


def _measure_one_repeat(
    *,
    requests: List[Dict[str, Any]],
    profile: TargetProfile,
    cfg: EvalConfig,
    repeat_idx: int,
) -> RepeatResult:
    sched = _build_arrival_schedule(requests)

    if cfg.debug_sched and repeat_idx == 0:
        print(f"SCHED first 3: {sched[:3]}")
        print(f"SCHED last 3: {sched[-3:]}")

    # TTFT proxy sample indices: random across full workload
    ttft_n = min(cfg.ttft_sample_n, len(sched))
    all_indices = [idx for (_, idx) in sched]
    random.shuffle(all_indices)
    ttft_indices = set(all_indices[:ttft_n])

    sem = threading.Semaphore(max(1, cfg.concurrency))

    t_start = _now()
    results: List[RequestResult] = []
    lock = threading.Lock()

    def do_completion(idx: int, arrival_s: float) -> RequestResult:
        # wait until arrival
        while True:
            now_s = _now() - t_start
            dt = arrival_s - now_s
            if dt <= 0:
                break
            time.sleep(min(0.005, dt))

        arrival_abs = t_start + arrival_s

        sem.acquire()
        send_start_abs = _now()
        try:
            r = requests[idx]
            prompt = _extract_prompt(r)
            mt = _extract_max_output_tokens(r)

            resp = _post_openai_nonstream(
                host=cfg.host,
                api_key=cfg.api_key,
                model=profile.model,
                prompt=prompt,
                max_tokens=mt,
                timeout_s=cfg.timeout_s,
            )
            done_abs = _now()

            usage = resp.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")

            qdelay_ms = max(0.0, (send_start_abs - arrival_abs) * 1000.0)
            service_ms = max(0.0, (done_abs - send_start_abs) * 1000.0)
            e2e_ms = max(0.0, (done_abs - arrival_abs) * 1000.0)

            return RequestResult(
                idx=idx,
                arrival_s=arrival_s,
                send_start_s=send_start_abs - t_start,
                done_s=done_abs - t_start,
                qdelay_ms=qdelay_ms,
                service_ms=service_ms,
                e2e_ms=e2e_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                success=True,
            )
        except Exception as e:
            done_abs = _now()
            return RequestResult(
                idx=idx,
                arrival_s=arrival_s,
                send_start_s=send_start_abs - t_start,
                done_s=done_abs - t_start,
                qdelay_ms=0.0,
                service_ms=0.0,
                e2e_ms=0.0,
                success=False,
                error=str(e)[:200],
            )
        finally:
            sem.release()

    def do_ttft_proxy(idx: int) -> float:
        """TTFT proxy = wall time for a non-stream request with max_tokens=1."""
        r = requests[idx]
        prompt = _extract_prompt(r)
        t0 = _now()
        try:
            _ = _post_openai_nonstream(
                host=cfg.host,
                api_key=cfg.api_key,
                model=profile.model,
                prompt=prompt,
                max_tokens=1,
                timeout_s=cfg.timeout_s,
            )
            return (_now() - t0) * 1000.0
        except Exception:
            return -1.0

    # execute completions
    with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as ex:
        futs = {ex.submit(do_completion, idx, arr_s): (idx, arr_s) for (arr_s, idx) in sched}
        for fut in as_completed(futs):
            res = fut.result()
            with lock:
                results.append(res)

    duration_s = _now() - t_start
    results.sort(key=lambda r: r.idx)

    # warmup exclusion: per repeat, but ONLY over successful samples
    success_n = sum(1 for r in results if r.success)

    if success_n < cfg.min_success_for_warmup:
        warmup_excluded_success = 0
    else:
        # cap so you never delete all SUCCESS samples
        warmup_excluded_success = min(cfg.warmup_requests, max(0, success_n - 1))

    # ttft proxy measurements (separate requests)
    ttft_samples = []
    with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as ex:
        futs = [ex.submit(do_ttft_proxy, i) for i in ttft_indices]
        for fut in as_completed(futs):
            v = fut.result()
            if v > 0:
                ttft_samples.append(v)

    return RepeatResult(
        repeat_idx=repeat_idx,
        request_results=results,
        ttft_proxy_samples_ms=ttft_samples,
        warmup_excluded_success=warmup_excluded_success,
        duration_s=duration_s,
    )


# -----------------------------
# Result Aggregation
# -----------------------------
@dataclass
class AggregatedMetrics:
    # TTFT proxy
    ttft_proxy_p50: float = float("nan")
    ttft_proxy_p90: float = float("nan")
    ttft_proxy_p95: float = float("nan")
    ttft_proxy_p99: float = float("nan")
    ttft_proxy_p90_ci: Tuple[float, float] = (float("nan"), float("nan"))

    # Service time (HTTP call duration)
    service_p50: float = float("nan")
    service_p90: float = float("nan")
    service_p95: float = float("nan")
    service_p99: float = float("nan")
    service_p90_ci: Tuple[float, float] = (float("nan"), float("nan"))

    # Queue delay (arrival->send_start)
    qdelay_p50: float = float("nan")
    qdelay_p90: float = float("nan")
    qdelay_p95: float = float("nan")
    qdelay_p99: float = float("nan")

    # End-to-end (arrival->done)
    e2e_p50: float = float("nan")
    e2e_p90: float = float("nan")
    e2e_p95: float = float("nan")
    e2e_p99: float = float("nan")
    e2e_p90_ci: Tuple[float, float] = (float("nan"), float("nan"))

    # Distribution stats
    service_cv: float = float("nan")
    distribution_fit: Dict[str, Any] = field(default_factory=dict)

    # Throughput
    actual_throughput_rps: float = float("nan")
    success_rate: float = float("nan")

    # Sample sizes / debug
    n_requests: int = 0
    n_success_total: int = 0
    n_ttft_proxy_samples: int = 0
    n_repeats: int = 0
    n_service_samples: int = 0
    n_e2e_samples: int = 0
    n_qdelay_samples: int = 0
    warmup_excluded_success_total: int = 0
    samples_dropped_reason: str = ""


def _aggregate_repeats(repeats: List[RepeatResult], cfg: EvalConfig) -> AggregatedMetrics:
    all_ttft_proxy: List[float] = []
    all_service: List[float] = []
    all_qdelay: List[float] = []
    all_e2e: List[float] = []

    total_requests = 0
    successful_requests = 0
    total_duration = 0.0
    warmup_excl_total = 0

    for rep in repeats:
        all_ttft_proxy.extend(rep.ttft_proxy_samples_ms)
        all_service.extend(rep.service_ms_list)
        all_qdelay.extend(rep.qdelay_ms_list)
        all_e2e.extend(rep.e2e_ms_list)

        total_requests += len(rep.request_results)
        successful_requests += sum(1 for r in rep.request_results if r.success)
        total_duration += rep.duration_s
        warmup_excl_total += rep.warmup_excluded_success

    m = AggregatedMetrics(
        n_requests=total_requests,
        n_success_total=successful_requests,
        n_ttft_proxy_samples=len(all_ttft_proxy),
        n_repeats=len(repeats),
        success_rate=successful_requests / max(1, total_requests),
        n_service_samples=len(all_service),
        n_qdelay_samples=len(all_qdelay),
        n_e2e_samples=len(all_e2e),
        warmup_excluded_success_total=warmup_excl_total,
        samples_dropped_reason=(
            "warmup disabled (too few successes)"
            if any(
                (sum(1 for r in rep.request_results if r.success) < cfg.min_success_for_warmup)
                for rep in repeats
            )
            else ""
        ),
    )

    if all_ttft_proxy:
        m.ttft_proxy_p50 = _percentile(all_ttft_proxy, 0.5)
        m.ttft_proxy_p90 = _percentile(all_ttft_proxy, 0.9)
        m.ttft_proxy_p95 = _percentile(all_ttft_proxy, 0.95)
        m.ttft_proxy_p99 = _percentile(all_ttft_proxy, 0.99)
        _, lo, hi = _bootstrap_ci(
            all_ttft_proxy,
            lambda x: _percentile(x, 0.9),
            n_bootstrap=cfg.bootstrap_samples,
            confidence=cfg.confidence_level,
        )
        m.ttft_proxy_p90_ci = (lo, hi)

    if all_service:
        m.service_p50 = _percentile(all_service, 0.5)
        m.service_p90 = _percentile(all_service, 0.9)
        m.service_p95 = _percentile(all_service, 0.95)
        m.service_p99 = _percentile(all_service, 0.99)
        _, lo, hi = _bootstrap_ci(
            all_service,
            lambda x: _percentile(x, 0.9),
            n_bootstrap=cfg.bootstrap_samples,
            confidence=cfg.confidence_level,
        )
        m.service_p90_ci = (lo, hi)
        m.service_cv = _cv(all_service)
        m.distribution_fit = _log_normal_test(all_service)

    if all_qdelay:
        m.qdelay_p50 = _percentile(all_qdelay, 0.5)
        m.qdelay_p90 = _percentile(all_qdelay, 0.9)
        m.qdelay_p95 = _percentile(all_qdelay, 0.95)
        m.qdelay_p99 = _percentile(all_qdelay, 0.99)

    if all_e2e:
        m.e2e_p50 = _percentile(all_e2e, 0.5)
        m.e2e_p90 = _percentile(all_e2e, 0.9)
        m.e2e_p95 = _percentile(all_e2e, 0.95)
        m.e2e_p99 = _percentile(all_e2e, 0.99)
        _, lo, hi = _bootstrap_ci(
            all_e2e,
            lambda x: _percentile(x, 0.9),
            n_bootstrap=cfg.bootstrap_samples,
            confidence=cfg.confidence_level,
        )
        m.e2e_p90_ci = (lo, hi)

    if total_duration > 0:
        m.actual_throughput_rps = successful_requests / total_duration

    return m


# -----------------------------
# Main Evaluation
# -----------------------------
def eval_workloads(
    *,
    inputs: List[Path],
    profile: TargetProfile,
    cal_path: Path,
    host: str,
    api_key: Optional[str],
    concurrency_list: List[int],
    repeats: int = 3,
    ttft_sample_n: int = 10,
    timeout_s: int = 120,
    debug_sched: bool = False,
    warmup_requests: int = 3,
    bootstrap_samples: int = 1000,
) -> Dict[str, Any]:
    cal = load_calibration(cal_path)

    out: Dict[str, Any] = {
        "eval_version": "2.2",
        "engine": getattr(cal, "engine", "unknown"),
        "model": getattr(cal, "model", profile.model),
        "host": host,
        "inputs": [str(p) for p in inputs],
        "concurrency_list": concurrency_list,
        "repeats": repeats,
        "bootstrap_samples": bootstrap_samples,
        "warmup_requests": warmup_requests,
        "results": [],
        "summary": {},
    }

    # accuracy trackers
    all_pred_service: List[float] = []
    all_meas_service: List[float] = []
    all_pred_e2e: List[float] = []
    all_meas_e2e: List[float] = []
    all_pred_ttft: List[float] = []
    all_meas_ttft_proxy: List[float] = []

    for inp in inputs:
        reqs = _read_jsonl(inp)

        for c in concurrency_list:
            pred = predict_workload(
                workload_jsonl=inp,
                profile=profile,
                cal=cal,
                concurrency=c,
            )

            pred_ttft_p90 = float(pred["predicted"]["ttft_p90_ms"])
            pred_service_p90 = float(pred["predicted"]["service_time_p90_ms"])
            pred_e2e_p90 = float(pred["predicted"]["e2e_time_p90_ms"])

            cfg = EvalConfig(
                host=host,
                api_key=api_key,
                concurrency=c,
                ttft_sample_n=ttft_sample_n,
                timeout_s=timeout_s,
                debug_sched=debug_sched,
                warmup_requests=warmup_requests,
                bootstrap_samples=bootstrap_samples,
            )

            reps: List[RepeatResult] = []
            for rep in range(repeats):
                rr = _measure_one_repeat(
                    requests=reqs,
                    profile=profile,
                    cfg=cfg,
                    repeat_idx=rep,
                )
                reps.append(rr)

            metrics = _aggregate_repeats(reps, cfg)

            # errors
            service_abs = _abs_pct_error(pred_service_p90, metrics.service_p90)
            service_signed = _signed_error(pred_service_p90, metrics.service_p90)
            e2e_abs = _abs_pct_error(pred_e2e_p90, metrics.e2e_p90)
            e2e_signed = _signed_error(pred_e2e_p90, metrics.e2e_p90)

            # TTFT proxy comparison (note: proxy)
            ttft_abs = _abs_pct_error(pred_ttft_p90, metrics.ttft_proxy_p90)
            ttft_signed = _signed_error(pred_ttft_p90, metrics.ttft_proxy_p90)

            # CI inclusion (measured CI)
            service_in_ci = (
                math.isfinite(metrics.service_p90_ci[0])
                and metrics.service_p90_ci[0] <= pred_service_p90 <= metrics.service_p90_ci[1]
            )
            e2e_in_ci = (
                math.isfinite(metrics.e2e_p90_ci[0])
                and metrics.e2e_p90_ci[0] <= pred_e2e_p90 <= metrics.e2e_p90_ci[1]
            )
            ttft_in_ci = (
                math.isfinite(metrics.ttft_proxy_p90_ci[0])
                and metrics.ttft_proxy_p90_ci[0] <= pred_ttft_p90 <= metrics.ttft_proxy_p90_ci[1]
            )

            # track summary only if measurement is valid (non-NaN)
            if math.isfinite(metrics.service_p90):
                all_pred_service.append(pred_service_p90)
                all_meas_service.append(metrics.service_p90)
            if math.isfinite(metrics.e2e_p90):
                all_pred_e2e.append(pred_e2e_p90)
                all_meas_e2e.append(metrics.e2e_p90)
            if math.isfinite(metrics.ttft_proxy_p90):
                all_pred_ttft.append(pred_ttft_p90)
                all_meas_ttft_proxy.append(metrics.ttft_proxy_p90)

            out["results"].append(
                {
                    "workload": str(inp),
                    "concurrency": c,
                    "n_requests": len(reqs),
                    "predicted": {
                        "ttft_p90_ms": pred_ttft_p90,
                        "service_time_p90_ms": pred_service_p90,
                        "e2e_time_p90_ms": pred_e2e_p90,
                    },
                    "measured": {
                        "ttft_proxy_p50_ms": metrics.ttft_proxy_p50,
                        "ttft_proxy_p90_ms": metrics.ttft_proxy_p90,
                        "ttft_proxy_p95_ms": metrics.ttft_proxy_p95,
                        "ttft_proxy_p99_ms": metrics.ttft_proxy_p99,
                        "ttft_proxy_p90_ci": list(metrics.ttft_proxy_p90_ci),

                        "service_p50_ms": metrics.service_p50,
                        "service_p90_ms": metrics.service_p90,
                        "service_p95_ms": metrics.service_p95,
                        "service_p99_ms": metrics.service_p99,
                        "service_p90_ci": list(metrics.service_p90_ci),

                        "qdelay_p50_ms": metrics.qdelay_p50,
                        "qdelay_p90_ms": metrics.qdelay_p90,
                        "qdelay_p95_ms": metrics.qdelay_p95,
                        "qdelay_p99_ms": metrics.qdelay_p99,

                        "e2e_p50_ms": metrics.e2e_p50,
                        "e2e_p90_ms": metrics.e2e_p90,
                        "e2e_p95_ms": metrics.e2e_p95,
                        "e2e_p99_ms": metrics.e2e_p99,
                        "e2e_p90_ci": list(metrics.e2e_p90_ci),

                        "service_cv": metrics.service_cv,
                        "distribution_fit": metrics.distribution_fit,
                        "actual_throughput_rps": metrics.actual_throughput_rps,
                        "success_rate": metrics.success_rate,
                        "n_ttft_proxy_samples": metrics.n_ttft_proxy_samples,

                        # debug
                        "n_success_total": metrics.n_success_total,
                        "n_service_samples": metrics.n_service_samples,
                        "n_e2e_samples": metrics.n_e2e_samples,
                        "n_qdelay_samples": metrics.n_qdelay_samples,
                        "warmup_excluded_success_total": metrics.warmup_excluded_success_total,
                        "samples_dropped_reason": metrics.samples_dropped_reason,
                    },
                    "error": {
                        "ttft_proxy_abs_pct": ttft_abs,
                        "ttft_proxy_signed_pct": ttft_signed,
                        "ttft_proxy_in_ci": ttft_in_ci,

                        "service_abs_pct": service_abs,
                        "service_signed_pct": service_signed,
                        "service_in_ci": service_in_ci,

                        "e2e_abs_pct": e2e_abs,
                        "e2e_signed_pct": e2e_signed,
                        "e2e_in_ci": e2e_in_ci,
                    },
                }
            )

    # summary (skips invalid points by construction)
    out["summary"] = {
        "service": {
            "mape": _mape(all_pred_service, all_meas_service),
            "rmse": _rmse(all_pred_service, all_meas_service),
            "r_squared": _r_squared(all_pred_service, all_meas_service),
            "mean_signed_error": _mean([_signed_error(p, m) for p, m in zip(all_pred_service, all_meas_service)]),
            "n_points": len(all_pred_service),
        },
        "e2e": {
            "mape": _mape(all_pred_e2e, all_meas_e2e),
            "rmse": _rmse(all_pred_e2e, all_meas_e2e),
            "r_squared": _r_squared(all_pred_e2e, all_meas_e2e),
            "mean_signed_error": _mean([_signed_error(p, m) for p, m in zip(all_pred_e2e, all_meas_e2e)]),
            "n_points": len(all_pred_e2e),
        },
        "ttft_proxy": {
            "mape": _mape(all_pred_ttft, all_meas_ttft_proxy),
            "rmse": _rmse(all_pred_ttft, all_meas_ttft_proxy),
            "r_squared": _r_squared(all_pred_ttft, all_meas_ttft_proxy),
            "mean_signed_error": _mean([_signed_error(p, m) for p, m in zip(all_pred_ttft, all_meas_ttft_proxy)]),
            "n_points": len(all_pred_ttft),
        },
        "n_experiments_total": len(out["results"]),
    }

    return out


def _fmt_ms(x: float) -> str:
    if x is None or not math.isfinite(x):
        return "NA"
    return f"{x:.1f}"


def _fmt_pct(x: float) -> str:
    if x is None or not math.isfinite(x):
        return "NA"
    return f"{x:.1%}"


def format_eval_text(obj: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"Model: {obj.get('model')}")
    lines.append(f"Host: {obj.get('host')}")
    lines.append(f"Repeats: {obj.get('repeats')}")
    lines.append(f"Warmup excluded per repeat (success-only, capped): {obj.get('warmup_requests')}")
    lines.append("")

    summary = obj.get("summary", {})
    if summary:
        lines.append("=== SUMMARY ===")
        for metric in ["service", "e2e", "ttft_proxy"]:
            s = summary.get(metric, {})
            mape = s.get("mape", float("nan"))
            rmse = s.get("rmse", float("nan"))
            r2 = s.get("r_squared", float("nan"))
            bias = s.get("mean_signed_error", float("nan"))
            npts = s.get("n_points", 0)
            lines.append(
                f"{metric.upper():9s}: "
                f"MAPE={_fmt_pct(mape)}, "
                f"RMSE={('NA' if not math.isfinite(rmse) else f'{rmse:.2f}ms')}, "
                f"R²={('NA' if not math.isfinite(r2) else f'{r2:.3f}')}, "
                f"Bias={('NA' if not math.isfinite(bias) else f'{bias:+.1%}')}, "
                f"n={npts}"
            )
        lines.append("")

    header = (
        "workload | conc | pred_ttft | meas_ttftP | pred_svc | meas_svc | pred_e2e | meas_e2e | "
        "qdelay_p90 | n_succ | n_svc_samp | warmup_excl | err_svc | err_e2e | in_ci(s/e)"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in obj.get("results", []):
        w = Path(r["workload"]).name
        c = int(r["concurrency"])

        ptt = float(r["predicted"]["ttft_p90_ms"])
        mtt = float(r["measured"]["ttft_proxy_p90_ms"])

        ps = float(r["predicted"]["service_time_p90_ms"])
        ms = float(r["measured"]["service_p90_ms"])

        pe = float(r["predicted"]["e2e_time_p90_ms"])
        me = float(r["measured"]["e2e_p90_ms"])

        qd = float(r["measured"]["qdelay_p90_ms"])

        n_succ = int(r["measured"].get("n_success_total", 0))
        n_svc_samp = int(r["measured"].get("n_service_samples", 0))
        warm_ex = int(r["measured"].get("warmup_excluded_success_total", 0))

        es = float(r["error"]["service_abs_pct"])
        ee = float(r["error"]["e2e_abs_pct"])

        in_ci = "✓" if (r["error"]["service_in_ci"] and r["error"]["e2e_in_ci"]) else "✗"

        lines.append(
            f"{w[:20]:20} | {c:>4} | {ptt:>9.1f} | {_fmt_ms(mtt):>9} | "
            f"{ps:>8.1f} | {_fmt_ms(ms):>8} | {pe:>8.1f} | {_fmt_ms(me):>8} | "
            f"{_fmt_ms(qd):>10} | {n_succ:>6} | {n_svc_samp:>10} | {warm_ex:>11} | "
            f"{_fmt_pct(es):>7} | {_fmt_pct(ee):>7} | {in_ci}"
        )

    return "\n".join(lines) + "\n"
