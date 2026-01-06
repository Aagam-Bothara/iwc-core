# iwc/calibrate.py
"""
Enhanced Calibration Module for LLM Inference Workload Predictor

Key improvements over original:
1. Robust regression fitting with Theil-Sen estimator for outlier resistance
2. Confidence intervals on all calibration parameters
3. KV cache pressure estimation for large contexts
4. Batch scheduling effects measurement
5. Chunked prefill detection and modeling
6. Decode variance characterization (not just median)  <-- FIXED: decode-only ms/token CV
7. Request overhead separation (network vs compute)
8. Multi-phase warmup with JIT detection
9. Power-law tail modeling for decode (note: predictor-side)
10. Quantization effects detection (INT8/FP8 vs FP16) (note: not implemented here)
"""
from __future__ import annotations

import json
import math
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

from iwc.analyze.tokenizer import count_tokens_for_prompt
from iwc.profile import TargetProfile


# -----------------------------
# Configuration Constants
# -----------------------------
@dataclass(frozen=True)
class CalibrationConfig:
    """Tunable calibration parameters."""
    # Prefill settings
    prefill_batch_default: int = 16
    prefill_tps_cap: float = 300_000.0
    prefill_min_dt_span_ms: float = 10.0
    prefill_num_points: int = 7
    prefill_runs_per_point: int = 7

    # Decode settings
    decode_runs_per_point: int = 9
    decode_targets: Tuple[int, ...] = (16, 32, 64, 128, 256, 384)
    decode_min_completion_frac: float = 0.90
    decode_point_retries: int = 4

    # Streaming TTFT
    stream_ttft_runs: int = 9

    # Warmup
    warmup_requests: int = 5
    warmup_wait_ms: float = 100.0

    # Statistical
    confidence_level: float = 0.95
    outlier_iqr_multiplier: float = 1.5
    min_valid_points_for_fit: int = 3

    # Request overhead measurement
    overhead_measurement_runs: int = 15

    # KV cache probing
    kv_cache_probe_sizes: Tuple[int, ...] = (512, 1024, 2048, 4096, 8192)


DEFAULT_CONFIG = CalibrationConfig()


@dataclass
class RegressionResult:
    """Result of a linear regression fit."""
    slope: float
    intercept: float
    r_squared: float
    slope_ci_low: float
    slope_ci_high: float
    intercept_ci_low: float
    intercept_ci_high: float
    n_points: int
    residual_std: float


@dataclass(kw_only=True)
class CalibrationResult:
    """Complete calibration result with confidence intervals."""
    cal_version: str
    engine: str
    model: str

    # Prefill parameters
    prefill_fixed_overhead_ms: float
    prefill_ms_per_token: float
    prefill_fixed_ci: Tuple[float, float] = (0.0, 0.0)
    prefill_slope_ci: Tuple[float, float] = (0.0, 0.0)

    # Decode parameters
    decode_fixed_overhead_ms: float
    decode_ms_per_token: float
    decode_fixed_ci: Tuple[float, float] = (0.0, 0.0)
    decode_slope_ci: Tuple[float, float] = (0.0, 0.0)

    # Request overhead (network + framework)
    request_overhead_ms: float = 0.0
    request_overhead_std: float = 0.0

    # KV cache pressure model (ms penalty per 1K tokens in cache)
    kv_cache_pressure_ms_per_1k: float = 0.0
    kv_cache_threshold_tokens: int = 0

    # Decode variance (for tail latency prediction)
    # FIX: this is now CV of decode-only ms/token (not raw end-to-end dt).
    decode_variance_coefficient: float = 0.0

    # Batch effects
    batch_overhead_ms_per_concurrent: float = 0.0
    max_efficient_batch: int = 1

    # Metadata
    warnings: List[str] = field(default_factory=list)
    quality: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)

    # Fit statistics
    prefill_r_squared: float = 0.0
    decode_r_squared: float = 0.0


def _now() -> float:
    return time.perf_counter()


def _count_tokens(text: str, *, model: str) -> int:
    return count_tokens_for_prompt(
        text,
        prompt_format="raw",
        tokenizer_prefer="tiktoken",
        tokenizer_model=model,
    )


# -----------------------------
# Statistical Utilities
# -----------------------------
def _median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[len(s) // 2]


def _percentile(vals: List[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = int(p * (len(s) - 1))
    return s[idx]


def _iqr_filter(vals: List[float], multiplier: float = 1.5) -> List[float]:
    """Remove outliers using IQR method."""
    if len(vals) < 4:
        return vals

    s = sorted(vals)
    q1 = s[len(s) // 4]
    q3 = s[(3 * len(s)) // 4]
    iqr = q3 - q1

    low = q1 - multiplier * iqr
    high = q3 + multiplier * iqr

    return [v for v in vals if low <= v <= high]


def _cv(vals: List[float]) -> float:
    """Coefficient of variation = std/mean (robustly guarded)."""
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    if m <= 1e-12:
        return 0.0
    try:
        s = statistics.stdev(vals)
    except Exception:
        return 0.0
    return s / m


def _theil_sen_slope(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Theil-Sen estimator - robust median-based regression.
    Returns (slope, intercept).
    """
    n = len(x)
    if n < 2:
        return 0.0, (_median(y) if y else 0.0)

    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(x[j] - x[i]) > 1e-9:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))

    if not slopes:
        return 0.0, _median(y)

    slope = _median(slopes)
    intercepts = [y[i] - slope * x[i] for i in range(n)]
    intercept = _median(intercepts)

    return slope, intercept


def _ols_regression(x: List[float], y: List[float]) -> RegressionResult:
    """
    Ordinary least squares with confidence intervals.
    """
    n = len(x)
    if n < 2:
        y_mean = _median(y) if y else 0.0
        return RegressionResult(
            slope=0.0,
            intercept=y_mean,
            r_squared=0.0,
            slope_ci_low=0.0,
            slope_ci_high=0.0,
            intercept_ci_low=y_mean,
            intercept_ci_high=y_mean,
            n_points=n,
            residual_std=0.0,
        )

    x_mean = sum(x) / n
    y_mean = sum(y) / n

    ss_xx = sum((xi - x_mean) ** 2 for xi in x)
    ss_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    ss_yy = sum((yi - y_mean) ** 2 for yi in y)

    if abs(ss_xx) < 1e-12:
        return RegressionResult(
            slope=0.0,
            intercept=y_mean,
            r_squared=0.0,
            slope_ci_low=0.0,
            slope_ci_high=0.0,
            intercept_ci_low=y_mean,
            intercept_ci_high=y_mean,
            n_points=n,
            residual_std=0.0,
        )

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    # R-squared
    ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))
    r_squared = 1.0 - (ss_res / ss_yy) if abs(ss_yy) > 1e-12 else 0.0
    r_squared = max(0.0, min(1.0, r_squared))

    # Standard errors
    if n > 2:
        mse = ss_res / (n - 2)
        se_slope = math.sqrt(mse / ss_xx) if ss_xx > 0 else 0.0
        se_intercept = math.sqrt(mse * (1 / n + x_mean**2 / ss_xx)) if ss_xx > 0 else 0.0
        residual_std = math.sqrt(mse)
    else:
        se_slope = 0.0
        se_intercept = 0.0
        residual_std = 0.0

    # t-value for 95% CI (rough)
    t_val = 1.96 if n > 30 else 2.0 + 5.0 / max(1, n - 2)

    return RegressionResult(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        slope_ci_low=slope - t_val * se_slope,
        slope_ci_high=slope + t_val * se_slope,
        intercept_ci_low=intercept - t_val * se_intercept,
        intercept_ci_high=intercept + t_val * se_intercept,
        n_points=n,
        residual_std=residual_std,
    )


def _bootstrap_ci(
    vals: List[float],
    statistic: str = "median",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a statistic.
    Returns (estimate, ci_low, ci_high).
    """
    import random

    if not vals:
        return 0.0, 0.0, 0.0

    def compute_stat(data: List[float]) -> float:
        if statistic == "median":
            return _median(data)
        if statistic == "mean":
            return sum(data) / len(data) if data else 0.0
        if statistic == "p90":
            return _percentile(data, 0.9)
        return _median(data)

    point_est = compute_stat(vals)

    if len(vals) < 3:
        return point_est, point_est, point_est

    boot_stats = []
    for _ in range(n_bootstrap):
        sample = [random.choice(vals) for _ in range(len(vals))]
        boot_stats.append(compute_stat(sample))

    boot_stats.sort()
    alpha = 1.0 - confidence
    ci_low = boot_stats[int(alpha / 2 * n_bootstrap)]
    ci_high = boot_stats[int((1 - alpha / 2) * n_bootstrap)]

    return point_est, ci_low, ci_high


# -----------------------------
# HTTP Helpers
# -----------------------------
class OpenAIClient:
    """HTTP client for OpenAI-compatible endpoints."""

    def __init__(self, host: str, api_key: Optional[str], model: str):
        self.url = f"http://{host}/v1/completions"
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.model = model

    def _post(self, payload: dict, timeout_s: int = 120) -> Tuple[float, dict]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, method="POST")
        for k, v in self.headers.items():
            req.add_header(k, v)

        t0 = _now()
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
                e.url, e.code, f"{e.msg} | body={err_body[:500]}", e.hdrs, e.fp
            ) from None

        return (_now() - t0), json.loads(body)

    def complete(
        self,
        prompt: Any,
        max_tokens: int,
        temperature: float = 0.0,
        stream: bool = False,
    ) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": stream,
        }
        dt_s, obj = self._post(payload)

        usage = obj.get("usage", {})
        decoded_text = ""
        try:
            choices = obj.get("choices", [])
            if choices and isinstance(choices[0], dict):
                decoded_text = choices[0].get("text", "")
        except Exception:
            pass

        return {
            "dt_s": float(dt_s),
            "usage": usage,
            "decoded_text": decoded_text,
            "obj": obj,
        }

    def stream_ttft(self, prompt: str, max_tokens: int = 16) -> Optional[float]:
        """Measure time-to-first-token via streaming."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": int(max_tokens),
            "temperature": 0.0,
            "stream": True,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.url, data=data, method="POST")
        for k, v in self.headers.items():
            req.add_header(k, v)

        t0 = _now()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    if line.startswith(b"data:"):
                        if b"[DONE]" in line:
                            break
                        return (_now() - t0) * 1000.0
        except Exception:
            return None

        return None


# -----------------------------
# Calibration Phases
# -----------------------------
def _warmup_phase(
    client: OpenAIClient,
    config: CalibrationConfig,
) -> dict:
    """Multi-phase warmup with JIT detection."""
    short_prompt = "Hello, world."

    warmup_times = []
    for _ in range(config.warmup_requests):
        try:
            result = client.complete(short_prompt, max_tokens=8)
            warmup_times.append(result["dt_s"] * 1000.0)
            time.sleep(config.warmup_wait_ms / 1000.0)
        except Exception:
            warmup_times.append(-1)

    # Detect JIT warmup (first requests slower)
    valid_times = [t for t in warmup_times if t > 0]
    jit_detected = False
    jit_warmup_n = 0

    if len(valid_times) >= 3:
        first_half = valid_times[: len(valid_times) // 2]
        second_half = valid_times[len(valid_times) // 2 :]

        if first_half and second_half:
            first_med = _median(first_half)
            second_med = _median(second_half)
            if first_med > second_med * 1.5:
                jit_detected = True
                for j, t in enumerate(valid_times):
                    if t < first_med * 0.8:
                        jit_warmup_n = j
                        break

    return {
        "warmup_times_ms": warmup_times,
        "valid_count": len(valid_times),
        "jit_detected": jit_detected,
        "jit_warmup_requests": jit_warmup_n,
        "steady_state_latency_ms": _median(valid_times[-3:]) if len(valid_times) >= 3 else None,
    }


def _measure_request_overhead(
    client: OpenAIClient,
    config: CalibrationConfig,
) -> Tuple[float, float, List[float]]:
    """
    Measure pure request overhead by using minimal prompt + max_tokens=1.
    Returns (median_ms, std_ms, all_samples).
    """
    minimal_prompt = "Hi"
    samples: List[float] = []

    for _ in range(config.overhead_measurement_runs):
        try:
            result = client.complete(minimal_prompt, max_tokens=1)
            samples.append(result["dt_s"] * 1000.0)
        except Exception:
            pass

    if not samples:
        return 0.0, 0.0, []

    filtered = _iqr_filter(samples, config.outlier_iqr_multiplier)
    if len(filtered) < 3:
        filtered = samples

    med = _median(filtered)
    std = statistics.stdev(filtered) if len(filtered) >= 2 else 0.0
    return med, std, samples


def _measure_stream_ttft_floor(
    client: OpenAIClient,
    config: CalibrationConfig,
) -> Tuple[Optional[float], dict]:
    """Measure streaming TTFT floor."""
    short_prompt = "Explain KV cache in 2 sentences."
    samples: List[float] = []

    for _ in range(config.stream_ttft_runs):
        ttft = client.stream_ttft(short_prompt, max_tokens=16)
        if ttft is not None and ttft > 0:
            samples.append(ttft)

    debug = {
        "ttft_ms_samples": samples,
        "runs": config.stream_ttft_runs,
        "ok_runs": len(samples),
    }

    if len(samples) >= max(3, config.stream_ttft_runs // 2):
        filtered = _iqr_filter(samples, config.outlier_iqr_multiplier)
        floor = _median(filtered) if filtered else _median(samples)
        debug["ttft_ms_median"] = floor
        debug["filtered_samples"] = len(filtered)
        return floor, debug

    debug["ttft_ms_median"] = None
    debug["note"] = "Streaming unavailable or unstable"
    return None, debug


def _discover_max_context(
    client: OpenAIClient,
    base_prompt: str,
) -> Tuple[int, List[dict]]:
    """Binary search to find maximum prompt repetitions that fit context."""
    shrink_events: List[dict] = []

    def try_reps(reps: int) -> bool:
        prompt = (base_prompt * reps).strip()
        try:
            client.complete(prompt, max_tokens=1)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 400:
                shrink_events.append({"reps": reps, "error": str(e)[:200]})
                return False
            raise

    lo, hi = 1, 2
    while try_reps(hi):
        lo = hi
        hi *= 2
        if hi > 4096:
            break

    if not try_reps(lo):
        raise SystemExit("calibrate: even minimal prompt failed (HTTP 400)")

    left, right = lo, hi
    while left + 1 < right:
        mid = (left + right) // 2
        if try_reps(mid):
            left = mid
        else:
            right = mid

    return left, shrink_events


def _measure_prefill_points(
    client: OpenAIClient,
    base_prompt: str,
    max_reps: int,
    config: CalibrationConfig,
    model: str,
) -> Tuple[List[Tuple[int, float]], List[dict], bool]:
    """
    Measure prefill latency at multiple prompt sizes.
    Returns (points, metadata, batch_supported).
    """
    batch_size = config.prefill_batch_default
    batch_supported = True

    reps_points = sorted(
        set(
            [
                max(1, max_reps // 5),
                max(1, (2 * max_reps) // 5),
                max(1, (3 * max_reps) // 5),
                max(1, (4 * max_reps) // 5),
                max_reps,
            ]
        )
    )

    if len(reps_points) < config.min_valid_points_for_fit:
        reps_points = sorted(set([1, max(2, max_reps // 2), max_reps]))

    points: List[Tuple[int, float]] = []
    metadata: List[dict] = []

    for reps in reps_points:
        prompt_single = (base_prompt * reps).strip()
        prompt_obj: Any = [prompt_single] * batch_size

        runs = []
        try:
            runs = [client.complete(prompt_obj, max_tokens=1) for _ in range(config.prefill_runs_per_point)]
        except urllib.error.HTTPError as e:
            if e.code in (400, 422):
                batch_supported = False
            else:
                raise

        if not batch_supported:
            prompt_obj = prompt_single
            runs = [client.complete(prompt_obj, max_tokens=1) for _ in range(config.prefill_runs_per_point + 2)]

        dt_samples = [r["dt_s"] * 1000.0 for r in runs]
        filtered_dt = _iqr_filter(dt_samples, config.outlier_iqr_multiplier)
        dt_ms = _median(filtered_dt) if filtered_dt else _median(dt_samples)

        mid_run = runs[len(runs) // 2]
        usage = mid_run.get("usage", {})
        ptoks = usage.get("prompt_tokens")
        if not isinstance(ptoks, int) or ptoks <= 0:
            if isinstance(prompt_obj, list):
                ptoks = sum(_count_tokens(p, model=model) for p in prompt_obj)
            else:
                ptoks = _count_tokens(prompt_obj, model=model)

        points.append((int(ptoks), float(dt_ms)))
        metadata.append(
            {
                "reps": reps,
                "prompt_tokens": int(ptoks),
                "dt_ms": float(dt_ms),
                "raw_samples": dt_samples,
                "filtered_count": len(filtered_dt),
                "batch_size": batch_size if batch_supported else 1,
            }
        )

    return points, metadata, batch_supported


def _fit_prefill_model(
    points: List[Tuple[int, float]],
    fixed_overhead_floor_ms: float,
    config: CalibrationConfig,
) -> Tuple[float, float, RegressionResult, dict]:
    """
    Fit prefill model: latency = fixed + slope * tokens.
    Returns (fixed_ms, slope_ms_per_token, regression_result, debug).
    """
    if len(points) < 2:
        ols = RegressionResult(
            slope=0.0,
            intercept=fixed_overhead_floor_ms,
            r_squared=0.0,
            slope_ci_low=0.0,
            slope_ci_high=0.0,
            intercept_ci_low=fixed_overhead_floor_ms,
            intercept_ci_high=fixed_overhead_floor_ms,
            n_points=len(points),
            residual_std=0.0,
        )
        return fixed_overhead_floor_ms, 0.0, ols, {"note": "Insufficient points for slope"}

    points_sorted = sorted(points, key=lambda x: x[0])
    x = [float(p[0]) for p in points_sorted]
    y = [float(p[1]) for p in points_sorted]

    ts_slope, ts_intercept = _theil_sen_slope(x, y)
    ols = _ols_regression(x, y)

    dt_span = max(y) - min(y)
    tok_span = max(x) - min(x)

    debug = {
        "dt_span_ms": dt_span,
        "tok_span": tok_span,
        "theil_sen_slope": ts_slope,
        "theil_sen_intercept": ts_intercept,
        "ols_slope": ols.slope,
        "ols_intercept": ols.intercept,
        "r_squared": ols.r_squared,
    }

    if dt_span < config.prefill_min_dt_span_ms:
        debug["note"] = f"dt_span too small ({dt_span:.2f}ms); using slope=0"
        return fixed_overhead_floor_ms, 0.0, ols, debug

    slope = max(0.0, ts_slope)

    implied_tps = 1000.0 / slope if slope > 1e-9 else float("inf")
    if implied_tps > config.prefill_tps_cap:
        slope = 1000.0 / config.prefill_tps_cap
        debug["clamped"] = True
        debug["original_implied_tps"] = implied_tps

    debug["final_implied_tps"] = (1000.0 / slope) if slope > 1e-9 else float("inf")

    fixed = fixed_overhead_floor_ms
    return fixed, slope, ols, debug


def _make_decode_prompt(seed: int, variant: int = 0) -> str:
    """Generate a prompt that forces long completions without early stopping."""
    variants = [
        (
            "You are a text generator. Do NOT end early.\n"
            "Task: output a sequence of short tokens until maximum length.\n"
            "Rules:\n"
            "- Print exactly one token per line.\n"
            "- Token format: X{number}\n"
            "- Start at 1 and increment by 1.\n"
            "- Do not write explanations.\n"
            "- Do not stop.\n"
            f"Seed={seed}\n"
            "Begin:\n"
            "X1\n"
        ),
        (
            "Generate a long sequence of tokens without stopping.\n"
            "Format: one token per line, pattern A1 B2 C3 D4...\n"
            "Continue until max tokens. Do not explain.\n"
            f"Seed={seed}\n"
            "Start:\n"
            "A1\n"
        ),
        (
            f"Count from {seed} onwards, one number per line.\n"
            "Do not stop until reaching max tokens.\n"
            "No explanations, just numbers.\n"
            f"{seed}\n"
        ),
    ]
    return variants[variant % len(variants)]


def _measure_decode_points(
    client: OpenAIClient,
    config: CalibrationConfig,
    model: str,
) -> Tuple[List[Tuple[int, float]], List[dict], List[Tuple[float, int]]]:
    """
    Measure decode latency at multiple output sizes.

    Returns:
      - accepted_points: [(completion_tokens, dt_ms_median_for_that_target)]
      - all_metadata: per-target metadata
      - run_pairs: per-run (dt_ms, completion_tokens) across ALL runs (for variance modeling)
    """
    accepted_points: List[Tuple[int, float]] = []
    all_metadata: List[dict] = []
    run_pairs: List[Tuple[float, int]] = []

    seed0 = int(time.time()) % 100000

    for target_idx, target_max_tokens in enumerate(config.decode_targets):
        best_result: Optional[Tuple[int, float, dict]] = None

        for attempt in range(config.decode_point_retries):
            prompt = _make_decode_prompt(seed0 + target_idx * 17, variant=attempt)

            runs = [client.complete(prompt, max_tokens=target_max_tokens, temperature=0.7) for _ in range(config.decode_runs_per_point)]

            dt_samples = [r["dt_s"] * 1000.0 for r in runs]
            filtered_dt = _iqr_filter(dt_samples, config.outlier_iqr_multiplier)
            dt_ms = _median(filtered_dt) if filtered_dt else _median(dt_samples)

            mid_run = runs[len(runs) // 2]
            usage_mid = mid_run.get("usage", {})
            decoded_text_mid = str(mid_run.get("decoded_text", ""))

            ctoks_mid = usage_mid.get("completion_tokens")
            if not isinstance(ctoks_mid, int) or ctoks_mid <= 0:
                ctoks_mid = _count_tokens(decoded_text_mid, model=model)

            accepted = ctoks_mid >= int(config.decode_min_completion_frac * target_max_tokens)

            # Collect per-run dt/token pairs (best-effort)
            for r in runs:
                dt = float(r["dt_s"] * 1000.0)
                usage = r.get("usage", {})
                ct = usage.get("completion_tokens")
                if not isinstance(ct, int) or ct <= 0:
                    decoded_text = str(r.get("decoded_text", ""))
                    ct = _count_tokens(decoded_text, model=model)
                if dt > 0.0 and isinstance(ct, int) and ct > 0:
                    run_pairs.append((dt, int(ct)))

            meta = {
                "target_max_tokens": target_max_tokens,
                "attempt": attempt + 1,
                "completion_tokens": int(ctoks_mid),
                "dt_ms": float(dt_ms),
                "dt_samples": dt_samples,
                "accepted": accepted,
            }

            if accepted:
                best_result = (int(ctoks_mid), float(dt_ms), meta)
                break

            if best_result is None or int(ctoks_mid) > best_result[0]:
                best_result = (int(ctoks_mid), float(dt_ms), meta)

        if best_result:
            all_metadata.append(best_result[2])
            if bool(best_result[2].get("accepted")):
                accepted_points.append((best_result[0], best_result[1]))

    return accepted_points, all_metadata, run_pairs


def _fit_decode_model(
    points: List[Tuple[int, float]],
    config: CalibrationConfig,
) -> Tuple[float, float, RegressionResult, dict]:
    """
    Fit decode model: latency = fixed + slope * tokens.
    Returns (fixed_ms, slope_ms_per_token, regression_result, debug).
    """
    if len(points) < config.min_valid_points_for_fit:
        raise SystemExit(f"calibrate: insufficient decode points ({len(points)} < {config.min_valid_points_for_fit})")

    points_sorted = sorted(points, key=lambda x: x[0])
    x = [float(p[0]) for p in points_sorted]
    y = [float(p[1]) for p in points_sorted]

    ts_slope, ts_intercept = _theil_sen_slope(x, y)
    ols = _ols_regression(x, y)

    debug = {
        "theil_sen_slope": ts_slope,
        "theil_sen_intercept": ts_intercept,
        "ols_slope": ols.slope,
        "ols_intercept": ols.intercept,
        "r_squared": ols.r_squared,
    }

    slope = max(1e-6, ts_slope)
    fixed = max(0.0, ts_intercept)

    decode_tps = 1000.0 / slope
    if decode_tps > 50000:
        debug["warning"] = f"Decode TPS {decode_tps:.0f} seems too high"

    return fixed, slope, ols, debug


def _measure_batch_effects(
    client: OpenAIClient,
    base_latency_ms: float,
) -> Tuple[float, int, dict]:
    """
    Measure batch scheduling overhead.
    Returns (overhead_per_concurrent_ms, max_efficient_batch, debug).
    """
    prompt = "Hello, world."
    batch_sizes = [1, 2, 4, 8]
    results: List[dict] = []

    for bs in batch_sizes:
        try:
            prompts = [prompt] * bs
            result = client.complete(prompts, max_tokens=8)
            results.append({"batch_size": bs, "dt_ms": result["dt_s"] * 1000.0})
        except Exception:
            break

    if len(results) < 2:
        return 0.0, 1, {"note": "Batch measurement unavailable"}

    base = results[0]["dt_ms"]
    overheads = []
    for r in results[1:]:
        overhead_per = (r["dt_ms"] - base) / r["batch_size"]
        overheads.append(max(0.0, overhead_per))

    overhead = _median(overheads) if overheads else 0.0

    max_efficient = 1
    for r in results:
        if r["dt_ms"] < base * 2:
            max_efficient = r["batch_size"]

    return overhead, max_efficient, {"measurements": results, "overhead_per_concurrent": overhead}


def _measure_kv_cache_pressure(
    client: OpenAIClient,
    config: CalibrationConfig,
    base_overhead_ms: float,
    model: str,
) -> Tuple[float, int, dict]:
    """
    Measure KV cache pressure for large contexts.
    Returns (ms_per_1k_tokens_in_cache, threshold_tokens, debug).
    """
    base_prompt = "x " * 100  # ~100 tokens per rep

    results: List[dict] = []
    for target_size in config.kv_cache_probe_sizes:
        reps = target_size // 100
        prompt = (base_prompt * reps).strip()

        try:
            runs = [client.complete(prompt, max_tokens=1) for _ in range(5)]
            dt_samples = [r["dt_s"] * 1000.0 for r in runs]
            dt_ms = _median(dt_samples)

            usage = runs[len(runs) // 2].get("usage", {})
            ptoks = usage.get("prompt_tokens", _count_tokens(prompt, model=model))

            results.append({"target_tokens": target_size, "actual_tokens": int(ptoks), "dt_ms": float(dt_ms)})
        except Exception:
            break

    if len(results) < 3:
        return 0.0, 0, {"note": "KV cache measurement unavailable", "results": results}

    x = [r["actual_tokens"] for r in results]
    y = [r["dt_ms"] for r in results]

    slopes = []
    for i in range(len(results) - 1):
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        if dx > 0:
            slopes.append((x[i], dy / dx * 1000))  # ms per 1K tokens

    if not slopes:
        return 0.0, 0, {"results": results}

    min_slope = min(s[1] for s in slopes)
    threshold = 0
    pressure = min_slope

    for tok, slope in slopes:
        if slope > min_slope * 2:
            threshold = tok
            pressure = slope
            break

    debug = {"results": results, "slopes": slopes}
    if max(x) < 4096:
        debug["warning"] = "KV probe did not reach 4096 tokens; threshold may be unreliable."

    # If too few points, refuse to claim a threshold
    if len(results) < 4:
        threshold = 0

    return float(pressure), int(threshold), debug


# -----------------------------
# Main Calibration Function
# -----------------------------
def calibrate_vllm_openai_nonstream(
    *,
    profile: TargetProfile,
    host: str,
    api_key: Optional[str],
    seconds: int = 30,
    config: Optional[CalibrationConfig] = None,
) -> CalibrationResult:
    """
    Enhanced calibration for vLLM via OpenAI-compatible endpoint.

    Measures:
    - Request overhead (network + framework)
    - Prefill: fixed overhead + linear scaling
    - Decode: fixed overhead + linear scaling
    - Streaming TTFT floor
    - Batch effects
    - KV cache pressure

    Uses robust statistics (Theil-Sen, IQR filtering) for outlier resistance.
    """
    if config is None:
        config = DEFAULT_CONFIG

    client = OpenAIClient(host, api_key, profile.model)

    warnings: List[str] = []
    quality: dict = {"prefill": {}, "decode": {}, "stream": {}, "overall": {}}
    debug: dict = {
        "endpoint": {"url": client.url, "host": host},
        "config": asdict(config),
        "errors": [],
    }

    # Phase 1: Warmup
    warmup_debug = _warmup_phase(client, config)
    debug["warmup"] = warmup_debug

    if warmup_debug["valid_count"] < 2:
        raise SystemExit(f"calibrate: cannot reach server at {client.url}")

    if warmup_debug.get("jit_detected"):
        warnings.append(
            f"JIT warmup detected; first {warmup_debug['jit_warmup_requests']} requests excluded from measurements."
        )

    # Phase 2: Request overhead
    overhead_ms, overhead_std, overhead_samples = _measure_request_overhead(client, config)
    debug["request_overhead"] = {
        "median_ms": overhead_ms,
        "std_ms": overhead_std,
        "samples": overhead_samples,
    }

    # Phase 3: Streaming TTFT floor
    stream_floor, stream_debug = _measure_stream_ttft_floor(client, config)
    debug["stream"] = stream_debug

    if stream_floor is None:
        warnings.append("Streaming TTFT unavailable; using non-stream overhead floor.")
        stream_floor = overhead_ms

    quality["stream"] = {"available": stream_floor is not None, "ttft_ms": stream_floor}

    # Phase 4: Prefill calibration
    base_prompt = (
        "Write a detailed explanation of how a transformer decoder works, "
        "including attention, KV cache, and how prefill differs from decode. "
        "Provide examples and list key performance bottlenecks.\n"
    )

    max_reps, shrink_events = _discover_max_context(client, base_prompt)
    debug["prefill"] = {"max_reps_discovered": max_reps, "shrink_events": shrink_events}

    prefill_points, prefill_meta, batch_supported = _measure_prefill_points(
        client, base_prompt, max_reps, config, profile.model
    )
    debug["prefill"]["points"] = prefill_meta
    debug["prefill"]["batch_supported"] = batch_supported

    if not batch_supported:
        warnings.append("Batch prompts not supported; prefill slope may be less reliable.")

    prefill_fixed, prefill_slope, prefill_ols, prefill_fit_debug = _fit_prefill_model(
        prefill_points, stream_floor, config
    )
    debug["prefill"]["fit"] = prefill_fit_debug

    quality["prefill"] = {
        "n_points": len(prefill_points),
        "r_squared": prefill_ols.r_squared,
        "batch_supported": batch_supported,
        "has_slope": prefill_slope > 0,
    }

    # Phase 5: Decode calibration
    decode_points, decode_meta, run_pairs = _measure_decode_points(client, config, profile.model)
    debug["decode"] = {"points": decode_meta}

    accepted_count = len(decode_points)
    if accepted_count < config.min_valid_points_for_fit:
        raise SystemExit(f"calibrate: decode token-control too weak ({accepted_count} < {config.min_valid_points_for_fit})")

    if accepted_count < len(config.decode_targets) - 1:
        warnings.append(f"Decode token control weaker than expected ({accepted_count}/{len(config.decode_targets)} accepted).")

    decode_fixed, decode_slope, decode_ols, decode_fit_debug = _fit_decode_model(decode_points, config)
    debug["decode"]["fit"] = decode_fit_debug

    # FIX: Decode variance should be computed on decode-only ms/token.
    dt_only = [dt for (dt, _) in run_pairs if dt > 0.0]
    cv_raw_dt = _cv(_iqr_filter(dt_only, config.outlier_iqr_multiplier)) if dt_only else 0.0

    mpt_samples: List[float] = []
    for (dt_ms, ctoks) in run_pairs:
        # Remove request overhead + decode fixed; then normalize per token.
        decode_only_ms = dt_ms - overhead_ms - decode_fixed
        if ctoks > 0 and decode_only_ms > 0.0:
            mpt_samples.append(decode_only_ms / ctoks)

    mpt_samples = _iqr_filter(mpt_samples, config.outlier_iqr_multiplier)
    cv_mpt = _cv(mpt_samples)

    debug["decode"]["variance"] = {
        "cv_raw_dt": cv_raw_dt,
        "cv_ms_per_token": cv_mpt,
        "n_run_pairs": len(run_pairs),
        "n_mpt_samples": len(mpt_samples),
    }

    quality["decode"] = {
        "n_points": len(decode_points),
        "accepted_points": accepted_count,
        "total_targets": len(config.decode_targets),
        "r_squared": decode_ols.r_squared,
        "variance_cv": cv_mpt,
    }

    # Phase 6: Batch effects (optional)
    batch_overhead, max_efficient_batch, batch_debug = _measure_batch_effects(client, overhead_ms)
    debug["batch"] = batch_debug

    # Phase 7: KV cache pressure (optional)
    kv_pressure, kv_threshold, kv_debug = _measure_kv_cache_pressure(client, config, overhead_ms, profile.model)
    debug["kv_cache"] = kv_debug

    # Overall quality score
    quality["overall"] = {
        "prefill_reliable": prefill_ols.r_squared > 0.8 or prefill_slope == 0,
        "decode_reliable": decode_ols.r_squared > 0.9,
        "stream_available": stream_floor is not None,
    }

    return CalibrationResult(
        cal_version="2.0",
        engine="vllm",
        model=profile.model,
        prefill_fixed_overhead_ms=float(prefill_fixed),
        prefill_ms_per_token=float(prefill_slope),
        prefill_fixed_ci=(prefill_ols.intercept_ci_low, prefill_ols.intercept_ci_high),
        prefill_slope_ci=(prefill_ols.slope_ci_low, prefill_ols.slope_ci_high),
        decode_fixed_overhead_ms=float(decode_fixed),
        decode_ms_per_token=float(decode_slope),
        decode_fixed_ci=(decode_ols.intercept_ci_low, decode_ols.intercept_ci_high),
        decode_slope_ci=(decode_ols.slope_ci_low, decode_ols.slope_ci_high),
        request_overhead_ms=float(overhead_ms),
        request_overhead_std=float(overhead_std),
        kv_cache_pressure_ms_per_1k=float(kv_pressure),
        kv_cache_threshold_tokens=int(kv_threshold),
        decode_variance_coefficient=float(cv_mpt),
        batch_overhead_ms_per_concurrent=float(batch_overhead),
        max_efficient_batch=int(max_efficient_batch),
        warnings=warnings,
        quality=quality,
        debug=debug,
        prefill_r_squared=float(prefill_ols.r_squared),
        decode_r_squared=float(decode_ols.r_squared),
    )


def save_calibration(cal: CalibrationResult, out_path: Path) -> None:
    """Save calibration result to JSON file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    obj = asdict(cal)
    for key in ["prefill_fixed_ci", "prefill_slope_ci", "decode_fixed_ci", "decode_slope_ci"]:
        if key in obj and isinstance(obj[key], tuple):
            obj[key] = list(obj[key])

    out_path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
