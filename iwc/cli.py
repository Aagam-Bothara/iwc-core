# iwc/cli.py
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import jsonschema

from iwc.profile import load_profile, validate_profile, load_and_validate_target_profile
from iwc.calibrate import calibrate_vllm_openai_nonstream, save_calibration, CalibrationConfig
from iwc.predict import load_calibration, predict_workload
from iwc.eval import eval_workloads, format_eval_text

from iwc.fingerprint import build_fingerprint_from_report_json, build_fingerprint_extended
from iwc.report import build_report, format_report, report_to_dict

from iwc.analyze.cli import add_analyze_subcommand
from iwc.compile import (
    SimpleJsonConfig,
    ShareGPTConfig,
    compile_simple_json,
    compile_jsonl_prompts,
    compile_sharegpt,
)
from iwc.export import ExportAiperfConfig, export_aiperf
from iwc.labeler.heuristics import label_record
from iwc.diff.cli import add_diff_subcommand


# -----------------------------
# Utilities
# -----------------------------
def _canonical_json_line(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _print_progress(message: str, verbose: bool = True) -> None:
    if verbose:
        print(f"[INFO] {message}", file=sys.stderr)


def _print_error(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)


def _print_warning(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def _load_config_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
            return yaml.safe_load(text) or {}
        except ImportError:
            _print_warning("PyYAML not installed; trying JSON parse")
    return json.loads(text)


def _to_float(x: Any) -> float | None:
    """Best-effort conversion; returns None if not convertible."""
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


# -----------------------------
# Command Implementations
# -----------------------------
def cmd_calibrate(args: argparse.Namespace) -> None:
    verbose = getattr(args, "verbose", False)
    start_time = time.time()
    repo_root = Path(__file__).resolve().parents[1]

    _print_progress(f"Loading profile from {args.profile}...", verbose)
    try:
        prof = load_and_validate_target_profile(Path(args.profile), repo_root=repo_root)
    except Exception as e:
        _print_error(f"Failed to load profile: {e}")
        raise SystemExit(1)

    if prof.engine != "vllm":
        _print_error(f"Unsupported engine: {prof.engine}. Only 'vllm' is supported.")
        raise SystemExit(1)

    _print_progress(f"Connecting to server at {args.host}...", verbose)

    cfg = CalibrationConfig()
    if getattr(args, "prefill_points", None):
        cfg = CalibrationConfig(
            prefill_num_points=int(args.prefill_points),
            decode_targets=tuple(int(x) for x in args.decode_targets.split(",")) if args.decode_targets else cfg.decode_targets,
        )

    try:
        _print_progress("Running calibration...", verbose)
        cal = calibrate_vllm_openai_nonstream(
            profile=prof,
            host=args.host,
            api_key=args.api_key,
            config=cfg,
        )
    except Exception as e:
        _print_error(f"Calibration failed: {e}")
        _print_error(f"Check reachability: curl http://{args.host}/v1/models")
        raise SystemExit(1)

    save_calibration(cal, Path(args.out))

    # health gate (if your calibrate wrote debug.health, keep it)
    cal_obj = json.loads(Path(args.out).read_text(encoding="utf-8"))
    if "debug" not in cal_obj:
        cal_obj["debug"] = {}
    if "health" not in cal_obj["debug"]:
        # If your calibrate.py already writes debug.health, this won't run.
        cal_obj["debug"]["health"] = {"status": "ok", "reasons": []}
    Path(args.out).write_text(json.dumps(cal_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    elapsed = time.time() - start_time
    print(f"✓ Calibration saved to {args.out} [{_format_duration(elapsed)}]")

    if verbose:
        print("\nCalibration Summary:")
        print(f"  Prefill: {cal.prefill_fixed_overhead_ms:.2f}ms + {cal.prefill_ms_per_token:.4f}ms/tok")
        print(f"  Decode:  {cal.decode_fixed_overhead_ms:.2f}ms + {cal.decode_ms_per_token:.4f}ms/tok")
        if cal.request_overhead_ms > 0:
            print(f"  Request overhead: {cal.request_overhead_ms:.2f}ms")
        if cal.decode_ms_per_token > 0:
            print(f"  Decode TPS: {1000.0/cal.decode_ms_per_token:.0f} tok/s")


def cmd_eval(args: argparse.Namespace) -> None:
    verbose = getattr(args, "verbose", False)
    start_time = time.time()

    repo_root = Path(__file__).resolve().parents[1]
    prof = load_and_validate_target_profile(Path(args.profile), repo_root=repo_root)

    inputs = [Path(p.strip()) for p in args.inputs.split(",") if p.strip()]
    for inp in inputs:
        if not inp.exists():
            _print_error(f"Workload file not found: {inp}")
            raise SystemExit(1)

    conc = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]

    _print_progress(f"Evaluating {len(inputs)} workload(s) at concurrency {conc}...", verbose)

    out = eval_workloads(
        inputs=inputs,
        profile=prof,
        cal_path=Path(args.cal),
        host=args.host,
        api_key=args.api_key,
        concurrency_list=conc,
        repeats=int(args.repeats),
        ttft_sample_n=int(args.ttft_sample_n),
        timeout_s=int(args.timeout_s),
        debug_sched=bool(getattr(args, "debug_sched", False)),
        bootstrap_samples=int(getattr(args, "bootstrap_samples", 1000)),
    )

    elapsed = time.time() - start_time

    if args.format == "json":
        s = json.dumps(out, indent=2, sort_keys=True)
        if args.out:
            Path(args.out).write_text(s + "\n", encoding="utf-8")
            print(f"✓ Results saved to {args.out} [{_format_duration(elapsed)}]")
        else:
            print(s)
        return

    txt = format_eval_text(out)
    if args.out:
        Path(args.out).write_text(txt, encoding="utf-8")
        print(f"✓ Results saved to {args.out} [{_format_duration(elapsed)}]")
    else:
        print(txt)


def cmd_predict(args: argparse.Namespace) -> None:
    verbose = getattr(args, "verbose", False)

    repo_root = Path(__file__).resolve().parents[1]
    prof = load_and_validate_target_profile(Path(args.profile), repo_root=repo_root)
    cal = load_calibration(Path(args.cal))

    _print_progress(f"Predicting latency for {args.input}...", verbose)

    out = predict_workload(
        workload_jsonl=Path(args.input),
        profile=prof,
        cal=cal,
        concurrency=int(args.concurrency),
        target_utilization=float(getattr(args, "target_utilization", 0.7)),
    )

    if args.format == "json":
        print(json.dumps(out, indent=2, sort_keys=True))
        return

    decision = out.get("decision", {})
    pred = out.get("predicted", {})
    breakdown = out.get("breakdown", {})
    throughput = out.get("throughput", {})
    queueing = out.get("queueing", {}) or {}

    # --- Verdict First ---
    print("=== DECISION ===")
    if decision:
        print(f"Verdict:    {decision.get('verdict', 'UNKNOWN')}")
        print(f"Confidence: {decision.get('confidence', 'UNKNOWN')}")
        print(f"SLA:        {float(decision.get('sla_ms', 0.0)):.1f} ms")
        print(f"P(breach):  {100.0*float(decision.get('sla_breach_probability', 0.0)):.1f}%")

        if decision.get("max_safe_concurrency_est") is not None:
            print(f"Max safe conc (est): {decision['max_safe_concurrency_est']}")
        if decision.get("recommended_max_concurrency") is not None:
            print(f"Recommended conc:    {decision['recommended_max_concurrency']}")
        if decision.get("recommended_replicas") is not None:
            print(f"Recommended replicas:{decision['recommended_replicas']}")

        rs = decision.get("reasons", [])
        if rs:
            print("Reasons:")
            for r in rs[:6]:
                print(f"  - {r}")

        ns = decision.get("notes", [])
        if ns and verbose:
            print("Notes:")
            for n in ns[:6]:
                print(f"  - {n}")
    else:
        print("Verdict: (missing decision layer)")

    # --- Supporting metrics ---
    print("\n=== SUPPORTING METRICS ===")
    print(f"E2E p90:        {float(pred.get('e2e_time_p90_ms', 0.0)):.2f} ms")
    print(f"E2E p99:        {float(pred.get('e2e_time_p99_ms', 0.0)):.2f} ms")
    print(f"Service p90:    {float(pred.get('service_time_p90_ms', 0.0)):.2f} ms")

    # TTFT is diagnostic: only show in verbose
    if verbose:
        print(f"TTFT p90:       {float(pred.get('ttft_p90_ms', 0.0)):.2f} ms (diagnostic)")

    if breakdown:
        print("\n=== BREAKDOWN ===")
        if "request_overhead_ms" in breakdown:
            print(f"Request overhead: {float(breakdown.get('request_overhead_ms', 0.0)):.2f} ms")
        print(f"Prefill:          {float(breakdown.get('prefill_ms', 0.0)):.2f} ms")
        print(f"Decode:           {float(breakdown.get('decode_ms', 0.0)):.2f} ms")
        if float(breakdown.get("queueing_delay_ms", 0.0)) > 0.1:
            print(f"Queueing delay:   {float(breakdown.get('queueing_delay_ms', 0.0)):.2f} ms")

    # -------------------------
    # QUEUEING (fixed: handle None / not computed)
    # -------------------------
    qc = bool(queueing.get("queueing_computed", True))
    util = _to_float(queueing.get("utilization"))
    qprob = _to_float(queueing.get("queue_probability"))
    qlen = _to_float(queueing.get("expected_queue_length"))

    if not qc:
        print("\n=== QUEUEING ===")
        print("Queueing:          N/A (arrival rate undefined)")
    elif util is not None and util > 0:
        print("\n=== QUEUEING ===")
        print(f"Utilization:       {100.0*util:.1f}%")
        if qprob is not None:
            print(f"Queue probability: {100.0*qprob:.1f}%")
        if qlen is not None:
            print(f"Expected q length: {qlen:.2f}")

    if throughput:
        print("\n=== THROUGHPUT ===")
        if "max_throughput_rps" in throughput:
            print(f"Max throughput:         {float(throughput.get('max_throughput_rps', 0.0)):.2f} req/s")
        if "sustainable_throughput_rps" in throughput:
            print(f"Sustainable throughput: {float(throughput.get('sustainable_throughput_rps', 0.0)):.2f} req/s")

    if out.get("warnings"):
        print("\n⚠️ Warnings:")
        for w in out["warnings"]:
            print(f"  - {w}")


def cmd_label(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    errors = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                _print_error(f"{in_path}:{line_no}: invalid JSON: {e}")
                errors += 1
                continue

            if not isinstance(obj, dict):
                _print_error(f"{in_path}:{line_no}: expected JSON object")
                errors += 1
                continue

            labeled = label_record(obj, overwrite=args.overwrite)
            fout.write(_canonical_json_line(labeled) + "\n")
            n += 1

    print(f"✓ Labeled {n} requests -> {out_path}")
    if errors > 0:
        _print_warning(f"{errors} lines had errors and were skipped")


def cmd_report(args: argparse.Namespace) -> None:
    if not Path(args.input).exists():
        _print_error(f"Workload file not found: {args.input}")
        raise SystemExit(1)

    r = build_report(Path(args.input))

    if args.format == "text":
        print(format_report(r, top_k_tags=args.top_k_tags))
        return

    obj = report_to_dict(r, top_k_tags=args.top_k_tags)
    print(json.dumps(obj, indent=2, sort_keys=True))


def cmd_fingerprint(args: argparse.Namespace) -> int:
    from iwc.fingerprint import build_fingerprint

    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"fingerprint: input not found: {inp}")

    if getattr(args, "extended", False):
        fp = build_fingerprint_extended(inp, include_distributions=True)
    else:
        fp = build_fingerprint(inp)

    s = json.dumps(fp, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if getattr(args, "out", None):
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(s, encoding="utf-8")
    else:
        print(s, end="")
    return 0

def cmd_profile_validate(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        prof = load_profile(Path(args.profile))
        validate_profile(prof, repo_root=repo_root)
        print(f"✓ {args.profile}")
    except Exception as e:
        _print_error(f"Profile validation failed: {e}")
        raise SystemExit(1)


def cmd_validate(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    schema_path = repo_root / "schema" / "workload.schema.json"

    if not schema_path.exists():
        _print_error(f"Schema not found: {schema_path}")
        raise SystemExit(1)

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)

    total_files = 0
    failed_files = 0

    for p in args.paths:
        path = Path(p)
        jsonl_files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]

        for jsonl in jsonl_files:
            total_files += 1
            errors_found = False

            with jsonl.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        _print_error(f"{jsonl}:{line_no}: invalid JSON: {e}")
                        errors_found = True
                        continue

                    errors = list(validator.iter_errors(obj))
                    if errors:
                        for err in errors[:3]:
                            _print_error(f"{jsonl}:{line_no}: {list(err.path)}: {err.message}")
                        errors_found = True

            if errors_found:
                failed_files += 1
                print(f"✗ {jsonl}")
            else:
                print(f"✓ {jsonl}")

    if failed_files > 0:
        print(f"\n{failed_files}/{total_files} files failed validation")
        raise SystemExit(1)

    print(f"\n{total_files} files validated successfully")


def cmd_compile_simple_json(args: argparse.Namespace) -> None:
    out_path = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else Path(str(out_path) + ".manifest.yaml")

    cfg = SimpleJsonConfig(
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        streaming=args.streaming,
        arrival=args.arrival,
        arrival_step_ms=args.arrival_step_ms,
        rate_rps=args.rate_rps,
        seed=args.seed,
    )

    compile_simple_json(Path(args.input), out_path, manifest_path, cfg)
    print(f"✓ Workload: {args.output}")
    print(f"✓ Manifest: {manifest_path}")


def cmd_compile_jsonl_prompts(args: argparse.Namespace) -> None:
    out_path = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else Path(str(out_path) + ".manifest.yaml")

    cfg = SimpleJsonConfig(
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        streaming=args.streaming,
        arrival=args.arrival,
        arrival_step_ms=args.arrival_step_ms,
        rate_rps=args.rate_rps,
        seed=args.seed,
    )

    compile_jsonl_prompts(
        Path(args.input),
        out_path,
        manifest_path,
        cfg,
        prompt_format=args.prompt_format,
    )
    print(f"✓ Workload: {args.output}")
    print(f"✓ Manifest: {manifest_path}")


def cmd_compile_sharegpt(args: argparse.Namespace) -> None:
    out_path = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else Path(str(out_path) + ".manifest.yaml")

    cfg = ShareGPTConfig(
        mode=args.mode,
        user_tag=args.user_tag,
        assistant_tag=args.assistant_tag,
        separator=args.separator,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        streaming=args.streaming,
        arrival=args.arrival,
        arrival_step_ms=args.arrival_step_ms,
        rate_rps=args.rate_rps,
        seed=args.seed,
    )

    compile_sharegpt(Path(args.input), out_path, manifest_path, cfg)
    print(f"✓ Workload: {args.output}")
    print(f"✓ Manifest: {manifest_path}")


def cmd_export_aiperf(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest) if args.manifest else None
    cfg = ExportAiperfConfig(time_mode=args.time_mode)

    mp = export_aiperf(
        Path(args.input),
        Path(args.output),
        manifest_path=manifest_path,
        cfg=cfg,
    )

    print(f"✓ Trace: {args.output}")
    print(f"✓ Manifest: {mp}")


# --------------------
# Main Parser
# --------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="iwc",
        description="Inference Workload Characterization toolkit for LLM inference prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--config", type=str, help="Path to config file (YAML or JSON)")

    sub = parser.add_subparsers(dest="cmd", required=True, title="commands")

    add_diff_subcommand(sub)
    add_analyze_subcommand(sub)

    # compile
    p_comp = sub.add_parser("compile", help="Compile a dataset into canonical workload JSONL")
    comp_sub = p_comp.add_subparsers(dest="compiler", required=True)

    p_sj = comp_sub.add_parser("simple-json", help="Compile simple JSON list")
    p_sj.add_argument("--input", required=True)
    p_sj.add_argument("--output", required=True)
    p_sj.add_argument("--manifest", default=None)
    p_sj.add_argument("--max-output-tokens", type=int, default=128)
    p_sj.add_argument("--temperature", type=float, default=0.0)
    p_sj.add_argument("--top-p", type=float, default=1.0)
    p_sj.add_argument("--streaming", action="store_true")
    p_sj.add_argument("--arrival", choices=["fixed-step", "poisson"], default="fixed-step")
    p_sj.add_argument("--arrival-step-ms", type=int, default=100)
    p_sj.add_argument("--rate-rps", type=float, default=None)
    p_sj.add_argument("--seed", type=int, default=None)
    p_sj.set_defaults(func=cmd_compile_simple_json)

    p_sh = comp_sub.add_parser("sharegpt", help="Compile ShareGPT-style JSON")
    p_sh.add_argument("--input", required=True)
    p_sh.add_argument("--output", required=True)
    p_sh.add_argument("--manifest", default=None)
    p_sh.add_argument("--mode", choices=["single-turn", "session"], default="single-turn")
    p_sh.add_argument("--user-tag", default="User")
    p_sh.add_argument("--assistant-tag", default="Assistant")
    p_sh.add_argument("--separator", default="\n")
    p_sh.add_argument("--max-output-tokens", type=int, default=128)
    p_sh.add_argument("--temperature", type=float, default=0.0)
    p_sh.add_argument("--top-p", type=float, default=1.0)
    p_sh.add_argument("--streaming", action="store_true")
    p_sh.add_argument("--arrival", choices=["fixed-step", "poisson"], default="fixed-step")
    p_sh.add_argument("--arrival-step-ms", type=int, default=100)
    p_sh.add_argument("--rate-rps", type=float, default=None)
    p_sh.add_argument("--seed", type=int, default=None)
    p_sh.set_defaults(func=cmd_compile_sharegpt)

    p_jl = comp_sub.add_parser("jsonl-prompts", help="Compile JSONL prompts")
    p_jl.add_argument("--input", required=True)
    p_jl.add_argument("--output", required=True)
    p_jl.add_argument("--manifest", default=None)
    p_jl.add_argument("--prompt-format", choices=["raw", "chatml", "openai_messages"], default="raw")
    p_jl.add_argument("--max-output-tokens", type=int, default=128)
    p_jl.add_argument("--temperature", type=float, default=0.0)
    p_jl.add_argument("--top-p", type=float, default=1.0)
    p_jl.add_argument("--streaming", action="store_true")
    p_jl.add_argument("--arrival", choices=["fixed-step", "poisson"], default="fixed-step")
    p_jl.add_argument("--arrival-step-ms", type=int, default=100)
    p_jl.add_argument("--rate-rps", type=float, default=None)
    p_jl.add_argument("--seed", type=int, default=None)
    p_jl.set_defaults(func=cmd_compile_jsonl_prompts)

    # validate
    p_val = sub.add_parser("validate", help="Validate workload JSONL against schema")
    p_val.add_argument("paths", nargs="+")
    p_val.set_defaults(func=cmd_validate)

    # profile validate
    p_prof = sub.add_parser("profile", help="Target profile utilities")
    prof_sub = p_prof.add_subparsers(dest="profile_cmd", required=True)
    p_pv = prof_sub.add_parser("validate", help="Validate a target profile")
    p_pv.add_argument("--profile", required=True)
    p_pv.set_defaults(func=cmd_profile_validate)

    # report
    p_rep = sub.add_parser("report", help="Generate workload summary report")
    p_rep.add_argument("--input", required=True)
    p_rep.add_argument("--format", choices=["text", "json"], default="text")
    p_rep.add_argument("--top-k-tags", type=int, default=10)
    p_rep.set_defaults(func=cmd_report)

    # fingerprint
    p_fp = sub.add_parser("fingerprint", help="Generate workload fingerprint")
    p_fp.add_argument("--input", required=True)
    p_fp.add_argument("--out", default=None)
    p_fp.add_argument("--extended", action="store_true")
    p_fp.set_defaults(func=cmd_fingerprint)

    # calibrate
    p_cal = sub.add_parser("calibrate", help="Calibrate engine/hardware for prediction")
    p_cal.add_argument("--profile", required=True)
    p_cal.add_argument("--host", default="localhost:8000")
    p_cal.add_argument("--api-key", default=None)
    p_cal.add_argument("--out", required=True)
    p_cal.add_argument("--prefill-points", type=int, default=None)
    p_cal.add_argument("--decode-targets", type=str, default=None)
    p_cal.set_defaults(func=cmd_calibrate)

    # export
    p_exp = sub.add_parser("export", help="Export workload to runner-specific formats")
    exp_sub = p_exp.add_subparsers(dest="target", required=True)
    p_ai = exp_sub.add_parser("aiperf", help="Export to aiperf trace format")
    p_ai.add_argument("--input", required=True)
    p_ai.add_argument("--output", required=True)
    p_ai.add_argument("--manifest", default=None)
    p_ai.add_argument("--time-mode", choices=["timestamp", "delay"], default="timestamp")
    p_ai.set_defaults(func=cmd_export_aiperf)

    # label
    p_lab = sub.add_parser("label", help="Label workload with semantic metadata")
    p_lab.add_argument("--input", required=True)
    p_lab.add_argument("--output", required=True)
    p_lab.add_argument("--overwrite", action="store_true")
    p_lab.set_defaults(func=cmd_label)

    # predict
    p_pred = sub.add_parser("predict", help="Predict latency for a workload")
    p_pred.add_argument("--input", required=True)
    p_pred.add_argument("--profile", required=True)
    p_pred.add_argument("--cal", required=True)
    p_pred.add_argument("--format", choices=["text", "json"], default="text")
    p_pred.add_argument("--concurrency", type=int, default=1)
    p_pred.add_argument("--target-utilization", type=float, default=0.7)
    p_pred.set_defaults(func=cmd_predict)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate predictor accuracy")
    p_eval.add_argument("--profile", required=True)
    p_eval.add_argument("--cal", required=True)
    p_eval.add_argument("--host", default="localhost:8000")
    p_eval.add_argument("--api-key", default=None)
    p_eval.add_argument("--inputs", required=True)
    p_eval.add_argument("--concurrency", default="1,2,4,8")
    p_eval.add_argument("--repeats", type=int, default=3)
    p_eval.add_argument("--ttft-sample-n", type=int, default=10)
    p_eval.add_argument("--timeout-s", type=int, default=120)
    p_eval.add_argument("--format", choices=["text", "json"], default="text")
    p_eval.add_argument("--out", default=None)
    p_eval.add_argument("--debug-sched", action="store_true")
    p_eval.add_argument("--bootstrap-samples", type=int, default=1000)
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()

    if args.config:
        cfg = _load_config_file(Path(args.config))
        for k, v in cfg.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)

    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n[Interrupted]", file=sys.stderr)
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception as e:
        if getattr(args, "verbose", False):
            import traceback
            traceback.print_exc()
        else:
            _print_error(str(e))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
