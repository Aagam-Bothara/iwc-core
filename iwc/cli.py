from __future__ import annotations

import argparse
import json
from pathlib import Path
from iwc.profile import load_profile, validate_profile


import jsonschema
from iwc.fingerprint import build_fingerprint_from_report_json


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
from iwc.report import build_report, format_report, report_to_dict

# Diff
from iwc.diff.cli import add_diff_subcommand


def _canonical_json_line(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def cmd_label(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{in_path}:{line_no}: invalid JSON: {e}") from e

            if not isinstance(obj, dict):
                raise SystemExit(f"{in_path}:{line_no}: expected JSON object per line")

            labeled = label_record(obj, overwrite=args.overwrite)
            fout.write(_canonical_json_line(labeled) + "\n")
            n += 1

    print(f"Labeled {n} requests -> {out_path}")


def cmd_report(args: argparse.Namespace) -> None:
    r = build_report(Path(args.input))

    if args.format == "text":
        print(format_report(r, top_k_tags=args.top_k_tags))
        return

    obj = report_to_dict(r, top_k_tags=args.top_k_tags)
    print(json.dumps(obj, indent=2, sort_keys=True))

def cmd_fingerprint(args: argparse.Namespace) -> None:
    # Reuse report pipeline (already stable + tested)
    r = build_report(Path(args.input))
    rj = report_to_dict(r, top_k_tags=0)  # tags not needed for fingerprint

    fp, _ = build_fingerprint_from_report_json(rj)

    s = json.dumps(fp, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(s + "\n", encoding="utf-8")
    else:
        print(s)
def cmd_profile_validate(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    prof = load_profile(Path(args.profile))
    validate_profile(prof, repo_root=repo_root)
    print(f"OK  {args.profile}")

def load_schema(schema_path: Path) -> dict:
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_jsonl(jsonl_path: Path, schema: dict) -> None:
    validator = jsonschema.Draft202012Validator(schema)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{jsonl_path}:{line_no}: invalid JSON: {e}") from e

            errors = sorted(validator.iter_errors(obj), key=lambda e: list(e.path))
            if errors:
                msg = "\n".join(f"  - {list(err.path)}: {err.message}" for err in errors)
                raise SystemExit(f"{jsonl_path}:{line_no}: schema validation failed:\n{msg}")


def cmd_validate(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    schema = load_schema(repo_root / "schema" / "workload.schema.json")

    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            for jsonl in sorted(path.glob("*.jsonl")):
                validate_jsonl(jsonl, schema)
                print(f"OK  {jsonl}")
        else:
            validate_jsonl(path, schema)
            print(f"OK  {path}")


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
    print(f"Wrote workload: {args.output}")
    print(f"Wrote manifest: {manifest_path}")


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
    compile_jsonl_prompts(Path(args.input), out_path, manifest_path, cfg, prompt_format=args.prompt_format)
    print(f"Wrote workload: {args.output}")
    print(f"Wrote manifest: {manifest_path}")


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
    print(f"Wrote workload: {args.output}")
    print(f"Wrote manifest: {manifest_path}")


def cmd_export_aiperf(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest) if args.manifest else None
    cfg = ExportAiperfConfig(time_mode=args.time_mode)
    mp = export_aiperf(Path(args.input), Path(args.output), manifest_path=manifest_path, cfg=cfg)

    print(f"Wrote aiperf trace: {args.output}")
    print(f"Wrote manifest: {mp}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="iwc")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --------------------
    # diff (Diff Lite enhancements are handled inside iwc.diff.cli / iwc.diff.core)
    # --------------------
    add_diff_subcommand(sub)

    # --------------------
    # analyze
    # --------------------
    add_analyze_subcommand(sub)

    # --------------------
    # compile
    # --------------------
    p_comp = sub.add_parser("compile", help="Compile a dataset into canonical workload JSONL.")
    comp_sub = p_comp.add_subparsers(dest="compiler", required=True)

    # compile simple-json
    p_sj = comp_sub.add_parser("simple-json", help="Compile a simple JSON list into workload JSONL.")
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

    # compile sharegpt
    p_sh = comp_sub.add_parser("sharegpt", help="Compile ShareGPT-style JSON into workload JSONL.")
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

    # compile jsonl-prompts
    p_jl = comp_sub.add_parser("jsonl-prompts", help="Compile JSONL prompts into workload JSONL.")
    p_jl.add_argument("--input", required=True)
    p_jl.add_argument("--output", required=True)
    p_jl.add_argument("--manifest", default=None)

    p_jl.add_argument("--prompt-format", choices=["raw","chatml","openai_messages"], default="raw")
    p_jl.add_argument("--max-output-tokens", type=int, default=128)
    p_jl.add_argument("--temperature", type=float, default=0.0)
    p_jl.add_argument("--top-p", type=float, default=1.0)
    p_jl.add_argument("--streaming", action="store_true")

    p_jl.add_argument("--arrival", choices=["fixed-step", "poisson"], default="fixed-step")
    p_jl.add_argument("--arrival-step-ms", type=int, default=100)
    p_jl.add_argument("--rate-rps", type=float, default=None)
    p_jl.add_argument("--seed", type=int, default=None)
    p_jl.set_defaults(func=cmd_compile_jsonl_prompts)

    # --------------------
    # validate
    # --------------------
    p_val = sub.add_parser("validate", help="Validate workload JSONL against schema.")
    p_val.add_argument("paths", nargs="+")
    p_val.set_defaults(func=cmd_validate)
    p_prof = sub.add_parser("profile", help="Target profile utilities.")
    prof_sub = p_prof.add_subparsers(dest="profile_cmd", required=True)

    p_pv = prof_sub.add_parser("validate", help="Validate a target profile YAML.")
    p_pv.add_argument("--profile", required=True)
    p_pv.set_defaults(func=cmd_profile_validate)

    # --------------------
    # report
    # --------------------
    p_rep = sub.add_parser("report", help="Summarize a workload JSONL (text or json).")
    p_rep.add_argument("--input", required=True)
    p_rep.add_argument("--format", choices=["text", "json"], default="text")
    p_rep.add_argument("--top-k-tags", type=int, default=10)
    p_rep.set_defaults(func=cmd_report)
    # --------------------
    # fingerprint
    # --------------------
    p_fp = sub.add_parser("fingerprint", help="Export a stable workload fingerprint (JSON + deterministic hash).")
    p_fp.add_argument("--input", required=True, help="Workload JSONL")
    p_fp.add_argument("--out", default=None, help="Write fingerprint JSON to file (default: stdout)")
    p_fp.set_defaults(func=cmd_fingerprint)

    # --------------------
    # export
    # --------------------
    p_exp = sub.add_parser("export", help="Export canonical workload into runner-specific trace formats.")
    exp_sub = p_exp.add_subparsers(dest="target", required=True)

    p_ai = exp_sub.add_parser("aiperf", help="Export workload JSONL into aiperf-style trace JSONL.")
    p_ai.add_argument("--input", required=True)
    p_ai.add_argument("--output", required=True)
    p_ai.add_argument(
        "--manifest",
        default=None,
        help="Path to output export manifest YAML. Default: <output>.manifest.yaml",
    )
    p_ai.add_argument("--source-manifest", default=None, help="Compile manifest YAML to link for provenance.")
    p_ai.add_argument("--time-mode", choices=["timestamp", "delay"], default="timestamp")
    p_ai.set_defaults(func=cmd_export_aiperf)

    # --------------------
    # label
    # --------------------
    p_lab = sub.add_parser("label", help="Heuristically label workload JSONL with semantic.task/difficulty/tags.")
    p_lab.add_argument("--input", required=True)
    p_lab.add_argument("--output", required=True)
    p_lab.add_argument("--overwrite", action="store_true", help="Overwrite existing semantic fields.")
    p_lab.set_defaults(func=cmd_label)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
