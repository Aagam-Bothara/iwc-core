# iwc/analyze/cli.py
from __future__ import annotations

import argparse

from iwc.analyze.read_jsonl import iter_requests_jsonl
from iwc.analyze.summary import build_summary, render_summary


def _cmd_analyze(args: argparse.Namespace) -> int:
    reqs = list(iter_requests_jsonl(args.trace))
    summ = build_summary(reqs, tokenizer_prefer=args.tokenizer, tokenizer_model=args.tokenizer_model)
    print(render_summary(summ))
    return 0


def add_analyze_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("analyze", help="Analyze a workload JSONL trace")
    p.add_argument("trace", help="Path to workload JSONL")
    p.add_argument("--tokenizer", choices=["tiktoken", "simple"], default="tiktoken")
    p.add_argument("--tokenizer-model", default="gpt-4o-mini")
    p.set_defaults(func=_cmd_analyze)
