from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import yaml

from iwc.compile import _canonical_json_line


# -------------------------
# helpers
# -------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_export_manifest(
    *,
    exporter: str,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    manifest = {
        "iwc_version": _pkg_version("iwc"),
        "exporter": exporter,
        "generated_at_utc": _utc_now_iso(),
        "input": {"path": str(input_path), "sha256": _sha256_file(input_path)},
        "output": {"path": str(output_path), "sha256": _sha256_file(output_path)},
        "summary": summary,
        "config": config,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


@dataclass(frozen=True)
class ExportAiperfConfig:
    include_sampling_params: bool = False  # AIPerf SingleTurn doesn't define sampling fields
    time_mode: str = "timestamp"           # "timestamp" or "delay"
    role: str = "user"


def read_iwc_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            yield obj


def _require_int(name: str, v: Any, *, min_value: Optional[int] = None) -> int:
    if not isinstance(v, int):
        raise ValueError(f"field '{name}' must be int, got {type(v).__name__}")
    if min_value is not None and v < min_value:
        raise ValueError(f"field '{name}' must be >= {min_value}, got {v}")
    return v


def export_aiperf(
    input_workload: Path,
    output_trace: Path,
    manifest_path: Optional[Path] = None,
    cfg: Optional[ExportAiperfConfig] = None,
    source_manifest: Optional[Path] = None,   # NEW (optional)
    id_prefix: str = "req",                   # NEW
) -> Path:
    """
    Export IWC canonical workload JSONL -> aiperf-style trace JSONL.

    NOTE: This is a minimal mapping until we confirm aiperf's exact schema.
    Keeps unmapped fields in meta for lossless translation.
    """
    cfg = cfg or ExportAiperfConfig()
    manifest_path = manifest_path or Path(str(output_trace) + ".manifest.yaml")

    output_trace.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    min_at = None
    max_at = None

    prev_at: Optional[int] = None

    with output_trace.open("w", encoding="utf-8") as out:
        for req in read_iwc_jsonl(input_workload):
            prompt = req.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("field 'prompt' must be a non-empty string for aiperf single_turn")

            at = _require_int("arrival_time_ms", req.get("arrival_time_ms"), min_value=0)

            trace_req: Dict[str, Any] = {
                "type": "single_turn",
                "text": prompt,
                "role": cfg.role,
            }

            # ✅ Keep max_output_tokens when present (matches existing golden tests)
            mot = req.get("max_output_tokens")
            if mot is not None:
                trace_req["max_output_tokens"] = _require_int("max_output_tokens", mot, min_value=1)

            if cfg.time_mode == "timestamp":
                trace_req["timestamp"] = at
            elif cfg.time_mode == "delay":
                trace_req["delay"] = 0 if prev_at is None else (at - prev_at)
                prev_at = at
            else:
                raise ValueError(f"unknown time_mode: {cfg.time_mode}")

            out.write(_canonical_json_line(trace_req) + "\n")

            n += 1
            min_at = at if min_at is None else min(min_at, at)
            max_at = at if max_at is None else max(max_at, at)

    arrival_span_ms = int(max_at - min_at) if (min_at is not None and max_at is not None) else 0

    # Build config dict (add source manifest hash if provided)
    config = {"include_sampling_params": cfg.include_sampling_params}
    if source_manifest is not None:
        config["source_manifest"] = {"path": str(source_manifest), "sha256": _sha256_file(source_manifest)}

    _write_export_manifest(
        exporter="aiperf",
        input_path=input_workload,
        output_path=output_trace,
        manifest_path=manifest_path,
        summary={"num_requests": n, "arrival_span_ms": arrival_span_ms},
        config=config,
    )

    return manifest_path
