from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml
from importlib.metadata import version as _pkg_version

from iwc.arrival import arrival_fixed_step, arrival_poisson


# -------------------------
# Shared helpers
# -------------------------
def _role_tag(role: str, cfg: ShareGPTConfig) -> str:
    return cfg.user_tag if role == "user" else cfg.assistant_tag

def _load_prompts_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Accepts JSONL where each line is either:
      - "prompt string"
      - {"prompt": "...", "semantic": {...}}
    Returns list of rows: {"prompt": str, "semantic": optional}
    """
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e

            if isinstance(obj, str):
                prompt = obj
                row: dict[str, Any] = {"prompt": prompt}
            elif isinstance(obj, dict) and isinstance(obj.get("prompt"), str):
                row = dict(obj)  # shallow copy
                prompt = row["prompt"]
            else:
                raise ValueError(
                    f"{path}:{line_no}: expected string or object with 'prompt' string"
                )

            if not str(prompt).strip():
                raise ValueError(f"{path}:{line_no}: empty/blank prompt")

            cleaned: dict[str, Any] = {"prompt": str(prompt)}
            if "semantic" in row:
                cleaned["semantic"] = row["semantic"]
            rows.append(cleaned)

    if not rows:
        raise ValueError("input JSONL contains 0 prompts")

    return rows


def compile_jsonl_prompts(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    cfg: SimpleJsonConfig,
    *,
    prompt_format: str = "raw",
) -> None:
    rows = _load_prompts_jsonl(input_path)
    n = len(rows)

    arrivals_ms = _arrival_times(n, cfg.arrival, cfg.arrival_step_ms, cfg.rate_rps, cfg.seed)
    arrival_span_ms = int(max(arrivals_ms) - min(arrivals_ms)) if arrivals_ms else 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, (row, at_ms) in enumerate(zip(rows, arrivals_ms), start=1):
            req = {
                "request_id": f"req-{idx:06d}",
                "prompt": row["prompt"],
                "prompt_format": prompt_format,
                "max_output_tokens": int(cfg.max_output_tokens),
                "arrival_time_ms": int(at_ms),
                "temperature": float(cfg.temperature),
                "top_p": float(cfg.top_p),
                "streaming": bool(cfg.streaming),
            }
            
            if "semantic" in row and row["semantic"] is not None:
                req["semantic"] = row["semantic"]
            f.write(_canonical_json_line(req) + "\n")

    _write_manifest(
        compiler="jsonl-prompts",
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        summary={"num_requests": n, "arrival_span_ms": arrival_span_ms},
        cfg={
            "prompt_format": prompt_format,
            "max_output_tokens": cfg.max_output_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "streaming": cfg.streaming,
            "arrival": cfg.arrival,
            "arrival_step_ms": cfg.arrival_step_ms,
            "rate_rps": cfg.rate_rps,
            "seed": cfg.seed,
        },
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json_line(obj: dict[str, Any]) -> str:
    # Stable hashing/diffing output
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _arrival_times(
    n: int,
    arrival: str,
    arrival_step_ms: int,
    rate_rps: Optional[float],
    seed: Optional[int],
) -> list[int]:
    if arrival == "fixed-step":
        return arrival_fixed_step(n, arrival_step_ms)
    if arrival == "poisson":
        if rate_rps is None:
            raise ValueError("rate_rps must be provided for poisson arrival model")
        return arrival_poisson(n, rate_rps, seed)
    raise ValueError(f"unknown arrival model: {arrival}")


def _write_manifest(
    *,
    compiler: str,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    cfg: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    schema_path = repo_root / "schema" / "workload.schema.json"

    manifest = {
        "iwc_version": _pkg_version("iwc"),
        "compiler": compiler,
        "generated_at_utc": _utc_now_iso(),
        "input": {"path": str(input_path), "sha256": _sha256_file(input_path)},
        "output": {"path": str(output_path), "sha256": _sha256_file(output_path)},
        "schema": {"path": "schema/workload.schema.json", "sha256": _sha256_file(schema_path)},
        "summary": summary,
        "config": cfg,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


# -------------------------
# simple-json compiler
# -------------------------


@dataclass(frozen=True)
class SimpleJsonConfig:
    max_output_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    streaming: bool = False

    arrival: str = "fixed-step"          # "fixed-step" | "poisson"
    arrival_step_ms: int = 100           # for fixed-step
    rate_rps: Optional[float] = None     # for poisson
    seed: Optional[int] = None           # for poisson randomness


def _load_simple_json(path: Path) -> list[dict[str, Any]]:
    """
    Accepts:
      - ["prompt1", "prompt2", ...]
      - [{"prompt": "...", "semantic": {...}}, ...]
    Returns list of rows containing:
      - prompt: str
      - semantic: optional (passthrough)
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("simple-json input must be a JSON list")

    rows: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            prompt = item
            row: dict[str, Any] = {"prompt": prompt}
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            row = dict(item)  # shallow copy
            prompt = row["prompt"]
        else:
            raise ValueError(
                f"invalid item at index {i}: expected string or object with 'prompt' string"
            )

        if not str(prompt).strip():
            raise ValueError(f"empty/blank prompt at index {i}")

        cleaned: dict[str, Any] = {"prompt": str(prompt)}
        if "semantic" in row:
            cleaned["semantic"] = row["semantic"]

        rows.append(cleaned)

    if not rows:
        raise ValueError("input dataset contains 0 prompts")

    return rows


def compile_simple_json(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    cfg: SimpleJsonConfig,
    *,
    prompt_format: str = "raw",
) -> None:
    rows = _load_simple_json(input_path)
    n = len(rows)

    arrivals_ms = _arrival_times(n, cfg.arrival, cfg.arrival_step_ms, cfg.rate_rps, cfg.seed)
    arrival_span_ms = int(max(arrivals_ms) - min(arrivals_ms)) if arrivals_ms else 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, (row, at_ms) in enumerate(zip(rows, arrivals_ms), start=1):
            req = {
                "request_id": f"req-{idx:06d}",
                "prompt": row["prompt"],
                "prompt_format": prompt_format,
                "max_output_tokens": int(cfg.max_output_tokens),
                "arrival_time_ms": int(at_ms),
                "temperature": float(cfg.temperature),
                "top_p": float(cfg.top_p),
                "streaming": bool(cfg.streaming),
            }
            if "semantic" in row and row["semantic"] is not None:
                req["semantic"] = row["semantic"]

            f.write(_canonical_json_line(req) + "\n")

    _write_manifest(
        compiler="simple-json",
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        summary={"num_requests": n, "arrival_span_ms": arrival_span_ms},
        cfg={
            "prompt_format": prompt_format,
            "max_output_tokens": cfg.max_output_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "streaming": cfg.streaming,
            "arrival": cfg.arrival,
            "arrival_step_ms": cfg.arrival_step_ms,
            "rate_rps": cfg.rate_rps,
            "seed": cfg.seed,
        },
    )


# -------------------------
# ShareGPT compiler
# -------------------------


@dataclass(frozen=True)
class ShareGPTConfig:
    """
    mode:
      - single-turn: take first human/user message as prompt
      - session: build a multi-turn transcript into a single prompt
    """
    mode: str = "single-turn"  # "single-turn" | "session"
    user_tag: str = "User"
    assistant_tag: str = "Assistant"
    separator: str = "\n"

    prompt_format: str = "raw"

    max_output_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    streaming: bool = False

    arrival: str = "fixed-step"
    arrival_step_ms: int = 100
    rate_rps: Optional[float] = None
    seed: Optional[int] = None


def _extract_sharegpt_turns(obj: dict[str, Any]) -> list[tuple[str, str]]:
    """
    Returns list of (role, text) where role is 'user' or 'assistant'.
    Supports common ShareGPT variants:
      - {"conversations":[{"from":"human","value":"..."}, {"from":"gpt","value":"..."}]}
      - {"conversations":[{"from":"user","value":"..."}, {"from":"assistant","value":"..."}]}
      - {"messages":[{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]}
    """
    turns: list[tuple[str, str]] = []

    if isinstance(obj.get("conversations"), list):
        conv = obj["conversations"]
        for m in conv:
            if not isinstance(m, dict):
                continue
            frm = m.get("from")
            val = m.get("value")
            if not isinstance(val, str):
                continue

            role: Optional[str] = None
            if frm in ("human", "user"):
                role = "user"
            elif frm in ("gpt", "assistant"):
                role = "assistant"

            if role and val.strip():
                turns.append((role, val))
        return turns

    if isinstance(obj.get("messages"), list):
        msgs = obj["messages"]
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                turns.append((role, content))
        return turns

    return turns


def _sharegpt_prompt_from_turns(turns: list[tuple[str, str]], cfg: ShareGPTConfig) -> str:
    if cfg.mode == "single-turn":
        for role, text in turns:
            if role == "user":
                return text.strip()
        return ""

    if cfg.mode == "session":
        lines: list[str] = []
        for role, text in turns:
            tag = cfg.user_tag if role == "user" else cfg.assistant_tag
            lines.append(f"{tag}: {text.strip()}")
        return cfg.separator.join(lines).strip()

    raise ValueError(f"unknown sharegpt mode: {cfg.mode}")


def _extract_conversation_id(obj: dict[str, Any], idx: int) -> str:
    """
    Stable session id per ShareGPT record.
    Uses common fields if present; otherwise falls back to deterministic index-based id.
    """
    raw = obj.get("conversation_id") or obj.get("id") or obj.get("uuid")
    if raw is None:
        return f"conv-{idx:06d}"
    s = str(raw).strip()
    return s if s else f"conv-{idx:06d}"


def compile_sharegpt(
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    cfg: ShareGPTConfig,
) -> None:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("sharegpt input must be a JSON list")

    requests: list[dict[str, Any]] = []
    skipped_records = 0

    # Build one request per assistant turn.
    # Prompt is the transcript up to (and including) the user message that precedes that assistant turn.
    for rec_idx, obj in enumerate(data, start=1):
        if not isinstance(obj, dict):
            skipped_records += 1
            continue

        turns = _extract_sharegpt_turns(obj)
        if not turns:
            skipped_records += 1
            continue

        session_id = _extract_conversation_id(obj, rec_idx)

        if cfg.mode == "single-turn":
            # Keep existing behavior: 1 request per record, first user message.
            prompt = _sharegpt_prompt_from_turns(turns, cfg)
            if not prompt.strip():
                skipped_records += 1
                continue

            requests.append({
                "session_id": session_id,  # harmless even in single-turn
                "prompt": prompt.strip(),
            })
            continue

        if cfg.mode != "session":
            raise ValueError(f"unknown sharegpt mode: {cfg.mode}")

        # Session mode: one request per assistant turn with growing context.
        transcript_lines: list[str] = []
        have_user_since_last_assistant = False

        for role, text in turns:
            text = (text or "").strip()
            if not text:
                continue

            tag = _role_tag(role, cfg)

            if role == "user":
                transcript_lines.append(f"{tag}: {text}")
                have_user_since_last_assistant = True
                continue

            # role == assistant
            # Only emit if we actually have a user turn to respond to.
            if not have_user_since_last_assistant:
                # Skip assistant messages that have no preceding user message in our transcript.
                transcript_lines.append(f"{tag}: {text}")
                continue

            # Emit request BEFORE appending this assistant message, since the model hasn't generated it yet.
            prompt = cfg.separator.join(transcript_lines).strip()
            if prompt:
                requests.append({
                    "session_id": session_id,
                    "prompt": prompt,
                })

            # Now append assistant text to transcript (for subsequent turns' context growth)
            transcript_lines.append(f"{tag}: {text}")
            have_user_since_last_assistant = False

    if not requests:
        raise ValueError("sharegpt input produced 0 requests (check format / mode)")

    n = len(requests)
    arrivals_ms = _arrival_times(n, cfg.arrival, cfg.arrival_step_ms, cfg.rate_rps, cfg.seed)
    arrival_span_ms = int(max(arrivals_ms) - min(arrivals_ms)) if arrivals_ms else 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for i, (row, at_ms) in enumerate(zip(requests, arrivals_ms), start=1):
            req: dict[str, Any] = {
                "request_id": f"req-{i:06d}",
                "prompt": row["prompt"],
                "prompt_format": cfg.prompt_format,
                "max_output_tokens": int(cfg.max_output_tokens),
                "arrival_time_ms": int(at_ms),
                "temperature": float(cfg.temperature),
                "top_p": float(cfg.top_p),
                "streaming": bool(cfg.streaming),
                "session_id": row["session_id"],
                "semantic": {"task": "chat", "tags": ["sharegpt", "session", "turn-level"]},
            }
            f.write(_canonical_json_line(req) + "\n")

    _write_manifest(
        compiler="sharegpt",
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        summary={
            "num_requests": n,
            "arrival_span_ms": arrival_span_ms,
            "skipped_records": skipped_records,
            "mode": cfg.mode,
            "emission": "assistant-turn",
        },
        cfg={
            "mode": cfg.mode,
            "user_tag": cfg.user_tag,
            "assistant_tag": cfg.assistant_tag,
            "separator": cfg.separator.encode("unicode_escape").decode("ascii"),
            "prompt_format": cfg.prompt_format,
            "max_output_tokens": cfg.max_output_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "streaming": cfg.streaming,
            "arrival": cfg.arrival,
            "arrival_step_ms": cfg.arrival_step_ms,
            "rate_rps": cfg.rate_rps,
            "seed": cfg.seed,
        },
    )
