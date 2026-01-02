from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import jsonschema
import yaml


def _load_schema(schema_path: Path) -> dict[str, Any]:
    return json.loads(schema_path.read_text(encoding="utf-8"))


def load_profile(profile_path: Path) -> dict[str, Any]:
    obj = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit(f"{profile_path}: profile must be a YAML mapping/object")
    return obj


def validate_profile(profile: dict[str, Any], *, repo_root: Path) -> None:
    schema = _load_schema(repo_root / "schema" / "profile.schema.json")
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(profile), key=lambda e: list(e.path))
    if errors:
        msg = "\n".join(f"  - {list(err.path)}: {err.message}" for err in errors)
        raise SystemExit(f"profile schema validation failed:\n{msg}")


@dataclass(frozen=True)
class TargetProfile:
    engine: str
    model: str
    gpu_model: str
    gpu_memory_gb: float

    max_num_seqs: int
    max_num_batched_tokens: int
    dtype: str
    tensor_parallel: int
    pipeline_parallel: int
    kv_cache_mode: str


def parse_target_profile(profile: dict[str, Any]) -> TargetProfile:
    hw = profile["hardware"]
    v = profile["vllm"]

    return TargetProfile(
        engine=str(profile["engine"]),
        model=str(profile["model"]),
        gpu_model=str(hw["gpu_model"]),
        gpu_memory_gb=float(hw["gpu_memory_gb"]),
        max_num_seqs=int(v["max_num_seqs"]),
        max_num_batched_tokens=int(v["max_num_batched_tokens"]),
        dtype=str(v["dtype"]),
        tensor_parallel=int(v.get("tensor_parallel", 1)),
        pipeline_parallel=int(v.get("pipeline_parallel", 1)),
        kv_cache_mode=str(v.get("kv_cache_mode", "paged")),
    )


def load_and_validate_target_profile(profile_path: Path, *, repo_root: Path) -> TargetProfile:
    p = load_profile(profile_path)
    validate_profile(p, repo_root=repo_root)
    return parse_target_profile(p)
