from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Protocol


class Tokenizer(Protocol):
    def encode(self, text: str) -> List[int]: ...


@dataclass
class SimpleWhitespaceTokenizer:
    """
    Fallback tokenizer if tiktoken isn't installed.
    Not accurate, but gives consistent relative stats for quick analysis.
    """

    def encode(self, text: str) -> List[int]:
        parts = text.strip().split()
        return list(range(len(parts)))


def get_tokenizer(prefer: str = "tiktoken", model: str = "gpt-4o-mini") -> Tokenizer:
    """
    prefer:
      - "tiktoken": use tiktoken if installed
      - "simple": always use fallback
    model: only used when prefer="tiktoken"
    """
    if prefer == "simple":
        return SimpleWhitespaceTokenizer()

    if prefer == "tiktoken":
        try:
            import tiktoken  # type: ignore
        except Exception:
            return SimpleWhitespaceTokenizer()

        # Try model-specific encoding; fallback to cl100k_base
        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return enc

    raise ValueError(f"Unknown tokenizer preference: {prefer}")


def _as_messages(prompt: str) -> list[dict[str, Any]]:
    """
    Parse OpenAI messages JSON.
    Accepts:
      - a JSON list of {"role","content"} dicts
      - a JSON object with {"messages": [...]}
    Returns only well-formed {"role": str, "content": str} messages.
    """
    obj = json.loads(prompt)

    if isinstance(obj, dict) and isinstance(obj.get("messages"), list):
        msgs = obj["messages"]
    elif isinstance(obj, list):
        msgs = obj
    else:
        raise ValueError("openai_messages must be a JSON list or an object with 'messages'")

    out: list[dict[str, Any]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if isinstance(role, str) and isinstance(content, str):
            out.append({"role": role, "content": content})
    return out


def count_tokens_for_prompt(
    prompt: str,
    *,
    prompt_format: str = "raw",
    tokenizer_prefer: str = "tiktoken",
    tokenizer_model: str = "gpt-4o-mini",
) -> int:
    """
    Best-effort prompt token counting.

    - raw/chatml: tokenize the text directly (chatml treated as raw string)
    - openai_messages: parse JSON messages and tokenize a deterministic flattened form

    NOTE: This is not exact OpenAI wire-format token counting, but it is stable and
    good enough for workload fingerprinting + prediction features.
    """
    tok = get_tokenizer(prefer=tokenizer_prefer, model=tokenizer_model)

    pf = (prompt_format or "raw").strip()
    if pf in ("raw", "chatml"):
        return len(tok.encode(prompt))

    if pf == "openai_messages":
        msgs = _as_messages(prompt)
        text = ""
        for m in msgs:
            text += f"[{m['role']}]: {m['content']}\n"
        return len(tok.encode(text))

    # Unknown format => treat as raw
    return len(tok.encode(prompt))
