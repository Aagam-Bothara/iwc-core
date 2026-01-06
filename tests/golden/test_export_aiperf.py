import json
from pathlib import Path

from iwc.export import export_aiperf


def _read_json_objects(path: Path) -> list[dict]:
    """
    Robust reader that supports:
    - JSONL: one JSON object per line
    - Pretty JSON objects (multi-line), optionally multiple objects concatenated
    """
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    # Fast path: JSONL (each non-empty line is a JSON object)
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if lines and all(ln.lstrip().startswith("{") and ln.rstrip().endswith("}") for ln in lines):
        return [json.loads(ln) for ln in lines]

    # Fallback: stream-decode multiple JSON objects from one text blob
    dec = json.JSONDecoder()
    out: list[dict] = []
    i = 0
    n = len(raw)
    while i < n:
        # skip whitespace
        while i < n and raw[i].isspace():
            i += 1
        if i >= n:
            break
        obj, j = dec.raw_decode(raw, i)
        out.append(obj)
        i = j
    return out


def _assert_aiperf_semantically_equal(got_path: Path, exp_path: Path) -> None:
    got = _read_json_objects(got_path)
    exp = _read_json_objects(exp_path)
    assert got == exp


def test_export_aiperf_single_request_matches_golden(tmp_path: Path) -> None:
    inp = Path("examples/single_request.jsonl")
    golden = Path("tests/golden/aiperf_single_request.jsonl")

    out = tmp_path / "out_single.jsonl"
    export_aiperf(inp, out)

    _assert_aiperf_semantically_equal(out, golden)


def test_export_aiperf_bursty_10req_matches_golden(tmp_path: Path) -> None:
    inp = Path("examples/bursty_10req.jsonl")
    golden = Path("tests/golden/aiperf_bursty_10req.jsonl")

    out = tmp_path / "out_burst.jsonl"
    export_aiperf(inp, out)

    _assert_aiperf_semantically_equal(out, golden)


def test_export_aiperf_session_chat_5turns_matches_golden(tmp_path: Path) -> None:
    inp = Path("examples/session_chat_5turns.jsonl")
    golden = Path("tests/golden/aiperf_session_chat_5turns.jsonl")

    out = tmp_path / "out_session.jsonl"
    export_aiperf(inp, out)

    _assert_aiperf_semantically_equal(out, golden)
