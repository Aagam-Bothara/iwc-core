import json
from pathlib import Path

from iwc.fingerprint import build_fingerprint_from_report_json
from iwc.report import build_report, report_to_dict


def test_golden_fingerprint_session_chat_5turns() -> None:
    repo = Path(__file__).resolve().parents[2]
    inp = repo / "examples" / "session_chat_5turns.jsonl"
    golden = repo / "tests" / "golden" / "fingerprint_session_chat_5turns.golden.json"

    r = build_report(inp)
    fp, _ = build_fingerprint_from_report_json(report_to_dict(r, top_k_tags=0))

    got = json.dumps(fp, indent=2, sort_keys=True)
    exp = golden.read_text(encoding="utf-8").strip()
    assert got.strip() == exp
