import json
from pathlib import Path

from eval.rag_reliability_gate import evaluate_case_record, evaluate_repeat
from eval.rag_reliability_schema import load_reliability_suite


ROOT = Path(__file__).resolve().parents[2]
SUITE = load_reliability_suite(ROOT / "eval/fixtures/rag_reliability_v1.json")


def _span_for_window(window):
    return {
        "id": f"span::{window.video_id}::{round(window.start * 1000)}::{round(window.end * 1000)}",
        "video_id": window.video_id,
        "title": window.video_id,
        "start": window.start,
        "end": window.end,
        "text": window.excerpt,
        "retrieval_score": 1.0,
        "source_chunk_ids": [f"{window.video_id}::0"],
        "query_origins": ["original"],
    }


def test_positive_record_requires_status_verdicts_groups_and_exact_reconstruction():
    case = next(item for item in SUITE.cases if item.id == "O10")
    groups = {group.id: group for group in SUITE.evidence_groups}
    spans = [
        _span_for_window(groups[group_id].windows[0])
        for group_id in case.required_evidence_group_ids
    ]
    record = {
        "case_id": case.id,
        "decision": {
            "status": "answerable",
            "claims": [{"central": True, "verdict": "entailed"}],
            "selected_spans": spans,
            "hint_hits": [],
        },
    }
    result = evaluate_case_record(case, record, SUITE, ROOT)
    assert result.passed, result.failures
    broken = json.loads(json.dumps(record))
    broken["decision"]["selected_spans"] = []
    result = evaluate_case_record(case, broken, SUITE, ROOT)
    assert not result.passed
    assert any("evidence group" in failure for failure in result.failures)


def test_negative_error_does_not_pass_as_unsupported():
    case = next(item for item in SUITE.cases if item.id == "NT01")
    record = {
        "case_id": case.id,
        "decision": {"status": "error", "claims": [], "selected_spans": [], "hint_hits": []},
    }
    result = evaluate_case_record(case, record, SUITE, ROOT)
    assert not result.passed
    assert any("expected unsupported" in failure for failure in result.failures)


def test_card_chunk_or_card_take_in_evidence_fails():
    case = next(item for item in SUITE.cases if item.id == "O10")
    group = next(group for group in SUITE.evidence_groups if group.id in case.required_evidence_group_ids)
    span = _span_for_window(group.windows[0])
    span["source_chunk_ids"] = ["card::0001"]
    span["text"] = "CARD TAKE"
    record = {
        "case_id": case.id,
        "decision": {
            "status": "answerable",
            "claims": [{"central": True, "verdict": "entailed"}],
            "selected_spans": [span],
            "hint_hits": [{"take": "CARD TAKE"}],
        },
    }
    result = evaluate_case_record(case, record, SUITE, ROOT)
    assert not result.passed
    assert any("card" in failure.lower() for failure in result.failures)


def test_repeat_compares_status_and_required_group_coverage():
    baseline = {"status": "answerable", "covered_groups": ["a", "b"]}
    assert evaluate_repeat("O01", baseline, dict(baseline)) == ()
    assert evaluate_repeat(
        "O01", baseline, {"status": "partial", "covered_groups": ["a", "b"]},
    )
