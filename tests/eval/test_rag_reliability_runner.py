import json
from pathlib import Path

import pytest

from eval.rag_reliability import smoke_server
from eval.rag_reliability_runner import RunIdentityMismatch, run_suite
from eval.rag_reliability_schema import load_reliability_suite
from pipeline.config import load_settings
from rag.evidence_pipeline import EvidenceDecision


ROOT = Path(__file__).resolve().parents[2]
SUITE_PATH = ROOT / "eval/fixtures/rag_reliability_v1.json"


def _unsupported(*args, **kwargs):
    return EvidenceDecision("unsupported", (), (), (), None, (), None, {"fake": True})


def test_runner_writes_atomic_case_and_repeat_records(tmp_path):
    suite = load_reliability_suite(SUITE_PATH)
    run_suite(
        suite=suite,
        suite_path=SUITE_PATH,
        settings=load_settings(),
        run_dir=tmp_path,
        assess_fn=_unsupported,
        verifier_factory=lambda config: object(),
        runtime_identity_fn=lambda config: {"ollama_version": "test", "model_digest": "digest"},
        resume=False,
    )
    assert (tmp_path / "run.json").is_file()
    assert len(list((tmp_path / "cases").glob("*.json"))) == 55
    assert len(list((tmp_path / "repeats").glob("*.json"))) == 6
    assert not list(tmp_path.rglob("*.tmp-*"))
    record = json.loads((tmp_path / "cases/NT01.json").read_text(encoding="utf-8"))
    assert record["decision"]["status"] == "unsupported"


def test_resume_skips_complete_records_only_under_identical_identity(tmp_path):
    suite = load_reliability_suite(SUITE_PATH)
    calls = []
    kwargs = dict(
        suite=suite,
        suite_path=SUITE_PATH,
        settings=load_settings(),
        run_dir=tmp_path,
        assess_fn=lambda *args, **kw: (calls.append(args[0]) or _unsupported()),
        verifier_factory=lambda config: object(),
        runtime_identity_fn=lambda config: {"ollama_version": "test", "model_digest": "digest"},
    )
    run_suite(**kwargs, resume=False)
    first_count = len(calls)
    run_suite(**kwargs, resume=True)
    assert len(calls) == first_count


def test_resume_rejects_identity_drift(tmp_path):
    suite = load_reliability_suite(SUITE_PATH)
    common = dict(
        suite=suite,
        suite_path=SUITE_PATH,
        settings=load_settings(),
        run_dir=tmp_path,
        assess_fn=_unsupported,
        verifier_factory=lambda config: object(),
    )
    run_suite(
        **common,
        runtime_identity_fn=lambda config: {"ollama_version": "a", "model_digest": "one"},
        resume=False,
    )
    with pytest.raises(RunIdentityMismatch):
        run_suite(
            **common,
            runtime_identity_fn=lambda config: {"ollama_version": "b", "model_digest": "two"},
            resume=True,
        )


@pytest.mark.parametrize("identity_state", ["missing", "invalid"])
def test_resume_refuses_existing_records_without_a_valid_run_identity(tmp_path, identity_state):
    suite = load_reliability_suite(SUITE_PATH)
    common = dict(
        suite=suite,
        suite_path=SUITE_PATH,
        settings=load_settings(),
        run_dir=tmp_path,
        verifier_factory=lambda config: object(),
    )
    run_suite(
        **common,
        assess_fn=_unsupported,
        runtime_identity_fn=lambda config: {"ollama_version": "a", "model_digest": "one"},
        resume=False,
    )
    identity_path = tmp_path / "run.json"
    if identity_state == "missing":
        identity_path.unlink()
    else:
        identity_path.write_text("not valid JSON", encoding="utf-8")
    calls = []
    with pytest.raises(RunIdentityMismatch):
        run_suite(
            **common,
            assess_fn=lambda *args, **kwargs: (calls.append(args[0]) or _unsupported()),
            runtime_identity_fn=lambda config: {"ollama_version": "b", "model_digest": "two"},
            resume=True,
        )
    assert calls == []


def test_server_smoke_never_calls_answer_model_for_unsupported(monkeypatch, tmp_path):
    suite = load_reliability_suite(SUITE_PATH)
    evidence = {
        "id": "span::video::0::1000",
        "video_id": "video",
        "title": "Video",
        "start": 0.0,
        "end": 1.0,
        "text": "Supported evidence.",
        "retrieval_score": 1.0,
        "score": 1.0,
        "source_chunk_ids": ["video::0"],
        "query_origins": ["original"],
    }
    expected_statuses = {case.question: case.expected_status for case in suite.smoke_cases}

    def retrieve(question, mode, *, url):
        if expected_statuses[question] == "unsupported":
            return {
                "schema_version": 2, "status": "unsupported", "grounded": False,
                "failure_reason": None, "claims": [], "evidence_hits": [], "hint_hits": [],
                "answer_question": None, "unsupported_claims": [],
            }
        if mode == "clean_with_card_hints":
            return {
                "schema_version": 2, "status": "partial", "grounded": True,
                "failure_reason": None,
                "claims": [
                    {
                        "claim_id": "c1", "claim": "Supported fact", "reason": "evidence",
                        "central": True, "verdict": "entailed", "evidence_ids": [evidence["id"]],
                    },
                    {
                        "claim_id": "c2", "claim": "Withheld assertion", "reason": "missing",
                        "central": True, "verdict": "not_found", "evidence_ids": [],
                    },
                ],
                "evidence_hits": [evidence], "hint_hits": [], "answer_question": "Supported fact?",
                "unsupported_claims": ["Withheld assertion"],
            }
        return {
            "schema_version": 2, "status": "answerable", "grounded": True,
            "failure_reason": None,
            "claims": [{
                "claim_id": "c1", "claim": "Supported fact", "reason": "evidence",
                "central": True, "verdict": "entailed", "evidence_ids": [evidence["id"]],
            }],
            "evidence_hits": [evidence], "hint_hits": [], "answer_question": "Supported fact?",
            "unsupported_claims": [],
        }

    monkeypatch.setattr("eval.rag_reliability.retrieve_decision", retrieve)
    output = tmp_path / "server-smoke.json"
    assert smoke_server(suite, "http://test", output)
    result = json.loads(output.read_text(encoding="utf-8"))
    rows = {row["id"]: row for row in result["rows"]}
    assert rows["SMOKE-UNSUPPORTED"]["answer_calls"] == 0
    assert sum(row["answer_calls"] for row in result["rows"]) == 3
