from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path

import pytest

import eval.card_eval as card_eval_module
from eval.card_eval import (
    _build_unit_spec,
    _ensure_run_manifest,
    _execute_unit,
    expand_stage,
    generate_report,
    load_review_file,
    result_path,
    review_path,
    write_atomic_json,
)
from eval.card_eval_ollama import GenerationResult, ModelIdentity
from eval.card_eval_schema import load_suite
from eval.card_eval_score import AnswerEvaluation, AnswerKey, MachineDiagnostics


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_PATH = REPO_ROOT / "eval" / "fixtures" / "verified_card_eval_v1.json"
DIGEST = "bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8"


@pytest.fixture(scope="module")
def suite():
    return load_suite(SUITE_PATH)


def _identity():
    return ModelIdentity(
        name="qwen3:14b",
        digest=DIGEST,
        ollama_version="0.31.2",
        format="gguf",
        family="qwen3",
        parameter_size="14.8B",
        quantization_level="Q4_K_M",
        file_type=15,
        quantization_version=2,
    )


class FixedClient:
    def __init__(self, identity):
        self.identity = identity

    def generate(self, messages, settings):
        return GenerationResult(
            content="doğru önerme",
            model=self.identity.name,
            done=True,
            done_reason="stop",
            prompt_eval_count=100,
            eval_count=5,
        )


def _review_json(case, key, *, style=None):
    required = [atom.id for atom in case.atoms if atom.required]
    return {
        "question_id": key.question_id,
        "stage": key.stage,
        "arm": key.arm,
        "seed": key.seed,
        "claim_labels": [
            {
                "answer_span": "doğru önerme",
                "label": "entailed",
                "atom_ids": required,
            }
        ],
        "critical_failures": [],
        "required_atom_ids_present": required,
        "supported_claim_precision": 1.0,
        "coherence": 3,
        "style_fidelity": style,
        "reviewer": "codex-semantic-review",
        "reviewed_at": "2026-07-11T12:00:00+03:00",
    }


def _evaluation(case, key):
    return AnswerEvaluation(
        key=key,
        answer="doğru önerme",
        result_cache_key="a" * 64,
        case_sha256=case.sha256,
        diagnostics=MachineDiagnostics(
            empty_answer=False,
            corrupt_unicode=False,
            think_leak=False,
            token_cap_stop=False,
            natural_stop=True,
        ),
    )


def test_review_loader_derives_fixture_atoms_and_binds_exact_answer(
    suite, tmp_path
):
    case = suite.cases[0]
    key = AnswerKey("fixed-diagnostic", case.id, "semantic_only", 20260711)
    evaluation = _evaluation(case, key)
    path = tmp_path / "review.json"
    write_atomic_json(path, _review_json(case, key))

    review = load_review_file(
        path, case=case, expected_key=key, evaluation=evaluation
    )
    assert review.required_atom_ids_expected == frozenset(
        atom.id for atom in case.atoms if atom.required
    )
    assert review.answer_sha256 == evaluation.answer_sha256
    assert review.result_cache_key == evaluation.result_cache_key


@pytest.mark.parametrize(
    "mutation",
    ["extra_field", "unknown_atom", "bad_span", "bad_timestamp", "style_on_semantic"],
)
def test_review_loader_fails_closed_on_malformed_or_stale_semantics(
    suite, tmp_path, mutation
):
    case = suite.cases[0]
    key = AnswerKey("fixed-diagnostic", case.id, "semantic_only", 20260711)
    evaluation = _evaluation(case, key)
    raw = _review_json(case, key)
    if mutation == "extra_field":
        raw["all_required_atoms_present"] = True
    elif mutation == "unknown_atom":
        raw["claim_labels"][0]["atom_ids"].append("invented")
    elif mutation == "bad_span":
        raw["claim_labels"][0]["answer_span"] = "yanıtta yok"
    elif mutation == "bad_timestamp":
        raw["reviewed_at"] = "2026-07-11T12:00:00"
    else:
        raw["style_fidelity"] = 3
    path = tmp_path / f"{mutation}.json"
    write_atomic_json(path, raw)

    with pytest.raises(ValueError):
        load_review_file(path, case=case, expected_key=key, evaluation=evaluation)


def test_unsupported_claim_label_requires_critical_failure(suite, tmp_path):
    case = suite.cases[0]
    key = AnswerKey("fixed-diagnostic", case.id, "semantic_plus_style", 20260711)
    evaluation = _evaluation(case, key)
    raw = _review_json(case, key)
    raw["claim_labels"][0]["label"] = "unsupported"
    raw["claim_labels"][0]["atom_ids"] = []
    raw["required_atom_ids_present"] = []
    raw["supported_claim_precision"] = 0.0
    path = tmp_path / "unsupported.json"
    write_atomic_json(path, raw)

    with pytest.raises(ValueError, match="unsupported_claim"):
        load_review_file(path, case=case, expected_key=key, evaluation=evaluation)


def _prepare_complete_stage(run_dir: Path, suite, stage: str) -> None:
    identity = _identity()
    _ensure_run_manifest(run_dir / "run.json", suite, SUITE_PATH, identity)
    for unit in expand_stage(suite, stage):
        destination = result_path(
            run_dir,
            stage,
            unit.case.id,
            unit.arm,
            unit.seed,
        )
        spec = _build_unit_spec(
            suite=suite,
            case=unit.case,
            stage=stage,
            arm=unit.arm,
            seed=unit.seed,
            identity=identity,
            settings=unit.settings,
            destination=destination,
        )
        _execute_unit(spec, FixedClient(identity), resume=True)
        key = AnswerKey(
            stage, unit.case.id, unit.arm, unit.seed
        )
        write_atomic_json(
            review_path(run_dir, key), _review_json(unit.case, key)
        )


def _prepare_complete_diagnostic(run_dir: Path, suite) -> None:
    _prepare_complete_stage(run_dir, suite, "fixed-diagnostic")


@pytest.fixture(scope="module")
def complete_diagnostic_dir(tmp_path_factory, suite):
    run_dir = tmp_path_factory.mktemp("complete-diagnostic")
    _prepare_complete_diagnostic(run_dir, suite)
    return run_dir


def _copy_run(source: Path, tmp_path: Path) -> Path:
    destination = tmp_path / "run"
    shutil.copytree(source, destination)
    return destination


def test_report_locks_semantics_then_allows_style_without_changing_lock(
    complete_diagnostic_dir, suite, tmp_path
):
    run_dir = _copy_run(complete_diagnostic_dir, tmp_path)
    first = generate_report(run_dir)
    stage = first["stages"][0]
    assert stage["id"] == "fixed-diagnostic"
    assert stage["semantic"]["status"] == "passed"
    assert stage["semantic"]["full_passes"] == 30
    assert stage["style"]["status"] == "unreviewed"

    lock_path = run_dir / "reviews" / "fixed-diagnostic" / "semantic-lock.json"
    assert lock_path.exists()
    lock_bytes = lock_path.read_bytes()
    summary_bytes = (run_dir / "summary.json").read_bytes()
    report_bytes = (run_dir / "report.md").read_bytes()
    assert generate_report(run_dir) == first
    assert (run_dir / "summary.json").read_bytes() == summary_bytes
    assert (run_dir / "report.md").read_bytes() == report_bytes

    cases = {case.id: case for case in suite.cases}
    for unit in expand_stage(suite, "fixed-diagnostic"):
        if unit.arm != "semantic_plus_style":
            continue
        key = AnswerKey(
            "fixed-diagnostic", unit.case.id, unit.arm, unit.seed
        )
        path = review_path(run_dir, key)
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["style_fidelity"] = 2
        write_atomic_json(path, raw)

    styled = generate_report(run_dir)
    assert styled["stages"][0]["semantic"]["status"] == "passed"
    assert styled["stages"][0]["style"]["status"] == "passed"
    assert lock_path.read_bytes() == lock_bytes

    changed_key = AnswerKey(
        "fixed-diagnostic", suite.cases[0].id, "semantic_plus_style", 20260711
    )
    changed_path = review_path(run_dir, changed_key)
    changed = json.loads(changed_path.read_text(encoding="utf-8"))
    changed["coherence"] = 2
    write_atomic_json(changed_path, changed)
    stale = generate_report(run_dir)
    assert stale["stages"][0]["semantic"]["status"] == "blocked"


def test_missing_review_is_incomplete_but_invalid_review_is_blocked(
    complete_diagnostic_dir, suite, tmp_path
):
    missing_dir = _copy_run(complete_diagnostic_dir, tmp_path / "missing")
    key = AnswerKey(
        "fixed-diagnostic", suite.cases[0].id, "semantic_only", 20260711
    )
    review_path(missing_dir, key).unlink()
    missing = generate_report(missing_dir)
    assert missing["stages"][0]["semantic"]["status"] == "incomplete"
    answer = next(
        item
        for item in missing["stages"][0]["answers"]
        if item["question_id"] == key.question_id and item["arm"] == key.arm
    )
    assert "unreviewed" in answer["failure_reasons"]

    invalid_dir = _copy_run(complete_diagnostic_dir, tmp_path / "invalid")
    invalid_path = review_path(invalid_dir, key)
    raw = json.loads(invalid_path.read_text(encoding="utf-8"))
    raw["reviewed_at"] = "not-a-time"
    write_atomic_json(invalid_path, raw)
    invalid = generate_report(invalid_dir)
    assert invalid["stages"][0]["semantic"]["status"] == "blocked"


def test_locked_missing_review_stays_incomplete_and_turkish_round_trips(
    complete_diagnostic_dir, suite, tmp_path
):
    run_dir = _copy_run(complete_diagnostic_dir, tmp_path)
    first = generate_report(run_dir)
    answer = first["stages"][0]["answers"][0]
    assert answer["answer"] == "doğru önerme"
    assert "doğru önerme" in (run_dir / "summary.json").read_text(
        encoding="utf-8"
    )

    key = AnswerKey(
        "fixed-diagnostic", suite.cases[0].id, "semantic_only", 20260711
    )
    review_path(run_dir, key).unlink()
    incomplete = generate_report(run_dir)
    assert incomplete["stages"][0]["semantic"]["status"] == "incomplete"


def test_style_before_first_semantic_lock_blocks_without_creating_lock(
    complete_diagnostic_dir, suite, tmp_path
):
    run_dir = _copy_run(complete_diagnostic_dir, tmp_path)
    key = AnswerKey(
        "fixed-diagnostic",
        suite.cases[0].id,
        "semantic_plus_style",
        20260711,
    )
    path = review_path(run_dir, key)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["style_fidelity"] = 3
    write_atomic_json(path, raw)

    summary = generate_report(run_dir)
    assert summary["stages"][0]["semantic"]["status"] == "blocked"
    assert not (
        run_dir / "reviews" / "fixed-diagnostic" / "semantic-lock.json"
    ).exists()


def test_tampered_semantic_lock_and_orphan_artifacts_block(
    complete_diagnostic_dir, tmp_path
):
    tampered_dir = _copy_run(complete_diagnostic_dir, tmp_path / "tampered")
    generate_report(tampered_dir)
    lock_path = (
        tampered_dir
        / "reviews"
        / "fixed-diagnostic"
        / "semantic-lock.json"
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["entries"][0]["semantic_review_sha256"] = "b" * 64
    write_atomic_json(lock_path, lock)
    tampered = generate_report(tampered_dir)
    assert tampered["stages"][0]["semantic"]["status"] == "blocked"
    assert "semantic lock mismatch" in (
        tampered_dir / "report.md"
    ).read_text(encoding="utf-8")

    orphan_dir = _copy_run(complete_diagnostic_dir, tmp_path / "orphan")
    write_atomic_json(
        orphan_dir / "fixed" / "diagnostic" / "orphan.json", {}
    )
    orphan = generate_report(orphan_dir)
    assert orphan["stages"][0]["semantic"]["status"] == "blocked"
    assert any(
        "unexpected result artifact" in reason
        for reason in orphan["stages"][0]["semantic"]["reasons"]
    )
    assert "unexpected result artifact" in (
        orphan_dir / "report.md"
    ).read_text(encoding="utf-8")


def test_semantic_lock_create_once_preserves_interleaving_winner(
    complete_diagnostic_dir, monkeypatch, tmp_path
):
    run_dir = _copy_run(complete_diagnostic_dir, tmp_path)
    original = card_eval_module._write_atomic_json_exclusive

    def competing_writer(path, value):
        winner = deepcopy(value)
        winner["entries"][0]["semantic_review_sha256"] = "b" * 64
        write_atomic_json(path, winner)
        return original(path, value)

    monkeypatch.setattr(
        card_eval_module, "_write_atomic_json_exclusive", competing_writer
    )
    summary = generate_report(run_dir)
    stage = summary["stages"][0]
    assert stage["semantic"]["status"] == "blocked"
    lock = json.loads(
        (
            run_dir
            / "reviews"
            / "fixed-diagnostic"
            / "semantic-lock.json"
        ).read_text(encoding="utf-8")
    )
    assert lock["entries"][0]["semantic_review_sha256"] == "b" * 64


def test_critical_failure_is_failed_not_blocked_and_keeps_exact_reasons(
    complete_diagnostic_dir, suite, tmp_path
):
    run_dir = _copy_run(complete_diagnostic_dir, tmp_path)
    key = AnswerKey(
        "fixed-diagnostic", suite.cases[0].id, "semantic_only", 20260711
    )
    path = review_path(run_dir, key)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["claim_labels"][0]["label"] = "unsupported"
    raw["claim_labels"][0]["atom_ids"] = []
    raw["critical_failures"] = ["unsupported_claim"]
    raw["required_atom_ids_present"] = []
    raw["supported_claim_precision"] = 0.0
    write_atomic_json(path, raw)

    summary = generate_report(run_dir)
    stage = summary["stages"][0]
    assert stage["semantic"]["status"] == "failed"
    answer = next(
        item
        for item in stage["answers"]
        if item["question_id"] == key.question_id and item["arm"] == key.arm
    )
    assert "critical:unsupported_claim" in answer["failure_reasons"]
    assert "claim:unsupported:doğru önerme" in answer["failure_reasons"]
    assert "critical:unsupported_claim" in (run_dir / "report.md").read_text(
        encoding="utf-8"
    )


@pytest.fixture(scope="module")
def complete_both_stages_dir(tmp_path_factory, suite):
    run_dir = tmp_path_factory.mktemp("complete-both-stages")
    _prepare_complete_diagnostic(run_dir, suite)
    assert generate_report(run_dir)["stages"][0]["semantic"]["status"] == "passed"
    _prepare_complete_stage(run_dir, suite, "fixed-production")
    return run_dir


def _set_production_coherence(run_dir: Path, keys, value: int) -> None:
    for key in keys:
        path = review_path(run_dir, key)
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["coherence"] = value
        write_atomic_json(path, raw)


def test_production_gate_accepts_exactly_42_of_45_with_two_per_question(
    complete_both_stages_dir, suite, tmp_path
):
    run_dir = _copy_run(complete_both_stages_dir, tmp_path)
    failures = [
        AnswerKey(
            "fixed-production",
            suite.cases[index].id,
            "semantic_plus_style",
            20260711,
        )
        for index in range(3)
    ]
    _set_production_coherence(run_dir, failures, 1)
    production = generate_report(run_dir)["stages"][1]
    assert production["semantic"]["full_passes"] == 42
    assert production["semantic"]["status"] == "passed"


def test_production_gate_rejects_question_below_two_of_three(
    complete_both_stages_dir, suite, tmp_path
):
    run_dir = _copy_run(complete_both_stages_dir, tmp_path)
    failures = [
        AnswerKey(
            "fixed-production",
            suite.cases[0].id,
            "semantic_plus_style",
            seed,
        )
        for seed in (20260711, 20260712)
    ]
    _set_production_coherence(run_dir, failures, 1)
    production = generate_report(run_dir)["stages"][1]
    assert production["semantic"]["full_passes"] == 43
    assert production["semantic"]["status"] == "failed"
    assert any(
        "questions below per-question" in reason
        for reason in production["semantic"]["reasons"]
    )


def test_fixed_production_is_blocked_offline_before_ollama_preflight(
    complete_diagnostic_dir, suite, monkeypatch, tmp_path
):
    run_dir = _copy_run(complete_diagnostic_dir, tmp_path)
    key = AnswerKey(
        "fixed-diagnostic", suite.cases[0].id, "semantic_only", 20260711
    )
    review_path(run_dir, key).unlink()
    called = False

    class ForbiddenClient:
        def __init__(self):
            nonlocal called
            called = True
            raise AssertionError("Ollama must not be contacted")

    monkeypatch.setattr(card_eval_module, "OllamaClient", ForbiddenClient)
    exit_code = card_eval_module.main(
        [
            "run",
            "--stage",
            "fixed-production",
            "--suite",
            str(SUITE_PATH),
            "--model",
            "qwen3:14b",
            "--run-dir",
            str(run_dir),
            "--resume",
        ]
    )
    assert exit_code == 2
    assert called is False


def test_report_contract_is_lexicographic_and_path_stable(
    complete_diagnostic_dir, tmp_path
):
    run_dir = _copy_run(complete_diagnostic_dir, tmp_path)
    summary = generate_report(run_dir)
    assert summary["suite"]["provenance"]["status"] == "passed"
    assert summary["model"]["digest"] == DIGEST
    assert summary["stages"][0]["retrieval"] == {
        "status": "not-applicable",
        "recall_at_3": None,
    }
    encoded = json.dumps(summary, ensure_ascii=False, sort_keys=True)
    assert "generated_at" not in encoded
    assert "weighted" not in encoded
    assert str(tmp_path) not in encoded

    for answer in summary["stages"][0]["answers"]:
        if answer["arm"] != "semantic_plus_style":
            continue
        path = run_dir / answer["review_path"]
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw["style_fidelity"] = 1
        write_atomic_json(path, raw)
    failed_style = generate_report(run_dir)
    assert failed_style["stages"][0]["style"]["status"] == "failed"
    assert "style thresholds not met" in (run_dir / "report.md").read_text(
        encoding="utf-8"
    )
