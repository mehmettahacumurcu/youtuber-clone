from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from eval.card_eval_schema import load_suite
from eval.card_eval_score import (
    AnswerEvaluation,
    AnswerKey,
    CaseContract,
    ClaimLabel,
    GatePolicy,
    HumanReview,
    MachineDiagnostics,
    answer_passes,
    diagnose_answer,
    evaluate_gate,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_PATH = REPO_ROOT / "eval" / "fixtures" / "verified_card_eval_v1.json"
CRITICAL_FAILURES = {
    "unsupported_claim",
    "attribution_inversion",
    "polarity_negation",
    "chronology",
    "refusal_failure",
}


@pytest.fixture(scope="module")
def suite():
    return load_suite(SUITE_PATH)


@pytest.fixture(scope="module")
def g01(suite):
    return next(case for case in suite.cases if case.id == "G01")


def test_diagnose_answer_accepts_only_authoritative_natural_stop(g01):
    good = diagnose_answer(
        g01,
        "Mutlak butlan anlatısı kabul ettirilemez; suç örgütü kısmı bir hikâye.",
        done_reason="stop",
        eval_count=20,
        num_predict=384,
    )
    assert good.natural_stop
    assert not good.blocking_reasons
    assert not good.corrupt_unicode

    unknown = diagnose_answer(g01, "Tamamdır.")
    assert not unknown.natural_stop
    assert "no_natural_stop" in unknown.blocking_reasons


@pytest.mark.parametrize(
    ("answer", "kwargs", "reason"),
    [
        ("   ", {"done_reason": "stop"}, "empty_answer"),
        ("bozuk \ufffd", {"done_reason": "stop"}, "corrupt_unicode"),
        ("gu\u0308ya", {"done_reason": "stop"}, "corrupt_unicode"),
        ("<think>gizli akıl yürütme</think> Yanıt", {"done_reason": "stop"}, "think_leak"),
        ("Yanıt", {"thinking": "gizli", "done_reason": "stop"}, "think_leak"),
        ("Yarım yanıt", {"done_reason": "length"}, "token_cap_stop"),
        (
            "Yarım yanıt",
            {"done_reason": "other", "eval_count": 384, "num_predict": 384},
            "token_cap_stop",
        ),
    ],
)
def test_diagnose_answer_objective_blockers(g01, answer, kwargs, reason):
    diagnostics = diagnose_answer(g01, answer, **kwargs)
    assert reason in diagnostics.blocking_reasons


def test_unexpected_names_and_numbers_are_review_flags_not_pass_evidence(g01):
    diagnostics = diagnose_answer(
        g01,
        "Hakan Fidan bu işi 777 günde çözdü.",
        done_reason="stop",
    )
    assert "Hakan Fidan" in diagnostics.unexpected_entities
    assert "777" in diagnostics.unexpected_numbers_dates
    assert diagnostics.review_flags
    assert not diagnostics.blocking_reasons


def test_unexpected_number_matching_is_token_exact_not_substring(g01):
    tiny_case = replace(
        g01,
        question="Süre 35 gündü.",
        cards=(),
        evidence=(),
        allowed_entities=(),
        allowed_numbers_dates=("35 gün",),
    )
    diagnostics = diagnose_answer(tiny_case, "Süre 3 gündü.", done_reason="stop")
    assert diagnostics.unexpected_numbers_dates == ("3",)


def _key(question: str = "G01", arm: str = "semantic_only", seed: int = 1):
    return AnswerKey(
        stage="fixed-diagnostic", question_id=question, arm=arm, seed=seed
    )


def _diagnostics(*, natural_stop: bool = True) -> MachineDiagnostics:
    return MachineDiagnostics(
        empty_answer=False,
        corrupt_unicode=False,
        think_leak=False,
        token_cap_stop=False,
        natural_stop=natural_stop,
        unexpected_entities=(),
        unexpected_numbers_dates=(),
        missing_required_aliases=(),
    )


CASE_SHA = "a" * 64
EXPECTED_ATOMS = frozenset({"stance", "attribution"})


def _cache_key(key: AnswerKey) -> str:
    return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()


def _evaluation(
    key: AnswerKey,
    *,
    diagnostics: MachineDiagnostics | None = None,
    answer: str = "doğru önerme; desteksiz önerme.",
) -> AnswerEvaluation:
    return AnswerEvaluation(
        key=key,
        answer=answer,
        result_cache_key=_cache_key(key),
        case_sha256=CASE_SHA,
        diagnostics=diagnostics or _diagnostics(),
    )


def _contract(question_id: str) -> CaseContract:
    return CaseContract(
        question_id=question_id,
        case_sha256=CASE_SHA,
        known_atom_ids=EXPECTED_ATOMS | frozenset({"optional"}),
        required_atom_ids=EXPECTED_ATOMS,
    )


def _review(
    key: AnswerKey,
    *,
    evaluation: AnswerEvaluation | None = None,
    critical: frozenset[str] = frozenset(),
    precision: float = 1.0,
    all_atoms: bool = True,
    coherence: int = 3,
    style: int | None = None,
) -> HumanReview:
    evaluation = evaluation or _evaluation(key)
    expected = EXPECTED_ATOMS
    present = expected if all_atoms else frozenset({"stance"})
    labels = (
        ClaimLabel(
            answer_span="doğru önerme",
            label="entailed",
            atom_ids=tuple(sorted(present)),
        ),
    )
    if precision < 1.0:
        labels += (
            ClaimLabel(
                answer_span="desteksiz önerme",
                label="unsupported",
                atom_ids=(),
            ),
        )
        precision = 0.5
        if not critical:
            critical = frozenset({"unsupported_claim"})
    return HumanReview(
        key=key,
        result_cache_key=evaluation.result_cache_key,
        answer_sha256=evaluation.answer_sha256,
        case_sha256=evaluation.case_sha256,
        claim_labels=labels,
        critical_failures=critical,
        required_atom_ids_present=present,
        required_atom_ids_expected=expected,
        supported_claim_precision=precision,
        coherence=coherence,
        style_fidelity=style,
        reviewer="codex-semantic-review",
        reviewed_at="2026-07-11T00:00:00Z",
    )


def test_answer_passes_is_semantic_and_style_cannot_compensate():
    key = _key()
    evaluation = _evaluation(key)
    contract = _contract(key.question_id)
    good_review = _review(key, evaluation=evaluation, style=0)
    assert answer_passes(good_review, evaluation, contract)

    assert not answer_passes(
        _review(
            key,
            evaluation=evaluation,
            critical=frozenset({"unsupported_claim"}),
            style=3,
        ),
        evaluation,
        contract,
    )
    assert not answer_passes(
        _review(key, evaluation=evaluation, precision=0.5, style=3),
        evaluation,
        contract,
    )
    assert not answer_passes(
        _review(key, evaluation=evaluation, all_atoms=False, style=3),
        evaluation,
        contract,
    )
    assert not answer_passes(
        _review(key, evaluation=evaluation, coherence=1, style=3),
        evaluation,
        contract,
    )
    stopped = replace(evaluation, diagnostics=replace(_diagnostics(), natural_stop=False))
    assert not answer_passes(
        replace(
            good_review,
            result_cache_key=stopped.result_cache_key,
            answer_sha256=stopped.answer_sha256,
        ),
        stopped,
        contract,
    )


def test_review_validation_rejects_unknown_failures_and_inconsistent_precision():
    with pytest.raises(ValueError, match="critical"):
        _review(_key(), critical=frozenset({"made_up_label"}))

    with pytest.raises(ValueError, match="precision"):
        HumanReview(
            key=_key(),
            result_cache_key="b" * 64,
            answer_sha256="c" * 64,
            case_sha256=CASE_SHA,
            claim_labels=(
                ClaimLabel("unsupported", "unsupported", ()),
            ),
            critical_failures=frozenset(),
            required_atom_ids_present=frozenset(),
            required_atom_ids_expected=frozenset(),
            supported_claim_precision=1.0,
            coherence=3,
            style_fidelity=None,
            reviewer="reviewer",
            reviewed_at="2026-07-11T00:00:00Z",
        )

    assert CRITICAL_FAILURES


def _diagnostic_policy() -> GatePolicy:
    keys = tuple(
        _key(question=f"Q{question:02d}", arm=arm, seed=20260711)
        for question in range(15)
        for arm in ("semantic_only", "semantic_plus_style")
    )
    return GatePolicy(
        gate_id="fixed-diagnostic",
        expected_keys=keys,
        case_contracts=tuple(_contract(f"Q{question:02d}") for question in range(15)),
        minimum_full_passes=30,
        minimum_passes_per_question=2,
        require_zero_critical_failures=True,
        require_styled_no_regression=True,
    )


def _production_policy() -> GatePolicy:
    keys = tuple(
        AnswerKey(
            stage="fixed-production",
            question_id=f"Q{question:02d}",
            arm="semantic_plus_style",
            seed=seed,
        )
        for question in range(15)
        for seed in (20260711, 20260712, 20260713)
    )
    return GatePolicy(
        gate_id="fixed-production",
        expected_keys=keys,
        case_contracts=tuple(_contract(f"Q{question:02d}") for question in range(15)),
        minimum_full_passes=42,
        minimum_passes_per_question=2,
        require_zero_critical_failures=True,
        style_min_question_median=2.0,
        style_min_suite_mean=2.0,
    )


def _passing_inputs(policy: GatePolicy, *, style: int | None = None):
    results = {key: _evaluation(key) for key in policy.expected_keys}
    reviews = {
        key: _review(key, evaluation=results[key], style=style)
        for key in policy.expected_keys
    }
    return results, reviews


def test_fixed_diagnostic_requires_all_30_and_surfaces_styled_regression():
    policy = _diagnostic_policy()
    results, reviews = _passing_inputs(policy)
    passed = evaluate_gate(results, reviews, policy)
    assert passed.status == "passed"
    assert passed.full_pass_count == 30
    assert not passed.styled_regressions

    styled = next(key for key in policy.expected_keys if key.arm == "semantic_plus_style")
    reviews[styled] = _review(styled, all_atoms=False)
    failed = evaluate_gate(results, reviews, policy)
    assert failed.status == "failed"
    assert styled.question_id in failed.styled_regressions

    reviews.pop(styled)
    incomplete = evaluate_gate(results, reviews, policy)
    assert incomplete.status == "incomplete"
    assert styled in incomplete.missing_review_keys


def test_production_boundary_is_42_of_45_and_two_of_three_per_question():
    policy = _production_policy()
    results, reviews = _passing_inputs(policy, style=2)

    three_questions = [f"Q{i:02d}" for i in range(3)]
    for question in three_questions:
        key = next(
            item
            for item in policy.expected_keys
            if item.question_id == question and item.seed == 20260711
        )
        reviews[key] = _review(key, coherence=1, style=2)

    exact = evaluate_gate(results, reviews, policy)
    assert exact.status == "passed"
    assert exact.full_pass_count == 42
    assert exact.style_status == "passed"

    fourth = next(
        key
        for key in policy.expected_keys
        if key.question_id == "Q03" and key.seed == 20260711
    )
    reviews[fourth] = _review(fourth, coherence=1, style=2)
    assert evaluate_gate(results, reviews, policy).status == "failed"

    results, reviews = _passing_inputs(policy, style=3)
    same_question = [
        key for key in policy.expected_keys if key.question_id == "Q00"
    ][:2]
    for key in same_question:
        reviews[key] = _review(key, coherence=1, style=3)
    per_question = evaluate_gate(results, reviews, policy)
    assert per_question.full_pass_count == 43
    assert per_question.status == "failed"


def test_any_critical_failure_fails_production_and_style_never_compensates():
    policy = _production_policy()
    results, reviews = _passing_inputs(policy, style=3)
    key = policy.expected_keys[0]
    reviews[key] = _review(
        key,
        critical=frozenset({"unsupported_claim"}),
        style=3,
    )

    gate = evaluate_gate(results, reviews, policy)
    assert gate.status == "failed"
    assert gate.style_status == "passed"
    assert gate.critical_failure_counts == (("unsupported_claim", 1),)


def test_gate_fails_closed_on_extra_keys_and_missing_style_scores():
    policy = _production_policy()
    results, reviews = _passing_inputs(policy, style=2)
    extra = AnswerKey(
        stage="live-production",
        question_id="Q00",
        arm="semantic_plus_style",
        seed=20260711,
    )
    results[extra] = _evaluation(extra)
    assert evaluate_gate(results, reviews, policy).status == "blocked"

    results.pop(extra)
    reviews[policy.expected_keys[0]] = _review(policy.expected_keys[0], style=None)
    gate = evaluate_gate(results, reviews, policy)
    assert gate.status == "passed"
    assert gate.style_status == "unreviewed"


def test_gate_blocks_stale_answer_binding_and_fake_required_atom_universe():
    policy = _diagnostic_policy()
    results, reviews = _passing_inputs(policy)
    key = policy.expected_keys[0]

    reviews[key] = replace(reviews[key], answer_sha256="b" * 64)
    stale = evaluate_gate(results, reviews, policy)
    assert stale.status == "blocked"
    assert any("answer_sha256" in reason for reason in stale.reasons)

    reviews[key] = HumanReview(
        key=key,
        result_cache_key=results[key].result_cache_key,
        answer_sha256=results[key].answer_sha256,
        case_sha256=CASE_SHA,
        claim_labels=(ClaimLabel("doğru önerme", "entailed", ("fake",)),),
        critical_failures=frozenset(),
        required_atom_ids_present=frozenset({"fake"}),
        required_atom_ids_expected=frozenset({"fake"}),
        supported_claim_precision=1.0,
        coherence=3,
        style_fidelity=None,
        reviewer="reviewer",
        reviewed_at="2026-07-11T00:00:00Z",
    )
    fake = evaluate_gate(results, reviews, policy)
    assert fake.status == "blocked"
    assert any("required-atom universe" in reason for reason in fake.reasons)


def test_styled_regression_detects_new_failure_even_when_base_already_fails():
    policy = _diagnostic_policy()
    results, reviews = _passing_inputs(policy)
    base = next(key for key in policy.expected_keys if key.arm == "semantic_only")
    styled = next(
        key
        for key in policy.expected_keys
        if key.question_id == base.question_id and key.arm == "semantic_plus_style"
    )
    reviews[base] = _review(base, evaluation=results[base], all_atoms=False)
    reviews[styled] = _review(
        styled,
        evaluation=results[styled],
        all_atoms=False,
        critical=frozenset({"chronology"}),
    )

    gate = evaluate_gate(results, reviews, policy)
    assert base.question_id in gate.styled_regressions


def test_style_failure_is_separate_from_a_passed_semantic_gate():
    policy = _production_policy()
    results, reviews = _passing_inputs(policy, style=3)
    for key in policy.expected_keys:
        if key.question_id == "Q00":
            reviews[key] = _review(key, evaluation=results[key], style=1)

    gate = evaluate_gate(results, reviews, policy)
    assert gate.status == "passed"
    assert gate.style_status == "failed"
    assert not any("style" in reason for reason in gate.reasons)
    assert gate.style_reasons == ("style thresholds not met",)


def test_stale_styled_review_cannot_produce_a_passing_style_subgate():
    policy = _production_policy()
    results, reviews = _passing_inputs(policy, style=3)
    key = policy.expected_keys[0]
    reviews[key] = replace(reviews[key], answer_sha256="b" * 64)

    gate = evaluate_gate(results, reviews, policy)
    assert gate.status == "blocked"
    assert gate.style_status == "unreviewed"
    assert gate.style_mean is None
    assert any("stale" in reason for reason in gate.style_reasons)


def test_orphan_styled_review_without_current_result_is_unreviewed_for_style():
    policy = _production_policy()
    results, reviews = _passing_inputs(policy, style=3)
    key = policy.expected_keys[0]
    results.pop(key)

    gate = evaluate_gate(results, reviews, policy)
    assert gate.status == "incomplete"
    assert gate.style_status == "unreviewed"
    assert gate.style_mean is None
    q00 = next(item for item in gate.per_question if item.question_id == "Q00")
    assert len(q00.style_scores) == 2


def test_lone_surrogate_answer_hashes_safely_and_remains_a_unicode_blocker(g01):
    answer = chr(0xD800)
    diagnostics = diagnose_answer(g01, answer, done_reason="stop")
    evaluation = _evaluation(_key(), diagnostics=diagnostics, answer=answer)

    assert evaluation.answer_sha256 == hashlib.sha256(
        answer.encode("utf-8", errors="surrogatepass")
    ).hexdigest()
    assert "corrupt_unicode" in diagnostics.blocking_reasons
