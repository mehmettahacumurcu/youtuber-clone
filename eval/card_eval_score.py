"""Machine diagnostics and non-compensating semantic gates for card evaluation."""

from __future__ import annotations

import hashlib
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from eval.card_eval_schema import EvalCase


Stage = Literal["fixed-diagnostic", "fixed-production", "live-production"]
ArmId = Literal["semantic_only", "semantic_plus_style"]
ReviewLabel = Literal["entailed", "unsupported", "contradicted", "nonclaim"]
GateStatus = Literal["blocked", "incomplete", "failed", "passed"]
StyleStatus = Literal["not-applicable", "unreviewed", "failed", "passed"]

STAGES = ("fixed-diagnostic", "fixed-production", "live-production")
ARM_IDS = ("semantic_only", "semantic_plus_style")
CLAIM_LABELS = ("entailed", "unsupported", "contradicted", "nonclaim")
CRITICAL_FAILURE_ORDER = (
    "unsupported_claim",
    "attribution_inversion",
    "polarity_negation",
    "chronology",
    "refusal_failure",
)
CRITICAL_FAILURES = frozenset(CRITICAL_FAILURE_ORDER)

_MOJIBAKE_MARKERS = ("Ã", "Â", "Å", "Ä", "â€", "ðŸ", "ï»¿")
_THINK_TAG_RE = re.compile(r"<\s*/?\s*think\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![\w])\d+(?:[.,]\d+)*(?![\w])", re.UNICODE)
_MULTIWORD_ENTITY_RE = re.compile(
    r"\b[A-ZÇĞİÖŞÜ][\w'’.-]*(?:\s+[A-ZÇĞİÖŞÜ][\w'’.-]*)+\b",
    re.UNICODE,
)
_ACRONYM_RE = re.compile(r"\b[A-ZÇĞİÖŞÜ]{2,}\b", re.UNICODE)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, order=True)
class AnswerKey:
    stage: Stage
    question_id: str
    arm: ArmId
    seed: int

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"unknown stage {self.stage!r}")
        if type(self.question_id) is not str or not self.question_id.strip():
            raise ValueError("question_id must be a non-empty string")
        if self.arm not in ARM_IDS:
            raise ValueError(f"unknown arm {self.arm!r}")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")


@dataclass(frozen=True)
class CaseContract:
    """Trusted fixture projection used to validate reviews and result bindings."""

    question_id: str
    case_sha256: str
    known_atom_ids: frozenset[str]
    required_atom_ids: frozenset[str]

    def __post_init__(self) -> None:
        if type(self.question_id) is not str or not self.question_id.strip():
            raise ValueError("contract question_id must be non-empty")
        _validate_sha256(self.case_sha256, "contract case_sha256")
        for name in ("known_atom_ids", "required_atom_ids"):
            values = getattr(self, name)
            if type(values) is not frozenset or any(
                type(value) is not str or not value for value in values
            ):
                raise ValueError(f"contract {name} must be a frozenset of strings")
        if not self.required_atom_ids <= self.known_atom_ids:
            raise ValueError("required atoms must be a subset of known atoms")

    @classmethod
    def from_case(cls, case: EvalCase) -> "CaseContract":
        if not isinstance(case, EvalCase):
            raise TypeError("case must be an EvalCase")
        return cls(
            question_id=case.id,
            case_sha256=case.sha256,
            known_atom_ids=frozenset(atom.id for atom in case.atoms),
            required_atom_ids=frozenset(
                atom.id for atom in case.atoms if atom.required
            ),
        )


@dataclass(frozen=True)
class MachineDiagnostics:
    empty_answer: bool
    corrupt_unicode: bool
    think_leak: bool
    token_cap_stop: bool
    natural_stop: bool
    unexpected_entities: tuple[str, ...] = ()
    unexpected_numbers_dates: tuple[str, ...] = ()
    missing_required_aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "empty_answer",
            "corrupt_unicode",
            "think_leak",
            "token_cap_stop",
            "natural_stop",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in (
            "unexpected_entities",
            "unexpected_numbers_dates",
            "missing_required_aliases",
        ):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                type(value) is not str or not value for value in values
            ):
                raise ValueError(f"{name} must be a tuple of non-empty strings")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must not contain duplicates")

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.empty_answer:
            reasons.append("empty_answer")
        if self.corrupt_unicode:
            reasons.append("corrupt_unicode")
        if self.think_leak:
            reasons.append("think_leak")
        if self.token_cap_stop:
            reasons.append("token_cap_stop")
        if not self.natural_stop:
            reasons.append("no_natural_stop")
        return tuple(reasons)

    @property
    def review_flags(self) -> tuple[str, ...]:
        return (
            *(f"unexpected_entity:{item}" for item in self.unexpected_entities),
            *(
                f"unexpected_number_or_date:{item}"
                for item in self.unexpected_numbers_dates
            ),
            *(f"missing_required_alias:{item}" for item in self.missing_required_aliases),
        )


@dataclass(frozen=True)
class AnswerEvaluation:
    """Current generated answer plus the cache/fixture identity it came from."""

    key: AnswerKey
    answer: str
    result_cache_key: str
    case_sha256: str
    diagnostics: MachineDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.key, AnswerKey):
            raise ValueError("evaluation key must be an AnswerKey")
        if type(self.answer) is not str:
            raise ValueError("evaluation answer must be a string")
        _validate_sha256(self.result_cache_key, "evaluation result_cache_key")
        _validate_sha256(self.case_sha256, "evaluation case_sha256")
        if not isinstance(self.diagnostics, MachineDiagnostics):
            raise ValueError("evaluation diagnostics must be MachineDiagnostics")

    @property
    def answer_sha256(self) -> str:
        return hashlib.sha256(
            self.answer.encode("utf-8", errors="surrogatepass")
        ).hexdigest()


@dataclass(frozen=True)
class ClaimLabel:
    answer_span: str
    label: ReviewLabel
    atom_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.answer_span) is not str or not self.answer_span.strip():
            raise ValueError("claim answer_span must be non-empty")
        if self.label not in CLAIM_LABELS:
            raise ValueError(f"unknown claim label {self.label!r}")
        if type(self.atom_ids) is not tuple or any(
            type(item) is not str or not item for item in self.atom_ids
        ):
            raise ValueError("claim atom_ids must be a tuple of non-empty strings")
        if len(self.atom_ids) != len(set(self.atom_ids)):
            raise ValueError("claim atom_ids must not contain duplicates")


@dataclass(frozen=True)
class HumanReview:
    key: AnswerKey
    result_cache_key: str
    answer_sha256: str
    case_sha256: str
    claim_labels: tuple[ClaimLabel, ...]
    critical_failures: frozenset[str]
    required_atom_ids_present: frozenset[str]
    required_atom_ids_expected: frozenset[str]
    supported_claim_precision: float
    coherence: int
    style_fidelity: int | None
    reviewer: str
    reviewed_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, AnswerKey):
            raise ValueError("review key must be an AnswerKey")
        _validate_sha256(self.result_cache_key, "review result_cache_key")
        _validate_sha256(self.answer_sha256, "review answer_sha256")
        _validate_sha256(self.case_sha256, "review case_sha256")
        if type(self.claim_labels) is not tuple or any(
            not isinstance(item, ClaimLabel) for item in self.claim_labels
        ):
            raise ValueError("claim_labels must be a tuple of ClaimLabel values")
        if type(self.critical_failures) is not frozenset:
            raise ValueError("critical_failures must be a frozenset")
        unknown = self.critical_failures - CRITICAL_FAILURES
        if unknown:
            raise ValueError(
                "unknown critical failure labels: " + ", ".join(sorted(unknown))
            )
        for name in ("required_atom_ids_present", "required_atom_ids_expected"):
            value = getattr(self, name)
            if type(value) is not frozenset or any(
                type(item) is not str or not item for item in value
            ):
                raise ValueError(f"{name} must be a frozenset of non-empty strings")
        extra_present = self.required_atom_ids_present - self.required_atom_ids_expected
        if extra_present:
            raise ValueError(
                "present required atom IDs are not expected: "
                + ", ".join(sorted(extra_present))
            )

        if type(self.supported_claim_precision) not in (int, float) or not math.isfinite(
            self.supported_claim_precision
        ):
            raise ValueError("supported_claim_precision must be finite")
        if not 0 <= self.supported_claim_precision <= 1:
            raise ValueError("supported_claim_precision must be in [0, 1]")
        if type(self.coherence) is not int or not 0 <= self.coherence <= 3:
            raise ValueError("coherence must be an integer in [0, 3]")
        if self.style_fidelity is not None and (
            type(self.style_fidelity) is not int
            or not 0 <= self.style_fidelity <= 3
        ):
            raise ValueError("style_fidelity must be None or an integer in [0, 3]")
        if type(self.reviewer) is not str or not self.reviewer.strip():
            raise ValueError("reviewer must be non-empty")
        if type(self.reviewed_at) is not str or not self.reviewed_at.strip():
            raise ValueError("reviewed_at must be non-empty")

        scored = tuple(item for item in self.claim_labels if item.label != "nonclaim")
        entailed_count = sum(item.label == "entailed" for item in scored)
        computed_precision = entailed_count / len(scored) if scored else 0.0
        if not math.isclose(
            float(self.supported_claim_precision),
            computed_precision,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "supported claim precision is inconsistent with claim labels"
            )
        entailed_atoms = {
            atom_id
            for label in self.claim_labels
            if label.label == "entailed"
            for atom_id in label.atom_ids
        }
        missing_support = self.required_atom_ids_present - entailed_atoms
        if missing_support:
            raise ValueError(
                "required atoms marked present lack an entailed claim label: "
                + ", ".join(sorted(missing_support))
            )

    @property
    def all_required_atoms_present(self) -> bool:
        return self.required_atom_ids_expected <= self.required_atom_ids_present


@dataclass(frozen=True)
class GatePolicy:
    gate_id: str
    expected_keys: tuple[AnswerKey, ...]
    case_contracts: tuple[CaseContract, ...]
    minimum_full_passes: int
    minimum_passes_per_question: int
    require_zero_critical_failures: bool = True
    require_styled_no_regression: bool = False
    style_min_question_median: float | None = None
    style_min_suite_mean: float | None = None
    prerequisite_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.gate_id) is not str or not self.gate_id.strip():
            raise ValueError("gate_id must be non-empty")
        if type(self.expected_keys) is not tuple or not self.expected_keys or any(
            not isinstance(key, AnswerKey) for key in self.expected_keys
        ):
            raise ValueError("expected_keys must be a non-empty tuple of AnswerKey values")
        if len(self.expected_keys) != len(set(self.expected_keys)):
            raise ValueError("expected_keys must not contain duplicates")
        if type(self.case_contracts) is not tuple or not self.case_contracts or any(
            not isinstance(contract, CaseContract) for contract in self.case_contracts
        ):
            raise ValueError(
                "case_contracts must be a non-empty tuple of CaseContract values"
            )
        contract_ids = tuple(contract.question_id for contract in self.case_contracts)
        if len(contract_ids) != len(set(contract_ids)):
            raise ValueError("case_contracts must not contain duplicate question IDs")
        expected_question_ids = {key.question_id for key in self.expected_keys}
        if set(contract_ids) != expected_question_ids:
            raise ValueError(
                "case_contracts must exactly match expected-key question IDs"
            )
        if type(self.minimum_full_passes) is not int or not (
            0 <= self.minimum_full_passes <= len(self.expected_keys)
        ):
            raise ValueError("minimum_full_passes is outside the expected-key count")
        per_question = Counter(key.question_id for key in self.expected_keys)
        if type(self.minimum_passes_per_question) is not int or not (
            0
            <= self.minimum_passes_per_question
            <= min(per_question.values())
        ):
            raise ValueError("minimum_passes_per_question is invalid")
        if type(self.require_zero_critical_failures) is not bool:
            raise ValueError("require_zero_critical_failures must be boolean")
        if type(self.require_styled_no_regression) is not bool:
            raise ValueError("require_styled_no_regression must be boolean")
        if (self.style_min_question_median is None) != (
            self.style_min_suite_mean is None
        ):
            raise ValueError("both style thresholds must be set together")
        for value in (self.style_min_question_median, self.style_min_suite_mean):
            if value is not None and (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 3
            ):
                raise ValueError("style thresholds must be finite and in [0, 3]")
        if type(self.prerequisite_errors) is not tuple or any(
            type(item) is not str or not item for item in self.prerequisite_errors
        ):
            raise ValueError("prerequisite_errors must be a tuple of strings")


@dataclass(frozen=True)
class QuestionGateResult:
    question_id: str
    expected_count: int
    reviewed_count: int
    full_pass_count: int
    critical_failures: tuple[str, ...]
    style_scores: tuple[int, ...]
    style_median: float | None


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    status: GateStatus
    expected_count: int
    result_count: int
    reviewed_count: int
    full_pass_count: int
    missing_result_keys: tuple[AnswerKey, ...]
    missing_review_keys: tuple[AnswerKey, ...]
    failed_answer_keys: tuple[AnswerKey, ...]
    critical_failure_counts: tuple[tuple[str, int], ...]
    styled_regressions: tuple[str, ...]
    per_question: tuple[QuestionGateResult, ...]
    semantic_passed: bool
    style_status: StyleStatus
    style_mean: float | None
    reasons: tuple[str, ...]
    style_reasons: tuple[str, ...]


def diagnose_answer(
    case: EvalCase,
    answer: str,
    *,
    thinking: str | None = None,
    done_reason: str | None = None,
    eval_count: int | None = None,
    num_predict: int | None = None,
) -> MachineDiagnostics:
    """Return fail-only diagnostics; semantic correctness remains human-reviewed."""

    if not isinstance(case, EvalCase):
        raise TypeError("case must be an EvalCase")
    if type(answer) is not str:
        raise TypeError("answer must be a string")
    if thinking is not None and type(thinking) is not str:
        raise TypeError("thinking must be a string or None")
    for name, value in (("eval_count", eval_count), ("num_predict", num_predict)):
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"{name} must be a non-negative integer or None")
    if done_reason is not None and type(done_reason) is not str:
        raise TypeError("done_reason must be a string or None")

    empty = not answer.strip()
    corrupt = _has_corrupt_unicode(answer)
    think_leak = bool((thinking or "").strip()) or bool(_THINK_TAG_RE.search(answer))
    token_cap = done_reason == "length" or (
        done_reason != "stop"
        and eval_count is not None
        and num_predict is not None
        and eval_count >= num_predict
    )
    natural_stop = not empty and done_reason == "stop" and not token_cap

    corpus = _semantic_corpus(case)
    corpus_folded = _comparison_fold(corpus)
    corpus_numbers = {
        _comparison_fold(token) for token in _NUMBER_RE.findall(corpus)
    }
    unexpected_numbers = tuple(
        _unique_in_order(
            token
            for token in _NUMBER_RE.findall(answer)
            if _comparison_fold(token) not in corpus_numbers
        )
    )
    candidates = [
        *_MULTIWORD_ENTITY_RE.findall(answer),
        *_ACRONYM_RE.findall(answer),
    ]
    unexpected_entities = tuple(
        _unique_in_order(
            candidate
            for candidate in candidates
            if _comparison_fold(candidate) not in corpus_folded
        )
    )

    return MachineDiagnostics(
        empty_answer=empty,
        corrupt_unicode=corrupt,
        think_leak=think_leak,
        token_cap_stop=token_cap,
        natural_stop=natural_stop,
        unexpected_entities=unexpected_entities,
        unexpected_numbers_dates=unexpected_numbers,
        missing_required_aliases=(),
    )


def answer_passes(
    review: HumanReview,
    evaluation: AnswerEvaluation,
    contract: CaseContract,
) -> bool:
    if not isinstance(review, HumanReview):
        raise TypeError("review must be a HumanReview")
    if not isinstance(evaluation, AnswerEvaluation):
        raise TypeError("evaluation must be an AnswerEvaluation")
    if not isinstance(contract, CaseContract):
        raise TypeError("contract must be a CaseContract")
    return (
        not _review_binding_errors(review, evaluation, contract)
        and not review.critical_failures
        and review.supported_claim_precision == 1.0
        and contract.required_atom_ids <= review.required_atom_ids_present
        and review.coherence >= 2
        and not evaluation.diagnostics.blocking_reasons
    )


def review_binding_errors(
    review: HumanReview,
    evaluation: AnswerEvaluation,
    contract: CaseContract,
) -> tuple[str, ...]:
    """Public strict binding check used by the review-file loader."""

    return _review_binding_errors(review, evaluation, contract)


def answer_failure_reasons(
    review: HumanReview,
    evaluation: AnswerEvaluation,
    contract: CaseContract,
) -> tuple[str, ...]:
    """Return deterministic semantic/health reasons; style is intentionally absent."""

    reasons = [
        *(f"binding:{item}" for item in _review_binding_errors(review, evaluation, contract)),
        *(f"machine:{item}" for item in evaluation.diagnostics.blocking_reasons),
        *(
            f"critical:{failure}"
            for failure in CRITICAL_FAILURE_ORDER
            if failure in review.critical_failures
        ),
        *(
            f"claim:{label.label}:{label.answer_span}"
            for label in review.claim_labels
            if label.label in ("unsupported", "contradicted")
        ),
        *(
            f"missing_required_atom:{atom_id}"
            for atom_id in sorted(
                contract.required_atom_ids - review.required_atom_ids_present
            )
        ),
    ]
    if review.supported_claim_precision != 1.0:
        reasons.append(
            f"supported_claim_precision:{review.supported_claim_precision:g}"
        )
    if review.coherence < 2:
        reasons.append(f"coherence_below_2:{review.coherence}")
    return tuple(reasons)


def evaluate_gate(
    results: Mapping[AnswerKey, AnswerEvaluation],
    reviews: Mapping[AnswerKey, HumanReview],
    policy: GatePolicy,
) -> GateResult:
    """Evaluate an explicit answer matrix without letting style offset semantics."""

    if not isinstance(results, Mapping) or not isinstance(reviews, Mapping):
        raise TypeError("results and reviews must be mappings keyed by AnswerKey")
    if not isinstance(policy, GatePolicy):
        raise TypeError("policy must be a GatePolicy")
    result_key_mismatches: list[str] = []
    for key, evaluation in results.items():
        if not isinstance(key, AnswerKey) or not isinstance(
            evaluation, AnswerEvaluation
        ):
            raise TypeError("results must map AnswerKey to AnswerEvaluation")
        if evaluation.key != key:
            result_key_mismatches.append(
                f"result mapping key {key!r} differs from embedded key "
                f"{evaluation.key!r}"
            )
    review_key_mismatches: list[str] = []
    for key, review in reviews.items():
        if not isinstance(key, AnswerKey) or not isinstance(review, HumanReview):
            raise TypeError("reviews must map AnswerKey to HumanReview")
        if review.key != key:
            review_key_mismatches.append(
                f"review mapping key {key!r} differs from embedded key {review.key!r}"
            )

    expected_order = policy.expected_keys
    contracts = {
        contract.question_id: contract for contract in policy.case_contracts
    }
    expected = set(expected_order)
    result_keys = set(results)
    review_keys = set(reviews)
    missing_results = tuple(key for key in expected_order if key not in result_keys)
    missing_reviews = tuple(key for key in expected_order if key not in review_keys)
    extra_results = tuple(sorted(result_keys - expected))
    extra_reviews = tuple(sorted(review_keys - expected))

    binding_errors: list[str] = []
    pass_by_key: dict[AnswerKey, bool] = {}
    for key in expected_order:
        if (
            key in results
            and key in reviews
            and reviews[key].key == key
            and results[key].key == key
        ):
            errors = _review_binding_errors(
                reviews[key], results[key], contracts[key.question_id]
            )
            binding_errors.extend(f"{key}: {error}" for error in errors)
            pass_by_key[key] = answer_passes(
                reviews[key], results[key], contracts[key.question_id]
            )

    failed_keys = tuple(
        key for key in expected_order if key in pass_by_key and not pass_by_key[key]
    )
    critical_counts = Counter(
        failure
        for key in expected_order
        if key in reviews
        for failure in reviews[key].critical_failures
    )

    bound_styled_keys = {
        key
        for key in expected_order
        if key.arm == "semantic_plus_style"
        and key in results
        and key in reviews
        and results[key].key == key
        and reviews[key].key == key
        and not _review_binding_errors(
            reviews[key], results[key], contracts[key.question_id]
        )
    }

    question_keys: dict[str, list[AnswerKey]] = defaultdict(list)
    for key in expected_order:
        question_keys[key.question_id].append(key)

    question_results: list[QuestionGateResult] = []
    for question_id, keys in question_keys.items():
        question_reviews = [reviews[key] for key in keys if key in reviews]
        style_scores = tuple(
            reviews[key].style_fidelity
            for key in keys
            if key.arm == "semantic_plus_style"
            and key in bound_styled_keys
            and key in reviews
            and reviews[key].style_fidelity is not None
        )
        style_median = (
            float(statistics.median(style_scores)) if style_scores else None
        )
        question_results.append(
            QuestionGateResult(
                question_id=question_id,
                expected_count=len(keys),
                reviewed_count=len(question_reviews),
                full_pass_count=sum(pass_by_key.get(key, False) for key in keys),
                critical_failures=tuple(
                    sorted(
                        {
                            failure
                            for review in question_reviews
                            for failure in review.critical_failures
                        }
                    )
                ),
                style_scores=style_scores,
                style_median=style_median,
            )
        )

    styled_regressions: list[str] = []
    if policy.require_styled_no_regression:
        grouped: dict[tuple[str, int], dict[str, AnswerKey]] = defaultdict(dict)
        for key in expected_order:
            grouped[(key.question_id, key.seed)][key.arm] = key
        for (question_id, _seed), arms in grouped.items():
            base = arms.get("semantic_only")
            styled = arms.get("semantic_plus_style")
            if (
                base is not None
                and styled is not None
                and base in reviews
                and styled in reviews
                and (
                    _human_semantic_failures(
                        reviews[styled], contracts[question_id]
                    )
                    - _human_semantic_failures(
                        reviews[base], contracts[question_id]
                    )
                )
                and question_id not in styled_regressions
            ):
                styled_regressions.append(question_id)

    reasons: list[str] = list(policy.prerequisite_errors)
    if extra_results:
        reasons.append(f"unexpected result keys: {len(extra_results)}")
    if extra_reviews:
        reasons.append(f"unexpected review keys: {len(extra_reviews)}")
    reasons.extend(result_key_mismatches)
    reasons.extend(review_key_mismatches)
    reasons.extend(binding_errors)
    blocked = bool(reasons)

    incomplete = bool(missing_results or missing_reviews)
    if missing_results:
        reasons.append(f"missing results: {len(missing_results)}")
    if missing_reviews:
        reasons.append(f"missing reviews: {len(missing_reviews)}")

    full_pass_count = sum(pass_by_key.values())
    semantic_thresholds_pass = True
    if full_pass_count < policy.minimum_full_passes:
        semantic_thresholds_pass = False
        reasons.append(
            f"full passes {full_pass_count}/{len(expected_order)} below "
            f"minimum {policy.minimum_full_passes}"
        )
    below_question_threshold = tuple(
        item.question_id
        for item in question_results
        if item.full_pass_count < policy.minimum_passes_per_question
    )
    if below_question_threshold:
        semantic_thresholds_pass = False
        reasons.append(
            "questions below per-question pass minimum: "
            + ", ".join(below_question_threshold)
        )
    if policy.require_zero_critical_failures and critical_counts:
        semantic_thresholds_pass = False
        reasons.append("one or more answers has a critical semantic failure")
    if policy.require_styled_no_regression and styled_regressions:
        semantic_thresholds_pass = False
        reasons.append(
            "styled semantic regressions: " + ", ".join(styled_regressions)
        )

    if blocked:
        status: GateStatus = "blocked"
    elif incomplete:
        status = "incomplete"
    elif semantic_thresholds_pass:
        status = "passed"
    else:
        status = "failed"
    semantic_passed = status == "passed"

    style_status: StyleStatus
    style_mean: float | None
    style_reasons: list[str] = []
    if policy.style_min_question_median is None:
        style_status = "not-applicable"
        style_mean = None
    else:
        styled_expected = tuple(
            key for key in expected_order if key.arm == "semantic_plus_style"
        )
        semantic_only_scored = tuple(
            key
            for key in expected_order
            if key.arm == "semantic_only"
            and key in reviews
            and reviews[key].style_fidelity is not None
        )
        styled_reviews = [reviews[key] for key in styled_expected if key in reviews]
        stale_styled = any(key not in bound_styled_keys for key in styled_expected)
        if stale_styled:
            style_status = "unreviewed"
            style_mean = None
            style_reasons.append("one or more styled reviews is stale")
        elif semantic_only_scored:
            style_status = "failed"
            style_mean = None
            style_reasons.append("semantic-only reviews must not carry style scores")
        elif not styled_expected:
            style_status = "failed"
            style_mean = None
            style_reasons.append("style policy has no styled answer keys")
        elif len(styled_reviews) != len(styled_expected) or any(
            review.style_fidelity is None for review in styled_reviews
        ):
            style_status = "unreviewed"
            style_mean = None
            style_reasons.append("one or more styled answers lacks a style score")
        else:
            scores = [
                review.style_fidelity
                for review in styled_reviews
                if review.style_fidelity is not None
            ]
            style_mean = float(statistics.fmean(scores))
            medians_pass = all(
                item.style_median is not None
                and item.style_median >= policy.style_min_question_median
                for item in question_results
            )
            mean_pass = style_mean >= policy.style_min_suite_mean
            style_status = "passed" if medians_pass and mean_pass else "failed"
            if style_status == "failed":
                style_reasons.append("style thresholds not met")

    return GateResult(
        gate_id=policy.gate_id,
        status=status,
        expected_count=len(expected_order),
        result_count=len(results),
        reviewed_count=len(expected & review_keys),
        full_pass_count=full_pass_count,
        missing_result_keys=missing_results,
        missing_review_keys=missing_reviews,
        failed_answer_keys=failed_keys,
        critical_failure_counts=tuple(sorted(critical_counts.items())),
        styled_regressions=tuple(styled_regressions),
        per_question=tuple(question_results),
        semantic_passed=semantic_passed,
        style_status=style_status,
        style_mean=style_mean,
        reasons=tuple(reasons),
        style_reasons=tuple(style_reasons),
    )


def _review_binding_errors(
    review: HumanReview,
    evaluation: AnswerEvaluation,
    contract: CaseContract,
) -> tuple[str, ...]:
    errors: list[str] = []
    if review.key != evaluation.key:
        errors.append("review and result answer keys differ")
    if contract.question_id != evaluation.key.question_id:
        errors.append("fixture contract question does not match the result key")
    if review.result_cache_key != evaluation.result_cache_key:
        errors.append("review result_cache_key is stale")
    if review.answer_sha256 != evaluation.answer_sha256:
        errors.append("review answer_sha256 is stale")
    if evaluation.case_sha256 != contract.case_sha256:
        errors.append("result case_sha256 differs from the fixture contract")
    if review.case_sha256 != contract.case_sha256:
        errors.append("review case_sha256 differs from the fixture contract")
    if review.required_atom_ids_expected != contract.required_atom_ids:
        errors.append("review required-atom universe differs from the fixture")
    if not review.required_atom_ids_present <= contract.required_atom_ids:
        errors.append("review marks unknown/non-required atoms as required-present")

    labelled_atoms = {
        atom_id for label in review.claim_labels for atom_id in label.atom_ids
    }
    unknown_atoms = labelled_atoms - contract.known_atom_ids
    if unknown_atoms:
        errors.append(
            "claim labels reference unknown fixture atoms: "
            + ", ".join(sorted(unknown_atoms))
        )
    entailed_atoms = {
        atom_id
        for label in review.claim_labels
        if label.label == "entailed"
        for atom_id in label.atom_ids
    }
    expected_present = contract.required_atom_ids & entailed_atoms
    if review.required_atom_ids_present != expected_present:
        errors.append(
            "required-present atoms do not equal entailed fixture-required atoms"
        )
    if any(label.label == "nonclaim" and label.atom_ids for label in review.claim_labels):
        errors.append("nonclaim labels must not reference atoms")
    if any(label.label == "unsupported" for label in review.claim_labels) and (
        "unsupported_claim" not in review.critical_failures
    ):
        errors.append("unsupported claim labels require unsupported_claim")
    if any(label.label == "contradicted" for label in review.claim_labels) and not (
        review.critical_failures
    ):
        errors.append("contradicted claim labels require a critical failure")

    cursor = 0
    seen_spans: set[str] = set()
    for label in review.claim_labels:
        if label.answer_span in seen_spans:
            errors.append("claim answer spans must be unique")
            continue
        seen_spans.add(label.answer_span)
        position = evaluation.answer.find(label.answer_span, cursor)
        if position < 0:
            errors.append(
                f"claim span is absent or out of order: {label.answer_span!r}"
            )
            continue
        cursor = position + len(label.answer_span)
    return tuple(errors)


def _human_semantic_failures(
    review: HumanReview, contract: CaseContract
) -> frozenset[str]:
    failures = {f"critical:{item}" for item in review.critical_failures}
    if review.supported_claim_precision != 1.0:
        failures.add("supported_claim_precision")
    failures.update(
        f"missing_required_atom:{item}"
        for item in contract.required_atom_ids - review.required_atom_ids_present
    )
    if review.coherence < 2:
        failures.add("coherence_below_2")
    return frozenset(failures)


def _has_corrupt_unicode(value: str) -> bool:
    return (
        not unicodedata.is_normalized("NFC", value)
        or "\ufffd" in value
        or any(marker in value for marker in _MOJIBAKE_MARKERS)
        or any(0xD800 <= ord(character) <= 0xDFFF for character in value)
    )


def _semantic_corpus(case: EvalCase) -> str:
    values = [
        case.question,
        *case.allowed_entities,
        *case.allowed_numbers_dates,
        *(card.q for card in case.cards),
        *(card.take for card in case.cards),
        *(evidence.excerpt for evidence in case.evidence),
    ]
    return "\n".join(values)


def _comparison_fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def _unique_in_order(values):
    seen: set[str] = set()
    for value in values:
        folded = _comparison_fold(value)
        if folded not in seen:
            seen.add(folded)
            yield value


def _validate_sha256(value: object, label: str) -> None:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
