"""Resumable, one-result-per-unit runner for the verified-card evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from eval.card_eval_ollama import (
    EmptyContentError,
    GenerationResult,
    GenerationSettings,
    ModelIdentity,
    OFFICIAL_MODEL_DIGEST,
    OFFICIAL_MODEL_METADATA,
    OllamaClient,
    OllamaError,
    ThinkingLeakError,
)
from eval.card_eval_prompt import ARM_IDS, build_messages
from eval.card_eval_schema import (
    EvalCase,
    EvalSuite,
    canonical_sha256,
    load_suite,
    validate_suite,
)
from eval.card_eval_score import (
    CRITICAL_FAILURE_ORDER,
    AnswerEvaluation,
    AnswerKey,
    CaseContract,
    ClaimLabel,
    GatePolicy,
    GateResult,
    HumanReview,
    MachineDiagnostics,
    answer_failure_reasons,
    diagnose_answer,
    evaluate_gate,
    review_binding_errors,
)


Stage = Literal["fixed-diagnostic", "fixed-production", "live-production"]
Disposition = Literal["ran", "skipped"]

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_SCHEMA = 1
STAGES = ("fixed-diagnostic", "fixed-production", "live-production")
FIXED_STAGES = ("fixed-diagnostic", "fixed-production")

_RESULT_FIELDS = {
    "result_schema",
    "status",
    "cache_key",
    "question_id",
    "stage",
    "arm",
    "seed",
    "suite_sha256",
    "case_sha256",
    "evidence_sha256",
    "prompt_sha256",
    "model_identity",
    "generation_settings",
    "messages",
    "generation_result",
    "machine_diagnostics",
    "terminal_error",
    "terminal_content",
    "created_at",
    "wall_seconds",
}
_IDENTITY_FIELDS = {
    "name",
    "digest",
    "ollama_version",
    "format",
    "family",
    "parameter_size",
    "quantization_level",
    "file_type",
    "quantization_version",
}
_GENERATION_RESULT_FIELDS = {
    "content",
    "model",
    "done",
    "done_reason",
    "created_at",
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
}
_DIAGNOSTIC_FIELDS = {
    "empty_answer",
    "corrupt_unicode",
    "think_leak",
    "token_cap_stop",
    "natural_stop",
    "unexpected_entities",
    "unexpected_numbers_dates",
    "missing_required_aliases",
}
_REVIEW_FIELDS = {
    "question_id",
    "stage",
    "arm",
    "seed",
    "claim_labels",
    "critical_failures",
    "required_atom_ids_present",
    "supported_claim_precision",
    "coherence",
    "style_fidelity",
    "reviewer",
    "reviewed_at",
}
_CLAIM_LABEL_FIELDS = {"answer_span", "label", "atom_ids"}
_LOCK_FIELDS = {"lock_schema", "stage", "suite_sha256", "entries"}
_LOCK_ENTRY_FIELDS = {
    "question_id",
    "stage",
    "arm",
    "seed",
    "result_cache_key",
    "answer_sha256",
    "case_sha256",
    "model_identity_sha256",
    "prompt_sha256",
    "generation_settings_sha256",
    "semantic_review_sha256",
}
_RUN_FIELDS = {
    "run_schema",
    "suite_path",
    "suite_id",
    "suite_sha256",
    "model_identity",
}
_GATE_ARTIFACT_FIELDS = {
    "gate_schema",
    "stage",
    "suite_sha256",
    "model_identity_sha256",
    "input_fingerprint",
    "gate",
}
_MOJIBAKE_MARKERS = ("Ã", "Â", "Å", "Ä", "â€", "ðŸ", "ï»¿")


@dataclass(frozen=True)
class StageUnit:
    case: EvalCase
    arm: str
    seed: int
    settings: GenerationSettings


@dataclass(frozen=True)
class UnitSpec:
    suite_sha256: str
    case: EvalCase
    stage: Stage
    arm: str
    seed: int
    identity: ModelIdentity
    settings: GenerationSettings
    messages: tuple[dict[str, str], ...]
    evidence_sha256: str
    prompt_sha256: str
    cache_key: str
    destination: Path


@dataclass(frozen=True)
class UnitExecution:
    disposition: Disposition
    destination: Path
    cache_key: str


@dataclass(frozen=True)
class CaseResult:
    status: Literal["complete", "terminal_output_failure"]
    question_id: str
    stage: str
    arm: str
    seed: int
    suite_sha256: str
    case_sha256: str
    evidence_sha256: str
    prompt_sha256: str
    cache_key: str
    model_identity: ModelIdentity
    generation_settings: GenerationSettings
    messages: tuple[dict[str, str], ...]
    generation_result: GenerationResult | None
    machine_diagnostics: MachineDiagnostics
    terminal_error: dict[str, str] | None
    terminal_content: str | None
    created_at: str
    wall_seconds: float

    def to_json(self) -> dict[str, object]:
        return {
            "result_schema": RESULT_SCHEMA,
            "status": self.status,
            "cache_key": self.cache_key,
            "question_id": self.question_id,
            "stage": self.stage,
            "arm": self.arm,
            "seed": self.seed,
            "suite_sha256": self.suite_sha256,
            "case_sha256": self.case_sha256,
            "evidence_sha256": self.evidence_sha256,
            "prompt_sha256": self.prompt_sha256,
            "model_identity": self.model_identity.to_json(),
            "generation_settings": self.generation_settings.to_json(),
            "messages": [dict(message) for message in self.messages],
            "generation_result": (
                self.generation_result.to_json()
                if self.generation_result is not None
                else None
            ),
            "machine_diagnostics": asdict(self.machine_diagnostics),
            "terminal_error": self.terminal_error,
            "terminal_content": self.terminal_content,
            "created_at": self.created_at,
            "wall_seconds": self.wall_seconds,
        }


def run_fixed_case(
    case: EvalCase,
    arm: str,
    seed: int,
    client: OllamaClient,
    identity: ModelIdentity,
    settings: GenerationSettings,
) -> CaseResult:
    """Generate and diagnose one fixed-card answer without filesystem policy."""

    if not isinstance(case, EvalCase):
        raise TypeError("case must be an EvalCase")
    if arm not in ARM_IDS:
        raise ValueError(f"unknown prompt arm {arm!r}")
    if type(seed) is not int or settings.seed != seed:
        raise ValueError("seed must equal generation settings seed")
    if not isinstance(identity, ModelIdentity):
        raise TypeError("identity must be ModelIdentity")
    if not isinstance(settings, GenerationSettings):
        raise TypeError("settings must be GenerationSettings")
    client_identity = getattr(client, "identity", identity)
    if client_identity != identity:
        raise ValueError("client identity differs from the supplied frozen identity")

    messages = tuple(build_messages(case, case.cards, arm))
    prompt_sha256 = canonical_sha256(messages)
    evidence_sha256 = canonical_sha256([asdict(card) for card in case.cards])
    inferred_stage = (
        "fixed-diagnostic" if settings.top_k is not None else "fixed-production"
    )
    started = time.perf_counter()
    generation_result: GenerationResult | None = None
    terminal_error: dict[str, str] | None = None
    terminal_content: str | None = None
    try:
        generation_result = client.generate(list(messages), settings)
        diagnostics = diagnose_answer(
            case,
            generation_result.content,
            done_reason=generation_result.done_reason,
            eval_count=generation_result.eval_count,
            num_predict=settings.num_predict,
        )
        status: Literal["complete", "terminal_output_failure"] = "complete"
    except EmptyContentError as exc:
        diagnostics = diagnose_answer(
            case,
            "",
            done_reason=None,
            num_predict=settings.num_predict,
        )
        status = "terminal_output_failure"
        terminal_error = {"type": type(exc).__name__, "message": str(exc)}
        terminal_content = ""
    except ThinkingLeakError as exc:
        diagnostics = diagnose_answer(
            case,
            exc.content,
            thinking=exc.thinking,
            done_reason=None,
            num_predict=settings.num_predict,
        )
        status = "terminal_output_failure"
        terminal_error = {"type": type(exc).__name__, "message": str(exc)}
        terminal_content = exc.content

    return CaseResult(
        status=status,
        question_id=case.id,
        stage=inferred_stage,
        arm=arm,
        seed=seed,
        suite_sha256="",
        case_sha256=case.sha256,
        evidence_sha256=evidence_sha256,
        prompt_sha256=prompt_sha256,
        cache_key="",
        model_identity=identity,
        generation_settings=settings,
        messages=messages,
        generation_result=generation_result,
        machine_diagnostics=diagnostics,
        terminal_error=terminal_error,
        terminal_content=terminal_content,
        created_at=_utc_now(),
        wall_seconds=max(0.0, time.perf_counter() - started),
    )


def expand_stage(suite: EvalSuite, stage: Stage) -> tuple[StageUnit, ...]:
    """Expand one stage in fixture case/arm/seed order."""

    if not isinstance(suite, EvalSuite):
        raise TypeError("suite must be an EvalSuite")
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    units: list[StageUnit] = []
    if stage == "fixed-diagnostic":
        settings = _diagnostic_settings(suite)
        for case in suite.cases:
            for arm in ARM_IDS:
                units.append(
                    StageUnit(
                        case=case,
                        arm=arm,
                        seed=settings.seed,
                        settings=settings,
                    )
                )
    else:
        for case in suite.cases:
            for seed in suite.production_generation.seeds:
                units.append(
                    StageUnit(
                        case=case,
                        arm="semantic_plus_style",
                        seed=seed,
                        settings=_production_settings(suite, seed),
                    )
                )
    return tuple(units)


def result_path(
    run_dir: Path,
    stage: Stage,
    question_id: str,
    arm: str,
    seed: int,
) -> Path:
    """Return the one-file-per-answer location for a unit."""

    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    _safe_component(question_id, "question_id")
    if arm not in ARM_IDS:
        raise ValueError(f"unknown arm {arm!r}")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if stage == "fixed-diagnostic":
        prefix = Path("fixed") / "diagnostic"
    elif stage == "fixed-production":
        prefix = Path("fixed") / "production"
    else:
        prefix = Path("live") / "production"
    return Path(run_dir) / prefix / question_id / arm / f"{seed}.json"


def review_path(run_dir: Path, key: AnswerKey) -> Path:
    _safe_component(key.question_id, "question_id")
    return (
        Path(run_dir)
        / "reviews"
        / key.stage
        / key.question_id
        / key.arm
        / f"{key.seed}.json"
    )


def load_review_file(
    path: Path,
    *,
    case: EvalCase,
    expected_key: AnswerKey,
    evaluation: AnswerEvaluation,
) -> HumanReview:
    """Load one strict persisted review and bind it to fixture + exact result."""

    raw = _read_strict_json(path, validate_unicode=True)
    if set(raw) != _REVIEW_FIELDS:
        raise ValueError("review has missing or unexpected fields")
    if (
        raw["question_id"] != expected_key.question_id
        or raw["stage"] != expected_key.stage
        or raw["arm"] != expected_key.arm
        or raw["seed"] != expected_key.seed
    ):
        raise ValueError("review key does not match its expected result/path")
    if expected_key.question_id != case.id or evaluation.key != expected_key:
        raise ValueError("review case/result binding is inconsistent")
    if raw["style_fidelity"] is not None and expected_key.arm == "semantic_only":
        raise ValueError("semantic_only reviews must keep style_fidelity null")

    claim_values = raw["claim_labels"]
    if type(claim_values) is not list:
        raise ValueError("claim_labels must be an array")
    claims: list[ClaimLabel] = []
    for index, value in enumerate(claim_values):
        if type(value) is not dict or set(value) != _CLAIM_LABEL_FIELDS:
            raise ValueError(f"claim_labels[{index}] has invalid fields")
        atom_ids = _strict_string_list(
            value["atom_ids"], f"claim_labels[{index}].atom_ids"
        )
        claims.append(
            ClaimLabel(
                answer_span=value["answer_span"],
                label=value["label"],
                atom_ids=atom_ids,
            )
        )

    critical = frozenset(
        _strict_string_list(raw["critical_failures"], "critical_failures")
    )
    present = frozenset(
        _strict_string_list(
            raw["required_atom_ids_present"], "required_atom_ids_present"
        )
    )
    reviewed_at = raw["reviewed_at"]
    if not _aware_iso8601(reviewed_at):
        raise ValueError("reviewed_at must be timezone-aware ISO-8601")
    style = raw["style_fidelity"]
    if style is not None and type(style) is not int:
        raise ValueError("style_fidelity must be null or an integer")

    contract = CaseContract.from_case(case)
    review = HumanReview(
        key=expected_key,
        result_cache_key=evaluation.result_cache_key,
        answer_sha256=evaluation.answer_sha256,
        case_sha256=evaluation.case_sha256,
        claim_labels=tuple(claims),
        critical_failures=critical,
        required_atom_ids_present=present,
        required_atom_ids_expected=contract.required_atom_ids,
        supported_claim_precision=raw["supported_claim_precision"],
        coherence=raw["coherence"],
        style_fidelity=style,
        reviewer=raw["reviewer"],
        reviewed_at=reviewed_at,
    )
    errors = review_binding_errors(review, evaluation, contract)
    if errors:
        raise ValueError("; ".join(errors))
    return review


def review_semantic_projection(review: HumanReview) -> dict[str, object]:
    """Return every immutable semantic review field, excluding style only."""

    return {
        "question_id": review.key.question_id,
        "stage": review.key.stage,
        "arm": review.key.arm,
        "seed": review.key.seed,
        "claim_labels": [asdict(label) for label in review.claim_labels],
        "critical_failures": [
            failure
            for failure in CRITICAL_FAILURE_ORDER
            if failure in review.critical_failures
        ],
        "required_atom_ids_present": sorted(review.required_atom_ids_present),
        "supported_claim_precision": review.supported_claim_precision,
        "coherence": review.coherence,
        "reviewer": review.reviewer,
        "reviewed_at": review.reviewed_at,
    }


def _expected_keys(suite: EvalSuite, stage: Stage) -> tuple[AnswerKey, ...]:
    return tuple(
        AnswerKey(stage, unit.case.id, unit.arm, unit.seed)
        for unit in expand_stage(suite, stage)
    )


def _gate_policy(
    suite: EvalSuite,
    stage: Stage,
    prerequisite_errors: Sequence[str] = (),
) -> GatePolicy:
    expected = _expected_keys(suite, stage)
    contracts = tuple(CaseContract.from_case(case) for case in suite.cases)
    if stage == "fixed-diagnostic":
        minimum_full_passes = len(expected)
        minimum_passes_per_question = 2
        styled_no_regression = True
    else:
        minimum_full_passes = 42
        minimum_passes_per_question = 2
        styled_no_regression = False
    return GatePolicy(
        gate_id=stage,
        expected_keys=expected,
        case_contracts=contracts,
        minimum_full_passes=minimum_full_passes,
        minimum_passes_per_question=minimum_passes_per_question,
        require_zero_critical_failures=True,
        require_styled_no_regression=styled_no_regression,
        style_min_question_median=2.0,
        style_min_suite_mean=2.0,
        prerequisite_errors=tuple(prerequisite_errors),
    )


def _load_evaluation(
    path: Path,
    *,
    suite: EvalSuite,
    unit: StageUnit,
    stage: Stage,
    identity: ModelIdentity,
) -> tuple[dict[str, Any], AnswerEvaluation, UnitSpec]:
    raw = _read_strict_json(path)
    if not _valid_result_shape(raw):
        raise ValueError("result has invalid schema")
    raw_identity = _model_identity_from_json(raw["model_identity"])
    raw_settings = _generation_settings_from_json(raw["generation_settings"])
    if raw_identity != identity:
        raise ValueError("result model identity differs from run manifest")
    if raw_settings != unit.settings:
        raise ValueError("result settings differ from the frozen stage settings")
    spec = _build_unit_spec(
        suite=suite,
        case=unit.case,
        stage=stage,
        arm=unit.arm,
        seed=unit.seed,
        identity=identity,
        settings=unit.settings,
        destination=path,
    )
    if not _cache_matches(raw, spec):
        raise ValueError("result cache identity is stale")
    if not _cached_diagnostics_match(raw, spec):
        raise ValueError("result diagnostics do not match the exact visible output")
    diagnostics = _diagnostics_from_json(raw["machine_diagnostics"])
    if raw["status"] == "complete":
        answer = raw["generation_result"]["content"]
    else:
        answer = raw["terminal_content"]
    evaluation = AnswerEvaluation(
        key=AnswerKey(stage, unit.case.id, unit.arm, unit.seed),
        answer=answer,
        result_cache_key=raw["cache_key"],
        case_sha256=raw["case_sha256"],
        diagnostics=diagnostics,
    )
    return raw, evaluation, spec


def _lock_entry(
    raw: Mapping[str, Any], review: HumanReview
) -> dict[str, object]:
    return {
        "question_id": review.key.question_id,
        "stage": review.key.stage,
        "arm": review.key.arm,
        "seed": review.key.seed,
        "result_cache_key": review.result_cache_key,
        "answer_sha256": review.answer_sha256,
        "case_sha256": review.case_sha256,
        "model_identity_sha256": canonical_sha256(raw["model_identity"]),
        "prompt_sha256": raw["prompt_sha256"],
        "generation_settings_sha256": canonical_sha256(
            raw["generation_settings"]
        ),
        "semantic_review_sha256": canonical_sha256(
            review_semantic_projection(review)
        ),
    }


def _validate_semantic_lock(
    raw: object,
    *,
    stage: Stage,
    suite_sha256: str,
    expected_keys: tuple[AnswerKey, ...],
) -> dict[str, Any]:
    if type(raw) is not dict or set(raw) != _LOCK_FIELDS:
        raise ValueError("semantic lock has missing or unexpected fields")
    if (
        type(raw["lock_schema"]) is not int
        or raw["lock_schema"] != 1
        or raw["stage"] != stage
        or raw["suite_sha256"] != suite_sha256
    ):
        raise ValueError("semantic lock identity is stale")
    entries = raw["entries"]
    if type(entries) is not list or len(entries) != len(expected_keys):
        raise ValueError("semantic lock must contain the exact answer matrix")
    observed: list[AnswerKey] = []
    for index, entry in enumerate(entries):
        if type(entry) is not dict or set(entry) != _LOCK_ENTRY_FIELDS:
            raise ValueError(f"semantic lock entry {index} has invalid fields")
        try:
            key = AnswerKey(
                entry["stage"],
                entry["question_id"],
                entry["arm"],
                entry["seed"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"semantic lock entry {index} has an invalid key"
            ) from exc
        observed.append(key)
        for field in (
            "result_cache_key",
            "answer_sha256",
            "case_sha256",
            "model_identity_sha256",
            "prompt_sha256",
            "generation_settings_sha256",
            "semantic_review_sha256",
        ):
            if not _is_sha256(entry[field]):
                raise ValueError(
                    f"semantic lock entry {index} has invalid {field}"
                )
    if tuple(observed) != expected_keys:
        raise ValueError("semantic lock answer order/set differs from the fixture")
    return raw


def _semantic_lock_mismatches(
    existing: Mapping[str, Any],
    expected_keys: tuple[AnswerKey, ...],
    current_entries: Mapping[AnswerKey, Mapping[str, object]],
) -> tuple[str, ...]:
    locked = {
        key: entry
        for key, entry in zip(expected_keys, existing["entries"], strict=True)
    }
    return tuple(
        "semantic lock mismatch for "
        f"{key.question_id}/{key.arm}/{key.seed}"
        for key, current in current_entries.items()
        if current != locked[key]
    )


def _semantic_lock_state(
    run_dir: Path,
    *,
    suite: EvalSuite,
    stage: Stage,
    expected_keys: tuple[AnswerKey, ...],
    raw_results: Mapping[AnswerKey, Mapping[str, Any]],
    reviews: Mapping[AnswerKey, HumanReview],
) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    """Create a complete pre-style lock or compare current artifacts to it."""

    path = Path(run_dir) / "reviews" / stage / "semantic-lock.json"
    errors: list[str] = []
    existing: dict[str, Any] | None = None
    if path.exists():
        try:
            existing = _validate_semantic_lock(
                _read_strict_json(path),
                stage=stage,
                suite_sha256=suite.sha256,
                expected_keys=expected_keys,
            )
        except (OSError, ValueError) as exc:
            return None, (f"invalid semantic lock: {exc}",)

    current_entries = {
        key: _lock_entry(raw_results[key], reviews[key])
        for key in expected_keys
        if key in raw_results and key in reviews
    }
    if existing is not None:
        return existing, _semantic_lock_mismatches(
            existing, expected_keys, current_entries
        )

    if any(review.style_fidelity is not None for review in reviews.values()):
        errors.append("style scores exist before semantic reviews were locked")
        return None, tuple(errors)
    if len(current_entries) != len(expected_keys):
        return None, ()

    desired = {
        "lock_schema": 1,
        "stage": stage,
        "suite_sha256": suite.sha256,
        "entries": [current_entries[key] for key in expected_keys],
    }
    try:
        _write_atomic_json_exclusive(path, desired)
    except FileExistsError:
        try:
            winner = _validate_semantic_lock(
                _read_strict_json(path),
                stage=stage,
                suite_sha256=suite.sha256,
                expected_keys=expected_keys,
            )
        except (OSError, ValueError) as exc:
            return None, (f"invalid concurrently-created semantic lock: {exc}",)
        return winner, _semantic_lock_mismatches(
            winner, expected_keys, current_entries
        )
    return desired, ()


def _relative_path(path: Path, run_dir: Path) -> str:
    return Path(path).relative_to(Path(run_dir)).as_posix()


def _stage_root(run_dir: Path, stage: Stage) -> Path:
    if stage == "fixed-diagnostic":
        return Path(run_dir) / "fixed" / "diagnostic"
    if stage == "fixed-production":
        return Path(run_dir) / "fixed" / "production"
    return Path(run_dir) / "live" / "production"


def _unexpected_stage_json(
    run_dir: Path,
    *,
    stage: Stage,
    expected_result_paths: frozenset[Path],
    expected_review_paths: frozenset[Path],
) -> tuple[str, ...]:
    errors: list[str] = []
    result_root = _stage_root(run_dir, stage)
    if result_root.exists():
        for path in sorted(result_root.rglob("*.json")):
            if path not in expected_result_paths:
                errors.append(
                    "unexpected result artifact: " + _relative_path(path, run_dir)
                )
    review_root = Path(run_dir) / "reviews" / stage
    if review_root.exists():
        for path in sorted(review_root.rglob("*.json")):
            if path == review_root / "semantic-lock.json":
                continue
            if path not in expected_review_paths:
                errors.append(
                    "unexpected review artifact: " + _relative_path(path, run_dir)
                )
    return tuple(errors)


def _report_stage(
    run_dir: Path,
    *,
    suite: EvalSuite,
    identity: ModelIdentity,
    stage: Stage,
    prerequisite_errors: Sequence[str] = (),
) -> tuple[dict[str, object], dict[str, object]]:
    units = expand_stage(suite, stage)
    expected_keys = _expected_keys(suite, stage)
    case_by_id = {case.id: case for case in suite.cases}
    raw_results: dict[AnswerKey, dict[str, Any]] = {}
    results: dict[AnswerKey, AnswerEvaluation] = {}
    specs: dict[AnswerKey, UnitSpec] = {}
    reviews: dict[AnswerKey, HumanReview] = {}
    invalid_results: dict[AnswerKey, str] = {}
    invalid_reviews: dict[AnswerKey, str] = {}
    errors = list(prerequisite_errors)
    expected_result_paths: set[Path] = set()
    expected_review_paths: set[Path] = set()

    for unit, key in zip(units, expected_keys, strict=True):
        result_file = result_path(
            run_dir, stage, key.question_id, key.arm, key.seed
        )
        review_file = review_path(run_dir, key)
        expected_result_paths.add(result_file)
        expected_review_paths.add(review_file)
        if result_file.exists():
            try:
                raw, evaluation, spec = _load_evaluation(
                    result_file,
                    suite=suite,
                    unit=unit,
                    stage=stage,
                    identity=identity,
                )
                raw_results[key] = raw
                results[key] = evaluation
                specs[key] = spec
            except (OSError, ValueError, RuntimeError) as exc:
                invalid_results[key] = str(exc)
                errors.append(
                    f"invalid result {key.question_id}/{key.arm}/{key.seed}: {exc}"
                )
        if review_file.exists():
            if key not in results:
                if key not in invalid_results:
                    errors.append(
                        "review exists without its result: "
                        f"{key.question_id}/{key.arm}/{key.seed}"
                    )
                continue
            try:
                reviews[key] = load_review_file(
                    review_file,
                    case=case_by_id[key.question_id],
                    expected_key=key,
                    evaluation=results[key],
                )
            except (OSError, ValueError) as exc:
                invalid_reviews[key] = str(exc)
                errors.append(
                    f"invalid review {key.question_id}/{key.arm}/{key.seed}: {exc}"
                )

    errors.extend(
        _unexpected_stage_json(
            run_dir,
            stage=stage,
            expected_result_paths=frozenset(expected_result_paths),
            expected_review_paths=frozenset(expected_review_paths),
        )
    )
    lock, lock_errors = _semantic_lock_state(
        run_dir,
        suite=suite,
        stage=stage,
        expected_keys=expected_keys,
        raw_results=raw_results,
        reviews=reviews,
    )
    errors.extend(lock_errors)

    policy = _gate_policy(suite, stage, errors)
    gate = evaluate_gate(results, reviews, policy)
    contracts = {
        contract.question_id: contract for contract in policy.case_contracts
    }
    answers: list[dict[str, object]] = []
    for key in expected_keys:
        result_file = result_path(
            run_dir, stage, key.question_id, key.arm, key.seed
        )
        review_file = review_path(run_dir, key)
        evaluation = results.get(key)
        review = reviews.get(key)
        if key in invalid_results:
            status = "invalid-result"
            failure_reasons = ("invalid_result:" + invalid_results[key],)
        elif evaluation is None:
            status = "missing-result"
            failure_reasons = ("missing_result",)
        elif key in invalid_reviews:
            status = "invalid-review"
            failure_reasons = ("invalid_review:" + invalid_reviews[key],)
        elif review is None:
            status = "unreviewed"
            failure_reasons = ("unreviewed",)
        else:
            failure_reasons = answer_failure_reasons(
                review, evaluation, contracts[key.question_id]
            )
            status = "failed" if failure_reasons else "passed"
        raw = raw_results.get(key)
        generation = raw.get("generation_result") if raw is not None else None
        answers.append(
            {
                "question_id": key.question_id,
                "arm": key.arm,
                "seed": key.seed,
                "status": status,
                "answer": evaluation.answer if evaluation is not None else None,
                "answer_sha256": (
                    evaluation.answer_sha256 if evaluation is not None else None
                ),
                "failure_reasons": list(failure_reasons),
                "review_flags": (
                    list(evaluation.diagnostics.review_flags)
                    if evaluation is not None
                    else []
                ),
                "coherence": review.coherence if review is not None else None,
                "supported_claim_precision": (
                    review.supported_claim_precision
                    if review is not None
                    else None
                ),
                "style_fidelity": (
                    review.style_fidelity if review is not None else None
                ),
                "critical_failures": (
                    [
                        item
                        for item in CRITICAL_FAILURE_ORDER
                        if item in review.critical_failures
                    ]
                    if review is not None
                    else []
                ),
                "result_path": _relative_path(result_file, run_dir),
                "review_path": _relative_path(review_file, run_dir),
                "wall_seconds": raw["wall_seconds"] if raw is not None else None,
                "prompt_tokens": (
                    generation.get("prompt_eval_count")
                    if type(generation) is dict
                    else None
                ),
                "output_tokens": (
                    generation.get("eval_count")
                    if type(generation) is dict
                    else None
                ),
            }
        )

    lock_sha256 = canonical_sha256(lock) if lock is not None else None
    input_fingerprint = canonical_sha256(
        {
            "stage": stage,
            "suite_sha256": suite.sha256,
            "model_identity": identity.to_json(),
            "expected_cache_keys": [
                (
                    specs[key].cache_key
                    if key in specs
                    else _build_unit_spec(
                        suite=suite,
                        case=case_by_id[key.question_id],
                        stage=stage,
                        arm=key.arm,
                        seed=key.seed,
                        identity=identity,
                        settings=next(
                            unit.settings
                            for unit in units
                            if unit.case.id == key.question_id
                            and unit.arm == key.arm
                            and unit.seed == key.seed
                        ),
                        destination=result_path(
                            run_dir,
                            stage,
                            key.question_id,
                            key.arm,
                            key.seed,
                        ),
                    ).cache_key
                )
                for key in expected_keys
            ],
            "semantic_lock_sha256": lock_sha256,
        }
    )
    reviewed_values = tuple(reviews.values())
    coherence_scores = [review.coherence for review in reviewed_values]
    total_wall = sum(
        float(raw["wall_seconds"]) for raw in raw_results.values()
    )
    prompt_tokens = sum(
        int(raw["generation_result"]["prompt_eval_count"] or 0)
        for raw in raw_results.values()
        if type(raw.get("generation_result")) is dict
    )
    output_tokens = sum(
        int(raw["generation_result"]["eval_count"] or 0)
        for raw in raw_results.values()
        if type(raw.get("generation_result")) is dict
    )
    stage_summary: dict[str, object] = {
        "id": stage,
        "input_fingerprint": input_fingerprint,
        "prerequisites": {"status": "passed" if not errors else "blocked", "errors": errors},
        "retrieval": {"status": "not-applicable", "recall_at_3": None},
        "semantic": {
            "status": gate.status,
            "expected": gate.expected_count,
            "generated": gate.result_count,
            "reviewed": gate.reviewed_count,
            "full_passes": gate.full_pass_count,
            "minimum_full_passes": policy.minimum_full_passes,
            "minimum_passes_per_question": policy.minimum_passes_per_question,
            "critical_failure_counts": [list(item) for item in gate.critical_failure_counts],
            "styled_regressions": list(gate.styled_regressions),
            "reasons": list(gate.reasons),
        },
        "coherence": {
            "reviewed": len(coherence_scores),
            "mean": (
                sum(coherence_scores) / len(coherence_scores)
                if coherence_scores
                else None
            ),
            "below_2": sum(score < 2 for score in coherence_scores),
        },
        "style": {
            "status": gate.style_status,
            "mean": gate.style_mean,
            "minimum_question_median": policy.style_min_question_median,
            "minimum_suite_mean": policy.style_min_suite_mean,
            "reasons": list(gate.style_reasons),
        },
        "runtime": {
            "wall_seconds": total_wall,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
        },
        "semantic_lock": {
            "status": "locked" if lock is not None and not lock_errors else (
                "blocked" if lock_errors else "pending"
            ),
            "sha256": lock_sha256,
            "path": f"reviews/{stage}/semantic-lock.json",
        },
        "answers": answers,
    }
    gate_artifact: dict[str, object] = {
        "gate_schema": 1,
        "stage": stage,
        "suite_sha256": suite.sha256,
        "model_identity_sha256": canonical_sha256(identity.to_json()),
        "input_fingerprint": input_fingerprint,
        "gate": asdict(gate),
    }
    return stage_summary, gate_artifact


def _build_unit_spec(
    *,
    suite: EvalSuite,
    case: EvalCase,
    stage: Stage,
    arm: str,
    seed: int,
    identity: ModelIdentity,
    settings: GenerationSettings,
    destination: Path,
) -> UnitSpec:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    if settings.seed != seed:
        raise ValueError("unit seed differs from settings seed")
    messages = tuple(build_messages(case, case.cards, arm))
    prompt_sha256 = canonical_sha256(messages)
    evidence_sha256 = canonical_sha256([asdict(card) for card in case.cards])
    arm_identity = {"id": arm, "messages_sha256": prompt_sha256}
    cache_key = canonical_sha256(
        {
            "harness_schema": RESULT_SCHEMA,
            "suite_hash": suite.sha256,
            "case_hash": case.sha256,
            "stage": stage,
            "arm": arm_identity,
            "seed": seed,
            "evidence_hash": evidence_sha256,
            "model": identity.to_json(),
            "generation": settings.to_json(),
        }
    )
    return UnitSpec(
        suite_sha256=suite.sha256,
        case=case,
        stage=stage,
        arm=arm,
        seed=seed,
        identity=identity,
        settings=settings,
        messages=messages,
        evidence_sha256=evidence_sha256,
        prompt_sha256=prompt_sha256,
        cache_key=cache_key,
        destination=Path(destination),
    )


def _execute_unit(
    spec: UnitSpec,
    client: OllamaClient,
    *,
    resume: bool,
) -> UnitExecution:
    """Skip an exact cache hit or atomically replace this unit only."""

    if resume and spec.destination.exists():
        cached = _load_cache_or_quarantine(spec.destination)
        if cached is not None and _cache_matches(cached, spec):
            if _cached_diagnostics_match(cached, spec):
                return UnitExecution("skipped", spec.destination, spec.cache_key)
            _quarantine(spec.destination)

    generated = run_fixed_case(
        spec.case,
        spec.arm,
        spec.seed,
        client,
        spec.identity,
        spec.settings,
    )
    if (
        generated.prompt_sha256 != spec.prompt_sha256
        or generated.evidence_sha256 != spec.evidence_sha256
        or generated.case_sha256 != spec.case.sha256
    ):
        raise RuntimeError("generated unit identity differs from its cache specification")
    persisted = replace(
        generated,
        stage=spec.stage,
        suite_sha256=spec.suite_sha256,
        cache_key=spec.cache_key,
    )
    write_atomic_json(spec.destination, persisted.to_json())
    return UnitExecution("ran", spec.destination, spec.cache_key)


def write_atomic_json(path: Path, value: object) -> None:
    """Write strict UTF-8 JSON through a unique same-directory temporary file."""

    path = Path(path)
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def _write_atomic_json_exclusive(path: Path, value: object) -> None:
    """Atomically create JSON once; never replace an existing destination."""

    path = Path(path)
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _write_atomic_text(path: Path, value: str) -> None:
    path = Path(path)
    if type(value) is not str:
        raise TypeError("atomic text value must be a string")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temp.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def _read_strict_json(
    path: Path, *, validate_unicode: bool = False
) -> dict[str, Any]:
    """Read one duplicate-free, finite, UTF-8 JSON object without mutation."""

    payload = Path(path).read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise ValueError(f"{path} is not valid UTF-8") from exc
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is not strict JSON: {exc}") from exc
    if type(raw) is not dict:
        raise ValueError(f"{path} must contain a JSON object")
    if validate_unicode:
        _validate_unicode_tree(raw)
    return raw


def _validate_unicode_tree(value: object, path: str = "$") -> None:
    if type(value) is str:
        if (
            not unicodedata.is_normalized("NFC", value)
            or "\ufffd" in value
            or any(marker in value for marker in _MOJIBAKE_MARKERS)
            or any(0xD800 <= ord(character) <= 0xDFFF for character in value)
        ):
            raise ValueError(f"{path} contains corrupt or non-NFC Unicode")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_unicode_tree(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            _validate_unicode_tree(key, f"{path}.<key>")
            _validate_unicode_tree(item, f"{path}.{key}")


def _strict_string_list(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise ValueError(f"{label} must be an array")
    if any(type(item) is not str or not item.strip() for item in value):
        raise ValueError(f"{label} must contain non-empty strings")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _model_identity_from_json(value: object) -> ModelIdentity:
    if not _valid_model_identity(value):
        raise ValueError("model identity has invalid fields")
    assert type(value) is dict
    identity = ModelIdentity(**value)
    if identity.digest != OFFICIAL_MODEL_DIGEST:
        raise ValueError("model identity is not the pinned official Qwen3 digest")
    for field, expected in OFFICIAL_MODEL_METADATA.items():
        if getattr(identity, field) != expected:
            raise ValueError(f"model identity has unexpected {field}")
    return identity


def _generation_settings_from_json(value: object) -> GenerationSettings:
    if not _valid_generation_settings(value):
        raise ValueError("generation settings have invalid fields")
    assert type(value) is dict
    options = value["options"]
    assert type(options) is dict
    return GenerationSettings(
        temperature=options["temperature"],
        seed=options["seed"],
        top_p=options["top_p"],
        repeat_penalty=options["repeat_penalty"],
        num_ctx=options["num_ctx"],
        num_predict=options["num_predict"],
        top_k=options.get("top_k"),
    )


def _diagnostics_from_json(value: object) -> MachineDiagnostics:
    if not _valid_diagnostics(value):
        raise ValueError("machine diagnostics have invalid fields")
    assert type(value) is dict
    return MachineDiagnostics(
        empty_answer=value["empty_answer"],
        corrupt_unicode=value["corrupt_unicode"],
        think_leak=value["think_leak"],
        token_cap_stop=value["token_cap_stop"],
        natural_stop=value["natural_stop"],
        unexpected_entities=tuple(value["unexpected_entities"]),
        unexpected_numbers_dates=tuple(value["unexpected_numbers_dates"]),
        missing_required_aliases=tuple(value["missing_required_aliases"]),
    )


def _load_cache_or_quarantine(path: Path) -> dict[str, Any] | None:
    try:
        payload = path.read_bytes()
    except OSError:
        raise
    try:
        text = payload.decode("utf-8")
        raw = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
        if not _valid_result_shape(raw):
            raise ValueError("result has invalid schema")
        return raw
    except (UnicodeError, ValueError, json.JSONDecodeError):
        _quarantine(path)
        return None


def _valid_result_shape(raw: object) -> bool:
    if type(raw) is not dict or set(raw) != _RESULT_FIELDS:
        return False
    scalar_fields_valid = (
        type(raw.get("result_schema")) is int
        and raw.get("result_schema") == RESULT_SCHEMA
        and raw.get("status") in ("complete", "terminal_output_failure")
        and _is_sha256(raw.get("cache_key"))
        and type(raw.get("question_id")) is str
        and raw.get("stage") in STAGES
        and raw.get("arm") in ARM_IDS
        and type(raw.get("seed")) is int
        and _is_sha256(raw.get("suite_sha256"))
        and _is_sha256(raw.get("case_sha256"))
        and _is_sha256(raw.get("evidence_sha256"))
        and _is_sha256(raw.get("prompt_sha256"))
        and _aware_iso8601(raw.get("created_at"))
        and type(raw.get("wall_seconds")) in (int, float)
        and math.isfinite(raw["wall_seconds"])
        and raw["wall_seconds"] >= 0
    )
    if not scalar_fields_valid:
        return False
    if not _valid_model_identity(raw["model_identity"]):
        return False
    if not _valid_generation_settings(raw["generation_settings"]):
        return False
    if not _valid_messages(raw["messages"]):
        return False
    if not _valid_diagnostics(raw["machine_diagnostics"]):
        return False
    if raw["status"] == "complete":
        return (
            raw["terminal_error"] is None
            and raw["terminal_content"] is None
            and _valid_generation_result(
                raw["generation_result"], raw["model_identity"]["name"]
            )
        )
    error_type = (
        raw["terminal_error"].get("type")
        if type(raw["terminal_error"]) is dict
        else None
    )
    terminal_shape_valid = (
        raw["generation_result"] is None
        and type(raw["terminal_content"]) is str
        and not any(
            0xD800 <= ord(character) <= 0xDFFF
            for character in raw["terminal_content"]
        )
        and type(raw["terminal_error"]) is dict
        and set(raw["terminal_error"]) == {"type", "message"}
        and error_type in {"EmptyContentError", "ThinkingLeakError"}
        and all(
            type(raw["terminal_error"][key]) is str
            and bool(raw["terminal_error"][key])
            for key in ("type", "message")
        )
    )
    if not terminal_shape_valid:
        return False
    diagnostics = raw["machine_diagnostics"]
    if diagnostics["natural_stop"] is not False:
        return False
    if error_type == "EmptyContentError":
        return (
            not raw["terminal_content"].strip()
            and diagnostics["empty_answer"] is True
            and diagnostics["think_leak"] is False
        )
    return diagnostics["think_leak"] is True


def _valid_model_identity(value: object) -> bool:
    if type(value) is not dict or set(value) != _IDENTITY_FIELDS:
        return False
    return (
        value.get("name") == "qwen3:14b"
        and _is_sha256(value.get("digest"))
        and all(
            type(value.get(key)) is str and bool(value[key])
            for key in (
                "ollama_version",
                "format",
                "family",
                "parameter_size",
                "quantization_level",
            )
        )
        and all(
            value.get(key) is None
            or (type(value[key]) is int and value[key] >= 0)
            for key in ("file_type", "quantization_version")
        )
    )


def _valid_generation_settings(value: object) -> bool:
    if type(value) is not dict or set(value) != {
        "think",
        "stream",
        "keep_alive",
        "options",
    }:
        return False
    if (
        value["think"] is not False
        or value["stream"] is not False
        or type(value["keep_alive"]) is not int
        or value["keep_alive"] != 0
    ):
        return False
    options = value["options"]
    if type(options) is not dict:
        return False
    required = {
        "temperature",
        "seed",
        "top_p",
        "repeat_penalty",
        "num_ctx",
        "num_predict",
    }
    if not required <= set(options) or set(options) - required not in (set(), {"top_k"}):
        return False
    return (
        all(
            type(options[key]) in (int, float) and math.isfinite(options[key])
            for key in ("temperature", "top_p", "repeat_penalty")
        )
        and type(options["seed"]) is int
        and all(
            type(options[key]) is int and options[key] > 0
            for key in ("num_ctx", "num_predict")
        )
        and (
            "top_k" not in options
            or (type(options["top_k"]) is int and options["top_k"] > 0)
        )
    )


def _valid_messages(value: object) -> bool:
    return (
        type(value) is list
        and bool(value)
        and all(
            type(message) is dict
            and set(message) == {"role", "content"}
            and message["role"] in ("system", "user", "assistant")
            and type(message["content"]) is str
            and bool(message["content"].strip())
            for message in value
        )
    )


def _valid_diagnostics(value: object) -> bool:
    if type(value) is not dict or set(value) != _DIAGNOSTIC_FIELDS:
        return False
    return all(
        type(value[key]) is bool
        for key in (
            "empty_answer",
            "corrupt_unicode",
            "think_leak",
            "token_cap_stop",
            "natural_stop",
        )
    ) and all(
        type(value[key]) is list
        and all(type(item) is str and bool(item) for item in value[key])
        for key in (
            "unexpected_entities",
            "unexpected_numbers_dates",
            "missing_required_aliases",
        )
    )


def _valid_generation_result(value: object, model_name: object) -> bool:
    if type(value) is not dict or set(value) != _GENERATION_RESULT_FIELDS:
        return False
    if (
        type(value["content"]) is not str
        or not value["content"].strip()
        or any(0xD800 <= ord(character) <= 0xDFFF for character in value["content"])
        or value["model"] != model_name
        or value["done"] is not True
        or (
            value["done_reason"] is not None
            and type(value["done_reason"]) is not str
        )
        or (
            value["created_at"] is not None
            and type(value["created_at"]) is not str
        )
    ):
        return False
    return all(
        value[key] is None or (type(value[key]) is int and value[key] >= 0)
        for key in (
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        )
    )


def _aware_iso8601(value: object) -> bool:
    if type(value) is not str or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def _cache_matches(raw: dict[str, Any], spec: UnitSpec) -> bool:
    expected = {
        "cache_key": spec.cache_key,
        "question_id": spec.case.id,
        "stage": spec.stage,
        "arm": spec.arm,
        "seed": spec.seed,
        "suite_sha256": spec.suite_sha256,
        "case_sha256": spec.case.sha256,
        "evidence_sha256": spec.evidence_sha256,
        "prompt_sha256": spec.prompt_sha256,
        "model_identity": spec.identity.to_json(),
        "generation_settings": spec.settings.to_json(),
        "messages": [dict(message) for message in spec.messages],
    }
    return all(raw.get(key) == value for key, value in expected.items())


def _cached_diagnostics_match(raw: dict[str, Any], spec: UnitSpec) -> bool:
    if raw["status"] == "terminal_output_failure":
        error_type = raw["terminal_error"]["type"]
        recomputed = diagnose_answer(
            spec.case,
            raw["terminal_content"],
            thinking=("<redacted-thinking>" if error_type == "ThinkingLeakError" else None),
            done_reason=None,
            num_predict=spec.settings.num_predict,
        )
        return canonical_sha256(asdict(recomputed)) == canonical_sha256(
            raw["machine_diagnostics"]
        )

    generation = raw["generation_result"]
    recomputed = diagnose_answer(
        spec.case,
        generation["content"],
        done_reason=generation["done_reason"],
        eval_count=generation["eval_count"],
        num_predict=raw["generation_settings"]["options"]["num_predict"],
    )
    return canonical_sha256(asdict(recomputed)) == canonical_sha256(
        raw["machine_diagnostics"]
    )


def _quarantine(path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    candidate = path.with_name(f"{path.name}.invalid-{timestamp}")
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.invalid-{timestamp}-{counter}")
        counter += 1
    path.replace(candidate)
    return candidate


def _diagnostic_settings(suite: EvalSuite) -> GenerationSettings:
    raw = suite.diagnostic_generation
    return GenerationSettings(
        temperature=raw.temperature,
        seed=raw.seed,
        top_p=raw.top_p,
        top_k=raw.top_k,
        repeat_penalty=raw.repeat_penalty,
        num_ctx=raw.num_ctx,
        num_predict=raw.num_predict,
    )


def _production_settings(suite: EvalSuite, seed: int) -> GenerationSettings:
    raw = suite.production_generation
    return GenerationSettings(
        temperature=raw.temperature,
        seed=seed,
        top_p=raw.top_p,
        top_k=None,
        repeat_penalty=raw.repeat_penalty,
        num_ctx=raw.num_ctx,
        num_predict=raw.num_predict,
    )


def _safe_component(value: str, label: str) -> None:
    if (
        type(value) is not str
        or not value
        or value in (".", "..")
        or "/" in value
        or "\\" in value
        or Path(value).name != value
    ):
        raise ValueError(f"unsafe {label} path component")


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON value {value}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_event(path: Path, event: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        dict(event), ensure_ascii=False, sort_keys=True, allow_nan=False
    )
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()


def _ensure_run_manifest(
    path: Path,
    suite: EvalSuite,
    suite_path: Path,
    identity: ModelIdentity,
) -> dict[str, object]:
    desired = {
        "run_schema": 1,
        "suite_path": str(Path(suite_path).resolve()),
        "suite_id": suite.suite_id,
        "suite_sha256": suite.sha256,
        "model_identity": identity.to_json(),
    }
    if path.exists():
        try:
            raw = json.loads(
                path.read_bytes().decode("utf-8"),
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_constant,
            )
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("existing run manifest is unreadable") from exc
        if raw != desired:
            raise ValueError(
                "existing run directory has an incompatible suite/model manifest"
            )
        return desired
    write_atomic_json(path, desired)
    return desired


def _report_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Verified-card evaluation report",
        "",
        f"- Suite: `{summary['suite']['id']}`",
        f"- Fixture SHA-256: `{summary['suite']['fixture_sha256']}`",
        f"- Provenance: **{summary['suite']['provenance']['status']}**",
        f"- Model: `{summary['model']['name']}`",
        f"- Model digest: `{summary['model']['digest']}`",
        "",
    ]
    provenance_errors = summary["suite"]["provenance"]["errors"]
    if provenance_errors:
        lines.extend(["## Provenance failures", ""])
        lines.extend(f"- {error}" for error in provenance_errors)
        lines.append("")

    for stage in summary["stages"]:
        semantic = stage["semantic"]
        style = stage["style"]
        coherence = stage["coherence"]
        retrieval = stage["retrieval"]
        runtime = stage["runtime"]
        lines.extend(
            [
                f"## {stage['id']}",
                "",
                f"- Retrieval: **{retrieval['status']}**",
                f"- Semantic: **{semantic['status']}** "
                f"({semantic['full_passes']}/{semantic['expected']} full passes)",
                f"- Coherence: {coherence['mean'] if coherence['mean'] is not None else 'unreviewed'} "
                f"({coherence['below_2']} below 2/3)",
                f"- Style: **{style['status']}** "
                f"(mean {style['mean'] if style['mean'] is not None else 'unreviewed'})",
                f"- Runtime: {runtime['wall_seconds']:.6f}s; "
                f"{runtime['prompt_tokens']} prompt tokens; "
                f"{runtime['output_tokens']} output tokens",
                f"- Semantic lock: **{stage['semantic_lock']['status']}**",
                "",
            ]
        )
        prerequisite_reasons = stage["prerequisites"]["errors"]
        semantic_reasons = [
            reason
            for reason in semantic["reasons"]
            if reason not in prerequisite_reasons
        ]
        style_reasons = style["reasons"]
        if prerequisite_reasons or semantic_reasons or style_reasons:
            lines.extend(["### Gate reasons", ""])
            lines.extend(
                f"- Prerequisite: {reason}" for reason in prerequisite_reasons
            )
            lines.extend(f"- Semantic: {reason}" for reason in semantic_reasons)
            lines.extend(f"- Style: {reason}" for reason in style_reasons)
            lines.append("")
        failures = [
            answer
            for answer in stage["answers"]
            if answer["failure_reasons"]
        ]
        if failures:
            lines.extend(["### Answer status and failures", ""])
            for answer in failures:
                key = (
                    f"{answer['question_id']}/{answer['arm']}/{answer['seed']}"
                )
                reasons = "; ".join(answer["failure_reasons"])
                lines.append(f"- `{key}` ({answer['status']}): {reasons}")
            lines.append("")
    decision = summary["decision"]
    lines.extend(
        [
            "## Decision",
            "",
            f"- Fixed diagnostic generator: **{decision['fixed_diagnostic']}**",
            f"- Production stability: **{decision['fixed_production']}**",
            f"- Retrieval: **{decision['retrieval']}**",
            f"- Live end-to-end: **{decision['live_end_to_end']}**",
            f"- Style: **{decision['style']}**",
            "",
        ]
    )
    return "\n".join(lines)


def generate_report(run_dir: Path) -> dict[str, object]:
    """Regenerate deterministic fixed-stage gates and reports without Ollama."""

    run_dir = Path(run_dir)
    manifest_path = run_dir / "run.json"
    manifest = _read_strict_json(manifest_path, validate_unicode=True)
    if (
        set(manifest) != _RUN_FIELDS
        or type(manifest.get("run_schema")) is not int
        or manifest.get("run_schema") != 1
    ):
        raise ValueError("run manifest has missing or unexpected fields")
    if any(
        type(manifest.get(field)) is not str or not manifest[field]
        for field in ("suite_path", "suite_id", "suite_sha256")
    ):
        raise ValueError("run manifest suite identity is invalid")
    identity = _model_identity_from_json(manifest["model_identity"])
    suite_path = Path(manifest["suite_path"])
    suite = load_suite(suite_path)
    provenance = validate_suite(suite, REPO_ROOT)
    provenance_errors = list(provenance.errors)
    if suite.suite_id != manifest["suite_id"]:
        provenance_errors.append("run manifest suite_id is stale")
    if suite.sha256 != manifest["suite_sha256"]:
        provenance_errors.append("run manifest fixture SHA-256 is stale")
    if suite.model_policy.required_model != identity.name:
        provenance_errors.append("fixture model policy differs from run identity")

    diagnostic, diagnostic_gate = _report_stage(
        run_dir,
        suite=suite,
        identity=identity,
        stage="fixed-diagnostic",
        prerequisite_errors=provenance_errors,
    )
    production_prerequisites = list(provenance_errors)
    if diagnostic["semantic"]["status"] != "passed":
        production_prerequisites.append(
            "fixed diagnostic semantic gate is not currently passed"
        )
    production, production_gate = _report_stage(
        run_dir,
        suite=suite,
        identity=identity,
        stage="fixed-production",
        prerequisite_errors=production_prerequisites,
    )

    preferred_style = (
        production["style"]["status"]
        if production["semantic"]["generated"]
        else diagnostic["style"]["status"]
    )
    summary: dict[str, object] = {
        "report_schema": 1,
        "suite": {
            "id": suite.suite_id,
            "fixture_sha256": suite.sha256,
            "provenance": {
                "status": "passed" if not provenance_errors else "failed",
                "errors": provenance_errors,
            },
        },
        "model": identity.to_json(),
        "stages": [diagnostic, production],
        "decision": {
            "fixed_diagnostic": diagnostic["semantic"]["status"],
            "fixed_production": production["semantic"]["status"],
            "retrieval": "not-run",
            "live_end_to_end": "not-run",
            "style": preferred_style,
        },
    }
    if set(diagnostic_gate) != _GATE_ARTIFACT_FIELDS or set(
        production_gate
    ) != _GATE_ARTIFACT_FIELDS:
        raise RuntimeError("internal gate artifact schema mismatch")
    write_atomic_json(run_dir / "fixed" / "diagnostic-gate.json", diagnostic_gate)
    write_atomic_json(run_dir / "fixed" / "production-gate.json", production_gate)
    write_atomic_json(run_dir / "summary.json", summary)
    _write_atomic_text(run_dir / "report.md", _report_markdown(summary))
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--suite", type=Path, required=True)
    validate_parser.add_argument("--offline", action="store_true")
    validate_parser.add_argument("--model")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--stage", choices=FIXED_STAGES, required=True)
    run_parser.add_argument("--suite", type=Path, required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--run-dir", type=Path, required=True)
    run_parser.add_argument("--resume", action="store_true")
    run_parser.add_argument("--case", action="append", default=[])

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "validate":
        return _command_validate(args)
    if args.command == "run":
        return _command_run(args)
    return _command_report(args)


def _validated_suite(path: Path) -> EvalSuite:
    suite = load_suite(path)
    report = validate_suite(suite, REPO_ROOT)
    print(
        f"{len(suite.cases)} cases; {len(report.errors)} provenance errors; "
        f"suite hash {suite.sha256}"
    )
    if not report.ok:
        for error in report.errors:
            print(error, file=sys.stderr)
        raise ValueError("suite provenance validation failed")
    return suite


def _command_validate(args) -> int:
    try:
        suite = _validated_suite(args.suite)
        if not args.offline:
            model = args.model or suite.model_policy.required_model
            identity = OllamaClient().preflight(model)
            print(
                f"model {identity.name}; digest {identity.digest}; "
                f"Ollama {identity.ollama_version}"
            )
        return 0
    except (OSError, ValueError, RuntimeError, OllamaError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 2


def _command_run(args) -> int:
    try:
        suite = _validated_suite(args.suite)
        if args.model != suite.model_policy.required_model:
            raise ValueError("--model differs from suite model policy")
        selected = set(args.case)
        known = {case.id for case in suite.cases}
        unknown = sorted(selected - known)
        if unknown:
            raise ValueError("unknown case IDs: " + ", ".join(unknown))

        run_dir = Path(args.run_dir)
        if args.stage == "fixed-production":
            summary = generate_report(run_dir)
            diagnostic = summary["stages"][0]
            if diagnostic["semantic"]["status"] != "passed":
                raise ValueError(
                    "fixed-production is blocked until the current fixed-diagnostic "
                    "semantic gate passes"
                )
        client = OllamaClient()
        identity = client.preflight(args.model)
        _ensure_run_manifest(
            run_dir / "run.json", suite, args.suite, identity
        )
        events_path = run_dir / "events.jsonl"
        for unit in expand_stage(suite, args.stage):
            if selected and unit.case.id not in selected:
                continue
            destination = result_path(
                run_dir,
                args.stage,
                unit.case.id,
                unit.arm,
                unit.seed,
            )
            spec = _build_unit_spec(
                suite=suite,
                case=unit.case,
                stage=args.stage,
                arm=unit.arm,
                seed=unit.seed,
                identity=identity,
                settings=unit.settings,
                destination=destination,
            )
            try:
                execution = _execute_unit(spec, client, resume=args.resume)
            except (OSError, ValueError, RuntimeError, OllamaError) as exc:
                _append_event(
                    events_path,
                    {
                        "event": "failed",
                        "at": _utc_now(),
                        "stage": args.stage,
                        "question_id": unit.case.id,
                        "arm": unit.arm,
                        "seed": unit.seed,
                        "cache_key": spec.cache_key,
                        "error_type": type(exc).__name__,
                    },
                )
                raise
            event = {
                "event": execution.disposition,
                "at": _utc_now(),
                "stage": args.stage,
                "question_id": unit.case.id,
                "arm": unit.arm,
                "seed": unit.seed,
                "cache_key": spec.cache_key,
                "path": str(destination),
            }
            _append_event(events_path, event)
            print(
                f"{execution.disposition}: {unit.case.id}/{unit.arm}/{unit.seed}",
                flush=True,
            )
        return 0
    except (OSError, ValueError, RuntimeError, OllamaError) as exc:
        print(f"run failed: {exc}", file=sys.stderr)
        return 2


def _command_report(args) -> int:
    try:
        summary = generate_report(Path(args.run_dir))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"report failed: {exc}", file=sys.stderr)
        return 2
    print(
        "fixed diagnostic "
        f"{summary['decision']['fixed_diagnostic']}; "
        f"fixed production {summary['decision']['fixed_production']}; "
        f"style {summary['decision']['style']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
