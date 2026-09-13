"""Strict schema and provenance checks for verified-card evaluation suites.

Loading is deliberately limited to syntax, types, structural invariants, and text
integrity.  Checks that depend on repository files are collected by
``validate_suite`` so callers can display every provenance defect at once.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


class SuiteValidationError(ValueError):
    """Raised when a suite cannot be decoded or does not match the strict schema."""


@dataclass(frozen=True)
class ValidationReport:
    """Repository-backed validation errors for an otherwise well-formed suite."""

    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class CardSource:
    video_id: str
    title: str
    start: float
    score: float


@dataclass(frozen=True)
class CardSnapshot:
    id: str
    q: str
    take: str
    support: float
    sources: tuple[CardSource, ...]


@dataclass(frozen=True)
class EvidenceRef:
    ref_id: str
    clean_file: str
    video_id: str
    start: float
    end: float
    excerpt: str
    clean_sha256: str


@dataclass(frozen=True)
class ClaimAtom:
    id: str
    required: bool
    weight: int
    statement: str
    provenance_ref_ids: tuple[str, ...]


@dataclass(frozen=True)
class EvalCase:
    id: str
    kind: Literal["grounded", "abstain"]
    question: str
    card_ids: tuple[str, ...]
    evidence: tuple[EvidenceRef, ...]
    atoms: tuple[ClaimAtom, ...]
    forbidden_claims: tuple[str, ...]
    retrieval_relevant_ids: tuple[str, ...]
    cards: tuple[CardSnapshot, ...] = ()
    allowed_entities: tuple[str, ...] = ()
    allowed_numbers_dates: tuple[str, ...] = ()
    packet_sha256: str = ""

    @property
    def sha256(self) -> str:
        """Canonical digest of the complete parsed case, including its packet hash."""

        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class ModelPolicy:
    required_model: str
    forbid_abliterated: bool


@dataclass(frozen=True)
class DiagnosticGeneration:
    think: bool
    temperature: float
    seed: int
    top_p: float
    top_k: int
    repeat_penalty: float
    num_ctx: int
    num_predict: int
    stream: bool
    keep_alive: int


@dataclass(frozen=True)
class ProductionGeneration:
    think: bool
    temperature: float
    seeds: tuple[int, ...]
    top_p: float
    repeat_penalty: float
    num_ctx: int
    num_predict: int
    stream: bool
    keep_alive: int


@dataclass(frozen=True)
class EvalSuite:
    schema_version: int
    suite_id: str
    model_policy: ModelPolicy
    diagnostic_generation: DiagnosticGeneration
    production_generation: ProductionGeneration
    cases: tuple[EvalCase, ...]

    @property
    def sha256(self) -> str:
        """Canonical digest of the complete parsed suite."""

        return canonical_sha256(asdict(self))


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_MOJIBAKE_MARKERS = (
    "Ã",
    "Â",
    "Å",
    "Ä",
    "â€",
    "ðŸ",
    "ï»¿",
)

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "suite_id",
    "model_policy",
    "diagnostic_generation",
    "production_generation",
    "cases",
}
_MODEL_POLICY_FIELDS = {"required_model", "forbid_abliterated"}
_DIAGNOSTIC_FIELDS = {
    "think",
    "temperature",
    "seed",
    "top_p",
    "top_k",
    "repeat_penalty",
    "num_ctx",
    "num_predict",
    "stream",
    "keep_alive",
}
_PRODUCTION_FIELDS = {
    "think",
    "temperature",
    "seeds",
    "top_p",
    "repeat_penalty",
    "num_ctx",
    "num_predict",
    "stream",
    "keep_alive",
}
_CASE_FIELDS = {
    "id",
    "kind",
    "question",
    "card_ids",
    "cards",
    "evidence",
    "atoms",
    "forbidden_claims",
    "allowed_entities",
    "allowed_numbers_dates",
    "retrieval_relevant_ids",
    "packet_sha256",
}
_CARD_FIELDS = {"id", "q", "take", "support", "sources"}
_CARD_SOURCE_FIELDS = {"video_id", "title", "start", "score"}
_EVIDENCE_FIELDS = {
    "ref_id",
    "clean_file",
    "video_id",
    "start",
    "end",
    "excerpt",
    "clean_sha256",
}
_ATOM_FIELDS = {
    "id",
    "required",
    "weight",
    "statement",
    "provenance_ref_ids",
}


def canonical_sha256(value: object) -> str:
    """Hash a JSON-like value after newline and NFC normalization."""

    normalized = _normalize_newlines_and_nfc(value)
    blob = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _normalize_newlines_and_nfc(value: object) -> object:
    if isinstance(value, str):
        return unicodedata.normalize(
            "NFC", value.replace("\r\n", "\n").replace("\r", "\n")
        )
    if isinstance(value, dict):
        return {key: _normalize_newlines_and_nfc(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_newlines_and_nfc(item) for item in value]
    return value


def load_suite(path: Path) -> EvalSuite:
    """Decode ``path`` as strict UTF-8 JSON and parse an immutable suite."""

    path = Path(path)
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise SuiteValidationError(f"cannot read suite {path}: {exc}") from exc

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SuiteValidationError(f"suite is not valid UTF-8: {exc}") from exc

    try:
        raw = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except SuiteValidationError:
        raise
    except json.JSONDecodeError as exc:
        raise SuiteValidationError(
            f"suite is not valid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    _validate_unicode_tree(raw, "$")
    return _parse_suite(raw)


def validate_suite(suite: EvalSuite, repo_root: Path) -> ValidationReport:
    """Return all repository-backed provenance errors without raising."""

    if not isinstance(suite, EvalSuite):
        raise TypeError("suite must be an EvalSuite returned by load_suite")

    root = Path(repo_root).resolve()
    errors: list[str] = []
    canonical_cards = _load_canonical_cards(root, errors)
    clean_cache: dict[Path, tuple[str | None, Any | None, str | None]] = {}

    for case in suite.cases:
        prefix = f"case {case.id}"
        expected_packet_hash = canonical_sha256(_case_packet_value(case))
        if case.packet_sha256 != expected_packet_hash:
            errors.append(
                f"{prefix}: packet_sha256 mismatch "
                f"(declared {case.packet_sha256}, computed {expected_packet_hash})"
            )

        snapshot_ids = tuple(card.id for card in case.cards)
        if case.card_ids != snapshot_ids:
            errors.append(
                f"{prefix}: card_ids do not match card snapshot IDs in the same order"
            )

        unknown_retrieval_ids = [
            card_id
            for card_id in case.retrieval_relevant_ids
            if card_id not in case.card_ids
        ]
        if unknown_retrieval_ids:
            errors.append(
                f"{prefix}: retrieval_relevant_ids reference cards outside card_ids: "
                f"{', '.join(unknown_retrieval_ids)}"
            )

        for snapshot in case.cards:
            canonical = canonical_cards.get(snapshot.id)
            if canonical is None:
                errors.append(f"{prefix}: canonical card {snapshot.id!r} was not found")
                continue
            if asdict(snapshot) != canonical:
                errors.append(
                    f"{prefix}: card snapshot mismatch for canonical card {snapshot.id}"
                )

        evidence_ids = {evidence.ref_id for evidence in case.evidence}
        for atom in case.atoms:
            missing_refs = [
                ref_id
                for ref_id in atom.provenance_ref_ids
                if ref_id not in evidence_ids
            ]
            if not atom.provenance_ref_ids:
                errors.append(
                    f"{prefix}: atom {atom.id!r} has no provenance_ref_ids"
                )
            if missing_refs:
                qualifier = "required atom" if atom.required else "atom"
                errors.append(
                    f"{prefix}: {qualifier} {atom.id!r} references unknown evidence: "
                    f"{', '.join(missing_refs)}"
                )

        for evidence in case.evidence:
            _validate_evidence(evidence, prefix, root, clean_cache, errors)

    return ValidationReport(errors=tuple(errors))


def _parse_suite(value: object) -> EvalSuite:
    obj = _strict_object(value, _TOP_LEVEL_FIELDS, "$")
    schema_version = _integer(obj["schema_version"], "$.schema_version")
    if schema_version != 1:
        raise SuiteValidationError("$.schema_version must be exactly 1")

    cases_value = _array(obj["cases"], "$.cases", nonempty=True)
    cases = tuple(_parse_case(item, f"$.cases[{index}]") for index, item in enumerate(cases_value))
    _require_unique((case.id for case in cases), "$.cases IDs")

    return EvalSuite(
        schema_version=schema_version,
        suite_id=_string(obj["suite_id"], "$.suite_id"),
        model_policy=_parse_model_policy(obj["model_policy"], "$.model_policy"),
        diagnostic_generation=_parse_diagnostic_generation(
            obj["diagnostic_generation"], "$.diagnostic_generation"
        ),
        production_generation=_parse_production_generation(
            obj["production_generation"], "$.production_generation"
        ),
        cases=cases,
    )


def _parse_model_policy(value: object, path: str) -> ModelPolicy:
    obj = _strict_object(value, _MODEL_POLICY_FIELDS, path)
    required_model = _string(obj["required_model"], f"{path}.required_model")
    if required_model != "qwen3:14b":
        raise SuiteValidationError(
            f"{path}.required_model must be exactly 'qwen3:14b'"
        )
    forbid_abliterated = _boolean(obj["forbid_abliterated"], f"{path}.forbid_abliterated")
    if not forbid_abliterated:
        raise SuiteValidationError(f"{path}.forbid_abliterated must be true")
    return ModelPolicy(
        required_model=required_model,
        forbid_abliterated=forbid_abliterated,
    )


def _parse_diagnostic_generation(value: object, path: str) -> DiagnosticGeneration:
    obj = _strict_object(value, _DIAGNOSTIC_FIELDS, path)
    think = _boolean(obj["think"], f"{path}.think")
    stream = _boolean(obj["stream"], f"{path}.stream")
    keep_alive = _integer(obj["keep_alive"], f"{path}.keep_alive")
    if think or stream or keep_alive != 0:
        raise SuiteValidationError(
            f"{path} must set think=false, stream=false, and keep_alive=0"
        )

    temperature = _number(obj["temperature"], f"{path}.temperature")
    top_p = _number(obj["top_p"], f"{path}.top_p")
    repeat_penalty = _number(obj["repeat_penalty"], f"{path}.repeat_penalty")
    top_k = _positive_integer(obj["top_k"], f"{path}.top_k")
    num_ctx = _positive_integer(obj["num_ctx"], f"{path}.num_ctx")
    num_predict = _positive_integer(obj["num_predict"], f"{path}.num_predict")
    _validate_generation_ranges(temperature, top_p, repeat_penalty, path)
    return DiagnosticGeneration(
        think=think,
        temperature=temperature,
        seed=_integer(obj["seed"], f"{path}.seed"),
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        num_ctx=num_ctx,
        num_predict=num_predict,
        stream=stream,
        keep_alive=keep_alive,
    )


def _parse_production_generation(value: object, path: str) -> ProductionGeneration:
    obj = _strict_object(value, _PRODUCTION_FIELDS, path)
    think = _boolean(obj["think"], f"{path}.think")
    stream = _boolean(obj["stream"], f"{path}.stream")
    keep_alive = _integer(obj["keep_alive"], f"{path}.keep_alive")
    if think or stream or keep_alive != 0:
        raise SuiteValidationError(
            f"{path} must set think=false, stream=false, and keep_alive=0"
        )

    temperature = _number(obj["temperature"], f"{path}.temperature")
    top_p = _number(obj["top_p"], f"{path}.top_p")
    repeat_penalty = _number(obj["repeat_penalty"], f"{path}.repeat_penalty")
    num_ctx = _positive_integer(obj["num_ctx"], f"{path}.num_ctx")
    num_predict = _positive_integer(obj["num_predict"], f"{path}.num_predict")
    _validate_generation_ranges(temperature, top_p, repeat_penalty, path)

    seeds_value = _array(obj["seeds"], f"{path}.seeds", nonempty=True)
    seeds = tuple(
        _integer(seed, f"{path}.seeds[{index}]")
        for index, seed in enumerate(seeds_value)
    )
    _require_unique(seeds, f"{path}.seeds")
    return ProductionGeneration(
        think=think,
        temperature=temperature,
        seeds=seeds,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
        num_ctx=num_ctx,
        num_predict=num_predict,
        stream=stream,
        keep_alive=keep_alive,
    )


def _validate_generation_ranges(
    temperature: float, top_p: float, repeat_penalty: float, path: str
) -> None:
    if temperature < 0:
        raise SuiteValidationError(f"{path}.temperature must be non-negative")
    if not 0 < top_p <= 1:
        raise SuiteValidationError(f"{path}.top_p must be in (0, 1]")
    if repeat_penalty <= 0:
        raise SuiteValidationError(f"{path}.repeat_penalty must be positive")


def _parse_case(value: object, path: str) -> EvalCase:
    obj = _strict_object(value, _CASE_FIELDS, path)
    kind = _string(obj["kind"], f"{path}.kind")
    if kind not in ("grounded", "abstain"):
        raise SuiteValidationError(f"{path}.kind must be 'grounded' or 'abstain'")

    card_ids = _string_tuple(obj["card_ids"], f"{path}.card_ids", nonempty=True)
    cards_value = _array(obj["cards"], f"{path}.cards", nonempty=True)
    cards = tuple(
        _parse_card(item, f"{path}.cards[{index}]")
        for index, item in enumerate(cards_value)
    )
    evidence_value = _array(obj["evidence"], f"{path}.evidence", nonempty=True)
    evidence = tuple(
        _parse_evidence(item, f"{path}.evidence[{index}]")
        for index, item in enumerate(evidence_value)
    )
    atoms_value = _array(obj["atoms"], f"{path}.atoms", nonempty=True)
    atoms = tuple(
        _parse_atom(item, f"{path}.atoms[{index}]")
        for index, item in enumerate(atoms_value)
    )
    retrieval_ids = _string_tuple(
        obj["retrieval_relevant_ids"],
        f"{path}.retrieval_relevant_ids",
        nonempty=True,
    )

    _require_unique(card_ids, f"{path}.card_ids")
    _require_unique((card.id for card in cards), f"{path}.cards IDs")
    _require_unique((item.ref_id for item in evidence), f"{path}.evidence ref_ids")
    _require_unique((atom.id for atom in atoms), f"{path}.atom IDs")
    _require_unique(retrieval_ids, f"{path}.retrieval_relevant_ids")

    return EvalCase(
        id=_string(obj["id"], f"{path}.id"),
        kind=kind,
        question=_string(obj["question"], f"{path}.question"),
        card_ids=card_ids,
        evidence=evidence,
        atoms=atoms,
        forbidden_claims=_string_tuple(
            obj["forbidden_claims"], f"{path}.forbidden_claims"
        ),
        retrieval_relevant_ids=retrieval_ids,
        cards=cards,
        allowed_entities=_string_tuple(
            obj["allowed_entities"], f"{path}.allowed_entities"
        ),
        allowed_numbers_dates=_string_tuple(
            obj["allowed_numbers_dates"], f"{path}.allowed_numbers_dates"
        ),
        packet_sha256=_sha256(obj["packet_sha256"], f"{path}.packet_sha256"),
    )


def _parse_card(value: object, path: str) -> CardSnapshot:
    obj = _strict_object(value, _CARD_FIELDS, path)
    sources_value = _array(obj["sources"], f"{path}.sources", nonempty=True)
    sources = tuple(
        _parse_card_source(item, f"{path}.sources[{index}]")
        for index, item in enumerate(sources_value)
    )
    return CardSnapshot(
        id=_string(obj["id"], f"{path}.id"),
        q=_string(obj["q"], f"{path}.q"),
        take=_string(obj["take"], f"{path}.take"),
        support=_number(obj["support"], f"{path}.support"),
        sources=sources,
    )


def _parse_card_source(value: object, path: str) -> CardSource:
    obj = _strict_object(value, _CARD_SOURCE_FIELDS, path)
    start = _number(obj["start"], f"{path}.start")
    if start < 0:
        raise SuiteValidationError(f"{path}.start must be non-negative")
    return CardSource(
        video_id=_string(obj["video_id"], f"{path}.video_id"),
        title=_string(obj["title"], f"{path}.title"),
        start=start,
        score=_number(obj["score"], f"{path}.score"),
    )


def _parse_evidence(value: object, path: str) -> EvidenceRef:
    obj = _strict_object(value, _EVIDENCE_FIELDS, path)
    start = _number(obj["start"], f"{path}.start")
    end = _number(obj["end"], f"{path}.end")
    if start < 0 or end <= start:
        raise SuiteValidationError(
            f"{path} must have non-negative start and end greater than start"
        )
    return EvidenceRef(
        ref_id=_string(obj["ref_id"], f"{path}.ref_id"),
        clean_file=_string(obj["clean_file"], f"{path}.clean_file"),
        video_id=_string(obj["video_id"], f"{path}.video_id"),
        start=start,
        end=end,
        excerpt=_string(obj["excerpt"], f"{path}.excerpt"),
        clean_sha256=_sha256(obj["clean_sha256"], f"{path}.clean_sha256"),
    )


def _parse_atom(value: object, path: str) -> ClaimAtom:
    obj = _strict_object(value, _ATOM_FIELDS, path)
    weight = _positive_integer(obj["weight"], f"{path}.weight")
    provenance_ids = _string_tuple(
        obj["provenance_ref_ids"], f"{path}.provenance_ref_ids"
    )
    _require_unique(provenance_ids, f"{path}.provenance_ref_ids")
    return ClaimAtom(
        id=_string(obj["id"], f"{path}.id"),
        required=_boolean(obj["required"], f"{path}.required"),
        weight=weight,
        statement=_string(obj["statement"], f"{path}.statement"),
        provenance_ref_ids=provenance_ids,
    )


def _strict_object(value: object, fields: set[str], path: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise SuiteValidationError(f"{path} must be an object")
    obj: dict[str, Any] = value
    missing = sorted(fields - obj.keys())
    extra = sorted(obj.keys() - fields)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected fields: {', '.join(extra)}")
        raise SuiteValidationError(f"{path} has invalid schema ({'; '.join(details)})")
    return obj


def _array(value: object, path: str, *, nonempty: bool = False) -> list[Any]:
    if type(value) is not list:
        raise SuiteValidationError(f"{path} must be an array")
    result: list[Any] = value
    if nonempty and not result:
        raise SuiteValidationError(f"{path} must not be empty")
    return result


def _string(value: object, path: str) -> str:
    if type(value) is not str or not value.strip():
        raise SuiteValidationError(f"{path} must be a non-empty string")
    return value


def _string_tuple(value: object, path: str, *, nonempty: bool = False) -> tuple[str, ...]:
    items = _array(value, path, nonempty=nonempty)
    return tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(items))


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise SuiteValidationError(f"{path} must be a boolean")
    return value


def _integer(value: object, path: str) -> int:
    if type(value) is not int:
        raise SuiteValidationError(f"{path} must be an integer")
    return value


def _positive_integer(value: object, path: str) -> int:
    integer = _integer(value, path)
    if integer <= 0:
        raise SuiteValidationError(f"{path} must be positive")
    return integer


def _number(value: object, path: str) -> float:
    if type(value) not in (int, float):
        raise SuiteValidationError(f"{path} must be a number")
    if not math.isfinite(value):
        raise SuiteValidationError(f"{path} must be finite")
    return value


def _sha256(value: object, path: str) -> str:
    digest = _string(value, path)
    if _SHA256_RE.fullmatch(digest) is None:
        raise SuiteValidationError(f"{path} must be a lowercase SHA-256 hex digest")
    return digest


def _require_unique(values: Any, path: str) -> None:
    seen: set[Any] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and str(value) not in duplicates:
            duplicates.append(str(value))
        seen.add(value)
    if duplicates:
        raise SuiteValidationError(
            f"{path} contains duplicate IDs/values: {', '.join(duplicates)}"
        )


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SuiteValidationError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise SuiteValidationError(f"JSON number {value} is not finite")


def _validate_unicode_tree(value: object, path: str) -> None:
    if isinstance(value, str):
        _validate_unicode_string(value, path)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_unicode_string(key, f"{path}.<key>")
            _validate_unicode_tree(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_unicode_tree(item, f"{path}[{index}]")


def _validate_unicode_string(value: str, path: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise SuiteValidationError(f"{path} contains invalid Unicode: {exc}") from exc
    if unicodedata.normalize("NFC", value) != value:
        raise SuiteValidationError(f"{path} is not NFC-normalized Unicode text")
    if "\ufffd" in value or any(marker in value for marker in _MOJIBAKE_MARKERS):
        raise SuiteValidationError(f"{path} contains corrupt Unicode/mojibake text")


def _case_packet_value(case: EvalCase) -> dict[str, object]:
    value = asdict(case)
    value.pop("packet_sha256")
    return value


def _load_canonical_cards(root: Path, errors: list[str]) -> dict[str, dict[str, object]]:
    path = root / "data" / "cards" / "speaker_cards.json"
    try:
        raw = _read_repository_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError, SuiteValidationError) as exc:
        errors.append(f"canonical cards file {path} could not be read: {exc}")
        return {}
    if type(raw) is not list:
        errors.append(f"canonical cards file {path} must contain a JSON array")
        return {}

    result: dict[str, dict[str, object]] = {}
    for index, item in enumerate(raw):
        if type(item) is not dict:
            errors.append(f"canonical card at index {index} is not an object")
            continue
        try:
            subset = {
                "id": item["id"],
                "q": item["q"],
                "take": item["take"],
                "support": item["support"],
                "sources": item["sources"],
            }
            parsed = _parse_card(subset, f"canonical_cards[{index}]")
        except (KeyError, SuiteValidationError) as exc:
            errors.append(f"canonical card at index {index} is malformed: {exc}")
            continue
        if parsed.id in result:
            errors.append(f"canonical cards contain duplicate ID {parsed.id!r}")
            continue
        result[parsed.id] = asdict(parsed)
    return result


def _read_repository_json(path: Path) -> object:
    text = path.read_bytes().decode("utf-8")
    raw = json.loads(
        text,
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=_reject_json_constant,
    )
    _validate_unicode_tree(raw, str(path))
    return raw


def _validate_evidence(
    evidence: EvidenceRef,
    case_prefix: str,
    root: Path,
    clean_cache: dict[Path, tuple[str | None, Any | None, str | None]],
    errors: list[str],
) -> None:
    prefix = f"{case_prefix}, evidence {evidence.ref_id}"
    path = _safe_repository_path(root, evidence.clean_file)
    if path is None:
        errors.append(f"{prefix}: clean_file escapes repository root")
        return

    if path not in clean_cache:
        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            clean_cache[path] = (None, None, str(exc))
        else:
            digest = hashlib.sha256(raw_bytes).hexdigest()
            try:
                clean_data = json.loads(
                    raw_bytes.decode("utf-8"),
                    object_pairs_hook=_object_without_duplicate_keys,
                    parse_constant=_reject_json_constant,
                )
                _validate_unicode_tree(clean_data, str(path))
            except (UnicodeError, json.JSONDecodeError, SuiteValidationError) as exc:
                clean_cache[path] = (digest, None, str(exc))
            else:
                clean_cache[path] = (digest, clean_data, None)

    actual_digest, clean_data, read_error = clean_cache[path]
    if actual_digest is None:
        errors.append(f"{prefix}: clean file does not exist or is unreadable: {path} ({read_error})")
        return
    if actual_digest != evidence.clean_sha256:
        errors.append(
            f"{prefix}: clean_sha256 mismatch "
            f"(declared {evidence.clean_sha256}, actual {actual_digest})"
        )
    if read_error is not None or type(clean_data) is not dict:
        errors.append(f"{prefix}: clean file is malformed: {read_error or 'root is not an object'}")
        return

    clean_video_id = clean_data.get("video_id")
    if clean_video_id != evidence.video_id:
        errors.append(
            f"{prefix}: video_id mismatch "
            f"(declared {evidence.video_id!r}, clean file has {clean_video_id!r})"
        )

    segments = clean_data.get("segments")
    if type(segments) is not list:
        errors.append(f"{prefix}: clean file has no valid segments array")
        return

    window_text: list[str] = []
    malformed_segment = False
    for segment in segments:
        if type(segment) is not dict:
            malformed_segment = True
            continue
        start = segment.get("start")
        end = segment.get("end")
        text = segment.get("text")
        if (
            type(start) not in (int, float)
            or type(end) not in (int, float)
            or not math.isfinite(start)
            or not math.isfinite(end)
            or type(text) is not str
        ):
            malformed_segment = True
            continue
        # Clean transcripts only timestamp whole segments.  A partially
        # overlapping segment cannot prove where inside that segment an
        # excerpt was spoken, so only wholly declared segments are admissible.
        if start >= evidence.start and end <= evidence.end:
            window_text.append(text)
    if malformed_segment:
        errors.append(f"{prefix}: clean file contains malformed segment records")

    excerpt = _normalized_match_text(evidence.excerpt)
    available = _normalized_match_text(" ".join(window_text))
    if not window_text or excerpt not in available:
        errors.append(
            f"{prefix}: excerpt not found in declared segment window "
            f"{evidence.start}-{evidence.end}"
        )


def _safe_repository_path(root: Path, relative: str) -> Path | None:
    candidate_input = Path(relative)
    if candidate_input.is_absolute():
        return None
    candidate = (root / candidate_input).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _normalized_match_text(value: str) -> str:
    normalized = _normalize_newlines_and_nfc(value)
    assert isinstance(normalized, str)
    return " ".join(normalized.split())


__all__ = [
    "CardSnapshot",
    "CardSource",
    "ClaimAtom",
    "DiagnosticGeneration",
    "EvalCase",
    "EvalSuite",
    "EvidenceRef",
    "ModelPolicy",
    "ProductionGeneration",
    "SuiteValidationError",
    "ValidationReport",
    "canonical_sha256",
    "load_suite",
    "validate_suite",
]
