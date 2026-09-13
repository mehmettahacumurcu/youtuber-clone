"""Strict schema and repository provenance checks for the production RAG reliability gate."""
from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


_MOJIBAKE_MARKERS = ("Ãƒ", "Ã‚", "Ã…", "Ã„", "Ã¢â‚¬", "Ã°Å¸", "Ã¯Â»Â¿")

_EXPECTED_GROUP_IDS_BY_PAIR = {
    "G01": ("g01::g01-story", "g01::g01-rejection", "g01::g01-overreach"),
    "G02": ("g02::g02-public-deal", "g02::g02-merkel-defense", "g02::g02-opposition-frame"),
    "G03": ("g03::g03-constitution-method",),
    "A01": ("a01::a01-henze-report",),
    "A02": ("a02::a02-historical-context", "a02::a02-distortion"),
    "A03": ("a03::a03-demirtas-claim", "a03::a03-ocalan-response", "a03::a03-narrator-reading"),
    "C01": ("c01::c01-chronology",),
    "C02": ("c02::c02-detailed-history", "c02::c02-prototype"),
    "C03": ("c03::c03-oil-chronology",),
    "F01": ("f01::f01-loan",),
    "F02": ("f02::f02-duration",),
    "N01": ("n01::n01-nihat-genc", "n01::n01-nihat-dogan"),
    "N02": ("n02::n02-iraq-roadmap", "n02::n02-us-context", "n02::n02-kaan-reading"),
    "R01": ("r01::r01-originating-claim", "r01::r01-shootdown-theory", "r01::r01-gaddafi-theory", "r01::r01-unresolved"),
    "R02": ("r02::r02-frugality-opinion", "r02::r02-table-uncertainty"),
}
_EXPECTED_PAIR_IDS = tuple(_EXPECTED_GROUP_IDS_BY_PAIR)
_EXPECTED_EVIDENCE_GROUP_IDS = tuple(
    group_id
    for group_ids in _EXPECTED_GROUP_IDS_BY_PAIR.values()
    for group_id in group_ids
)
_EXPECTED_VERDICTS_BY_PAIR = {
    "G01": ("entailed",),
    "G02": ("contradicted", "entailed"),
    "G03": ("contradicted",),
    "A01": ("contradicted",),
    "A02": ("contradicted",),
    "A03": ("contradicted", "entailed"),
    "C01": ("contradicted",),
    "C02": ("contradicted", "entailed"),
    "C03": ("contradicted",),
    "F01": ("entailed",),
    "F02": ("entailed",),
    "N01": ("contradicted", "entailed"),
    "N02": ("entailed",),
    "R01": ("explicitly_unresolved",),
    "R02": ("explicitly_unresolved",),
}
_EXPECTED_SMOKE_IDS = (
    "SMOKE-ANSWERABLE",
    "SMOKE-PARTIAL",
    "SMOKE-UNRESOLVED",
    "SMOKE-UNSUPPORTED",
)
_EXPECTED_SMOKE_SEMANTICS = {
    "SMOKE-ANSWERABLE": (
        "F01", "clean", "answerable", ("entailed",), _EXPECTED_GROUP_IDS_BY_PAIR["F01"],
    ),
    "SMOKE-PARTIAL": (
        "F01", "clean_with_card_hints", "partial", ("entailed", "not_found"),
        _EXPECTED_GROUP_IDS_BY_PAIR["F01"],
    ),
    "SMOKE-UNRESOLVED": (
        "R01", "clean", "answerable", ("explicitly_unresolved",),
        _EXPECTED_GROUP_IDS_BY_PAIR["R01"],
    ),
    "SMOKE-UNSUPPORTED": (
        None, "clean_with_card_hints", "unsupported", ("not_found",), (),
    ),
}


class ReliabilityValidationError(ValueError):
    """The frozen reliability suite is malformed or internally inconsistent."""


class EvidenceWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ref_id: str
    clean_file: str
    clean_sha256: str
    video_id: str
    start: float
    end: float
    excerpt: str


class EvidenceGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    windows: tuple[EvidenceWindow, ...] = Field(min_length=1)


class ReliabilityCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    category: Literal["original", "paraphrase", "off_domain", "fabricated_near_topic", "smoke"]
    pair_id: str | None
    question: str
    retrieval_mode: Literal["clean", "clean_with_card_hints"]
    expected_status: Literal["answerable", "partial", "unsupported"]
    required_central_verdicts: tuple[
        Literal["entailed", "contradicted", "explicitly_unresolved", "not_found"], ...
    ]
    required_evidence_group_ids: tuple[str, ...]
    regression_tags: tuple[str, ...]


class ReliabilitySuite(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    suite_id: Literal["rag-reliability-v1"]
    evidence_groups: tuple[EvidenceGroup, ...]
    cases: tuple[ReliabilityCase, ...]
    repeat_case_ids: tuple[str, ...]
    smoke_cases: tuple[ReliabilityCase, ...]

    @model_validator(mode="after")
    def validate_contract(self):
        expected_ids = (
            *(f"O{index:02d}" for index in range(1, 16)),
            *(f"P{index:02d}" for index in range(1, 16)),
            *(f"OD{index:02d}" for index in range(1, 16)),
            *(f"NT{index:02d}" for index in range(1, 11)),
        )
        actual_ids = tuple(case.id for case in self.cases)
        if actual_ids != expected_ids:
            raise ValueError("case IDs and ordering must match the frozen 55-case contract")
        questions = [case.question for case in self.cases]
        if len(questions) != len(set(questions)):
            raise ValueError("case questions must be unique")
        group_ids = [group.id for group in self.evidence_groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("evidence group IDs must be unique")
        if tuple(group_ids) != _EXPECTED_EVIDENCE_GROUP_IDS:
            raise ValueError("evidence group IDs must match the frozen 30-group contract")
        known_groups = set(group_ids)
        by_id = {case.id: case for case in self.cases}
        for index in range(1, 16):
            original = by_id[f"O{index:02d}"]
            paraphrase = by_id[f"P{index:02d}"]
            pair_id = _EXPECTED_PAIR_IDS[index - 1]
            if original.category != "original" or paraphrase.category != "paraphrase":
                raise ValueError("positive pair categories are invalid")
            if original.retrieval_mode != "clean" or paraphrase.retrieval_mode != "clean_with_card_hints":
                raise ValueError("positive pair retrieval modes are invalid")
            if original.pair_id != pair_id or paraphrase.pair_id != pair_id:
                raise ValueError("positive pair IDs are invalid")
            if (
                original.expected_status != "answerable"
                or paraphrase.expected_status != "answerable"
            ):
                raise ValueError("positive pair statuses are invalid")
            expected_group_ids = _EXPECTED_GROUP_IDS_BY_PAIR[pair_id]
            if (
                original.required_evidence_group_ids != expected_group_ids
                or paraphrase.required_evidence_group_ids != expected_group_ids
            ):
                raise ValueError("positive pair evidence groups must match the frozen contract")
            expected_verdicts = _EXPECTED_VERDICTS_BY_PAIR[pair_id]
            if (
                original.required_central_verdicts != expected_verdicts
                or paraphrase.required_central_verdicts != expected_verdicts
            ):
                raise ValueError("positive pair verdicts must match the frozen contract")
            pair_contract = (
                original.pair_id,
                original.expected_status,
                original.required_central_verdicts,
                original.required_evidence_group_ids,
            )
            if pair_contract != (
                paraphrase.pair_id,
                paraphrase.expected_status,
                paraphrase.required_central_verdicts,
                paraphrase.required_evidence_group_ids,
            ):
                raise ValueError("positive pair contracts differ")
        for case in self.cases[30:45]:
            if case.category != "off_domain":
                raise ValueError("off-domain categories must match the frozen contract")
            if case.retrieval_mode != "clean_with_card_hints":
                raise ValueError("negative categories must use clean_with_card_hints retrieval")
            if case.expected_status != "unsupported" or case.required_evidence_group_ids:
                raise ValueError("negative cases must be unsupported with no evidence groups")
        for case in self.cases[45:]:
            if case.category != "fabricated_near_topic":
                raise ValueError("fabricated-near-topic categories must match the frozen contract")
            if case.retrieval_mode != "clean_with_card_hints":
                raise ValueError("negative categories must use clean_with_card_hints retrieval")
            if case.expected_status != "unsupported" or case.required_evidence_group_ids:
                raise ValueError("negative cases must be unsupported with no evidence groups")
        for case in (*self.cases, *self.smoke_cases):
            unknown = set(case.required_evidence_group_ids) - known_groups
            if unknown:
                raise ValueError(f"case {case.id} references unknown evidence groups {sorted(unknown)}")
        used_groups = {
            group_id
            for case in (*self.cases, *self.smoke_cases)
            for group_id in case.required_evidence_group_ids
        }
        if used_groups != known_groups:
            raise ValueError("evidence groups must not be orphaned")
        expected_repeats = {"NT01", "P03", "O08", "P13", "O14", "P01"}
        if len(self.repeat_case_ids) != 6 or set(self.repeat_case_ids) != expected_repeats:
            raise ValueError("six repeat regression cases must match the frozen set")
        if [case.expected_status for case in self.smoke_cases] != [
            "answerable", "partial", "answerable", "unsupported",
        ]:
            raise ValueError("smoke cases must cover answerable, partial, unresolved-answerable, unsupported")
        if tuple(case.id for case in self.smoke_cases) != _EXPECTED_SMOKE_IDS:
            raise ValueError("smoke IDs must match the frozen contract")
        if any(case.category != "smoke" for case in self.smoke_cases):
            raise ValueError("smoke categories must match the frozen contract")
        for case in self.smoke_cases:
            semantic_contract = (
                case.pair_id,
                case.retrieval_mode,
                case.expected_status,
                case.required_central_verdicts,
                case.required_evidence_group_ids,
            )
            if semantic_contract != _EXPECTED_SMOKE_SEMANTICS[case.id]:
                raise ValueError("smoke semantic contract must match the frozen contract")
        _assert_clean_text(self.model_dump(mode="json"))
        return self


def load_reliability_suite(path: Path) -> ReliabilitySuite:
    try:
        raw = Path(path).read_bytes()
        text = raw.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda token: (_raise(ValueError(f"non-finite JSON {token}"))),
        )
        _assert_clean_text(value)
        return ReliabilitySuite.model_validate(value)
    except (OSError, UnicodeError, ValueError, ValidationError) as exc:
        raise ReliabilityValidationError(str(exc)) from exc


def validate_repository_sources(suite: ReliabilitySuite, repo_root: Path) -> tuple[str, ...]:
    root = Path(repo_root).resolve()
    errors: list[str] = []
    seen_refs: set[str] = set()
    for group in suite.evidence_groups:
        for window in group.windows:
            if window.ref_id in seen_refs:
                errors.append(f"duplicate evidence ref {window.ref_id}")
            seen_refs.add(window.ref_id)
            path = (root / window.clean_file).resolve()
            try:
                path.relative_to(root)
            except ValueError:
                errors.append(f"unsafe clean path {window.clean_file}")
                continue
            if not path.is_file():
                errors.append(f"missing clean file {window.clean_file}")
                continue
            if hashlib.sha256(path.read_bytes()).hexdigest() != window.clean_sha256:
                errors.append(f"clean hash mismatch {window.ref_id}")
                continue
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                errors.append(f"invalid clean JSON {window.ref_id}: {exc}")
                continue
            if document.get("video_id") != window.video_id:
                errors.append(f"video_id mismatch {window.ref_id}")
                continue
            if not math.isfinite(window.start) or not math.isfinite(window.end) or not 0 <= window.start < window.end:
                errors.append(f"invalid window timing {window.ref_id}")
                continue
            reconstructed = unicodedata.normalize(
                "NFC",
                " ".join(
                    (segment.get("text") or "").strip()
                    for segment in document.get("segments", [])
                    if float(segment.get("start", math.inf)) >= window.start - 1e-6
                    and float(segment.get("end", -math.inf)) <= window.end + 1e-6
                    and (segment.get("text") or "").strip()
                ),
            )
            if reconstructed != window.excerpt:
                errors.append(f"excerpt mismatch {window.ref_id}")
    return tuple(errors)


def _assert_clean_text(value: object) -> None:
    if isinstance(value, str):
        if (
            not unicodedata.is_normalized("NFC", value)
            or "\ufffd" in value
            or any(marker in value for marker in _MOJIBAKE_MARKERS)
        ):
            raise ValueError("all text must be clean NFC without replacement characters")
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            raise ValueError("all text must be clean NFC without surrogates")
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_clean_text(key)
            _assert_clean_text(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_clean_text(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("numeric values must be finite")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _raise(error: Exception) -> None:
    raise error
