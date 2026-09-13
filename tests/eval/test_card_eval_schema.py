from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from eval.card_eval_schema import (
    SuiteValidationError,
    canonical_sha256,
    load_suite,
    validate_suite,
)


CARD = {
    "id": "card::0001",
    "q": "Deneme sorusu?",
    "take": "Kanıta bağlı bir yanıt.",
    "support": 1.25,
    "sources": [
        {
            "video_id": "vid-1",
            "title": "Başlık",
            "start": 10.0,
            "score": 0.75,
        }
    ],
}

CLEAN = {
    "video_id": "vid-1",
    "segments": [
        {"start": 10.0, "end": 12.0, "text": "Kanıt cümlesi burada."},
        {"start": 20.0, "end": 22.0, "text": "Pencerenin dışındaki alıntı."},
    ],
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def _seal_case(case: dict[str, object]) -> None:
    case.pop("packet_sha256", None)
    case["packet_sha256"] = canonical_sha256(case)


def _minimal_suite(clean_sha256: str) -> dict[str, object]:
    case: dict[str, object] = {
        "id": "G01",
        "kind": "grounded",
        "question": "Bu konuda ne düşünüyorsun?",
        "card_ids": ["card::0001"],
        "cards": [copy.deepcopy(CARD)],
        "evidence": [
            {
                "ref_id": "ref-1",
                "clean_file": "data/clean/vid-1.json",
                "video_id": "vid-1",
                "start": 10.0,
                "end": 12.0,
                "excerpt": "Kanıt cümlesi burada.",
                "clean_sha256": clean_sha256,
            }
        ],
        "atoms": [
            {
                "id": "atom-1",
                "required": True,
                "weight": 2,
                "statement": "Kanıt cümlesi aktarılmalıdır.",
                "provenance_ref_ids": ["ref-1"],
            }
        ],
        "forbidden_claims": ["Kanıtsız kesin hüküm"],
        "allowed_entities": ["Örnek Kişi"],
        "allowed_numbers_dates": ["2026"],
        "retrieval_relevant_ids": ["card::0001"],
    }
    _seal_case(case)
    return {
        "schema_version": 1,
        "suite_id": "test-suite-v1",
        "model_policy": {
            "required_model": "qwen3:14b",
            "forbid_abliterated": True,
        },
        "diagnostic_generation": {
            "think": False,
            "temperature": 0,
            "seed": 20260711,
            "top_p": 1.0,
            "top_k": 1,
            "repeat_penalty": 1.05,
            "num_ctx": 4096,
            "num_predict": 384,
            "stream": False,
            "keep_alive": 0,
        },
        "production_generation": {
            "think": False,
            "temperature": 0.7,
            "seeds": [20260711, 20260712, 20260713],
            "top_p": 0.85,
            "repeat_penalty": 1.2,
            "num_ctx": 4096,
            "num_predict": 640,
            "stream": False,
            "keep_alive": 0,
        },
        "cases": [case],
    }


@pytest.fixture
def repo_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    clean_path = tmp_path / "data" / "clean" / "vid-1.json"
    _write_json(clean_path, CLEAN)
    clean_sha256 = hashlib.sha256(clean_path.read_bytes()).hexdigest()

    canonical_card = copy.deepcopy(CARD)
    canonical_card.update(
        {"n_takes": 1, "alt_qs": [], "n_strong": 1, "date": "20260711"}
    )
    _write_json(tmp_path / "data" / "cards" / "speaker_cards.json", [canonical_card])
    return tmp_path, _minimal_suite(clean_sha256)


def _load_mapping(tmp_path: Path, mapping: dict[str, object]):
    path = tmp_path / "suite.json"
    _write_json(path, mapping)
    return load_suite(path)


def _first_case(mapping: dict[str, object]) -> dict[str, object]:
    return mapping["cases"][0]  # type: ignore[index,return-value]


def test_canonical_hash_ignores_key_order_and_windows_newlines():
    assert canonical_sha256({"b": "x\r\ny", "a": 1}) == canonical_sha256(
        {"a": 1, "b": "x\ny"}
    )


def test_canonical_hash_normalizes_nfc_and_tuples():
    assert canonical_sha256({"text": "s\u0327imdi", "ids": ("a", "b")}) == (
        canonical_sha256({"text": "şimdi", "ids": ["a", "b"]})
    )


def test_loads_frozen_suite_with_stable_sha256(repo_fixture, tmp_path: Path):
    repo_root, mapping = repo_fixture
    suite = _load_mapping(tmp_path, mapping)

    assert suite.suite_id == "test-suite-v1"
    assert suite.cases[0].cards[0].id == "card::0001"
    assert isinstance(suite.cases, tuple)
    assert suite.sha256 == canonical_sha256(mapping)
    assert validate_suite(suite, repo_root).ok
    with pytest.raises(FrozenInstanceError):
        suite.suite_id = "changed"  # type: ignore[misc]

    reordered_path = tmp_path / "reordered.json"
    reordered_path.write_text(
        json.dumps(mapping, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    assert load_suite(reordered_path).sha256 == suite.sha256


def test_eval_case_has_stable_sha256(repo_fixture, tmp_path: Path):
    _, mapping = repo_fixture
    suite = _load_mapping(tmp_path, mapping)

    assert suite.cases[0].sha256 == canonical_sha256(_first_case(mapping))


@pytest.mark.parametrize(
    "bad", ["bozuk \ufffd", "gÃ¼ya", "ÅŸimdi", "deÄŸil", "Ä°stanbul", "doÄŸru"]
)
def test_rejects_corrupt_unicode(repo_fixture, tmp_path: Path, bad: str):
    _, mapping = repo_fixture
    _first_case(mapping)["question"] = bad
    _seal_case(_first_case(mapping))
    path = tmp_path / "suite.json"
    _write_json(path, mapping)

    with pytest.raises(SuiteValidationError, match="Unicode"):
        load_suite(path)


def test_rejects_non_nfc_text(repo_fixture, tmp_path: Path):
    _, mapping = repo_fixture
    _first_case(mapping)["question"] = "s\u0327imdi"
    _seal_case(_first_case(mapping))

    with pytest.raises(SuiteValidationError, match="NFC"):
        _load_mapping(tmp_path, mapping)


def test_rejects_invalid_utf8_bytes(tmp_path: Path):
    path = tmp_path / "suite.json"
    path.write_bytes(b'{"suite_id":"\xff"}')

    with pytest.raises(SuiteValidationError, match="UTF-8"):
        load_suite(path)


def test_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "suite.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(SuiteValidationError, match="JSON"):
        load_suite(path)


@pytest.mark.parametrize(
    "model_name",
    [
        "qwen3:8b",
        "huihui_ai/qwen3-abliterated:14b-v2",
        "speaker-v5-a636",
        "qwen3:14b-lora",
    ],
    ids=["alternate-base", "abliterated-qwen", "a636", "lora"],
)
def test_rejects_non_stock_required_model(
    repo_fixture, tmp_path: Path, model_name: str
):
    _, mapping = repo_fixture
    mapping["model_policy"]["required_model"] = model_name  # type: ignore[index]

    with pytest.raises(SuiteValidationError, match=r"required_model.*qwen3:14b"):
        _load_mapping(tmp_path, mapping)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda suite: suite.update({"unexpected": True}),
        lambda suite: _first_case(suite).update({"unexpected": True}),
        lambda suite: _first_case(suite).pop("question"),
        lambda suite: _first_case(suite).update({"kind": "other"}),
        lambda suite: _first_case(suite).update({"question": 123}),
    ],
    ids=["unknown-top", "unknown-case", "missing-case", "bad-literal", "bad-type"],
)
def test_rejects_malformed_schema(repo_fixture, tmp_path: Path, mutation):
    _, mapping = repo_fixture
    mutation(mapping)
    if "packet_sha256" in _first_case(mapping):
        _seal_case(_first_case(mapping))

    with pytest.raises(SuiteValidationError):
        _load_mapping(tmp_path, mapping)


@pytest.mark.parametrize(
    "field_path,value",
    [
        (("cases", 0, "evidence", 0, "start"), float("nan")),
        (("cases", 0, "cards", 0, "support"), float("inf")),
        (("cases", 0, "cards", 0, "sources", 0, "score"), float("-inf")),
        (("diagnostic_generation", "temperature"), float("nan")),
    ],
    ids=["timestamp", "support", "source-score", "generation-score"],
)
def test_rejects_non_finite_numbers(
    repo_fixture, tmp_path: Path, field_path: tuple[object, ...], value: float
):
    _, mapping = repo_fixture
    target: object = mapping
    for part in field_path[:-1]:
        target = target[part]  # type: ignore[index]
    target[field_path[-1]] = value  # type: ignore[index]

    with pytest.raises(SuiteValidationError, match="finite"):
        _load_mapping(tmp_path, mapping)


@pytest.mark.parametrize("collection", ["cases", "cards", "evidence", "atoms"])
def test_rejects_duplicate_ids(repo_fixture, tmp_path: Path, collection: str):
    _, mapping = repo_fixture
    if collection == "cases":
        mapping["cases"].append(copy.deepcopy(_first_case(mapping)))  # type: ignore[union-attr]
    else:
        case = _first_case(mapping)
        case[collection].append(copy.deepcopy(case[collection][0]))  # type: ignore[index,union-attr]
        _seal_case(case)

    with pytest.raises(SuiteValidationError, match="duplicate"):
        _load_mapping(tmp_path, mapping)


def test_missing_clean_file_is_a_provenance_error(repo_fixture, tmp_path: Path):
    repo_root, mapping = repo_fixture
    (repo_root / "data" / "clean" / "vid-1.json").unlink()
    report = validate_suite(_load_mapping(tmp_path, mapping), repo_root)

    assert not report.ok
    assert any("clean file does not exist" in error for error in report.errors)


def test_mismatched_clean_video_id_is_a_provenance_error(
    repo_fixture, tmp_path: Path
):
    repo_root, mapping = repo_fixture
    clean_path = repo_root / "data" / "clean" / "vid-1.json"
    clean = copy.deepcopy(CLEAN)
    clean["video_id"] = "different-video"
    _write_json(clean_path, clean)
    evidence = _first_case(mapping)["evidence"][0]  # type: ignore[index]
    evidence["clean_sha256"] = hashlib.sha256(clean_path.read_bytes()).hexdigest()
    _seal_case(_first_case(mapping))

    report = validate_suite(_load_mapping(tmp_path, mapping), repo_root)
    assert not report.ok
    assert any("video_id mismatch" in error for error in report.errors)


def test_clean_file_hash_mismatch_is_a_provenance_error(repo_fixture, tmp_path: Path):
    repo_root, mapping = repo_fixture
    evidence = _first_case(mapping)["evidence"][0]  # type: ignore[index]
    evidence["clean_sha256"] = "0" * 64
    _seal_case(_first_case(mapping))

    report = validate_suite(_load_mapping(tmp_path, mapping), repo_root)
    assert not report.ok
    assert any("clean_sha256 mismatch" in error for error in report.errors)


def test_excerpt_must_appear_inside_declared_segment_window(
    repo_fixture, tmp_path: Path
):
    repo_root, mapping = repo_fixture
    evidence = _first_case(mapping)["evidence"][0]  # type: ignore[index]
    evidence["excerpt"] = "Pencerenin dışındaki alıntı."
    _seal_case(_first_case(mapping))

    report = validate_suite(_load_mapping(tmp_path, mapping), repo_root)
    assert not report.ok
    assert any("excerpt not found" in error for error in report.errors)


def test_partial_segment_overlap_cannot_prove_excerpt(repo_fixture, tmp_path: Path):
    repo_root, mapping = repo_fixture
    evidence = _first_case(mapping)["evidence"][0]  # type: ignore[index]
    evidence.update({"start": 11.9, "end": 12.1})
    _seal_case(_first_case(mapping))

    report = validate_suite(_load_mapping(tmp_path, mapping), repo_root)
    assert not report.ok
    assert any("excerpt not found" in error for error in report.errors)


def test_required_atom_must_reference_known_evidence(repo_fixture, tmp_path: Path):
    repo_root, mapping = repo_fixture
    atom = _first_case(mapping)["atoms"][0]  # type: ignore[index]
    atom["provenance_ref_ids"] = ["missing-ref"]
    _seal_case(_first_case(mapping))

    report = validate_suite(_load_mapping(tmp_path, mapping), repo_root)
    assert not report.ok
    assert any("required atom" in error and "missing-ref" in error for error in report.errors)


def test_optional_positive_weight_atom_requires_provenance(
    repo_fixture, tmp_path: Path
):
    repo_root, mapping = repo_fixture
    atom = _first_case(mapping)["atoms"][0]  # type: ignore[index]
    atom.update({"required": False, "provenance_ref_ids": []})
    _seal_case(_first_case(mapping))

    report = validate_suite(_load_mapping(tmp_path, mapping), repo_root)
    assert not report.ok
    assert any(
        "atom 'atom-1' has no provenance_ref_ids" in error
        for error in report.errors
    )


def test_card_snapshot_must_match_canonical_cards(repo_fixture, tmp_path: Path):
    repo_root, mapping = repo_fixture
    cards_path = repo_root / "data" / "cards" / "speaker_cards.json"
    canonical_cards = json.loads(cards_path.read_text(encoding="utf-8"))
    canonical_cards[0]["take"] = "Canonical text changed."
    _write_json(cards_path, canonical_cards)

    report = validate_suite(_load_mapping(tmp_path, mapping), repo_root)
    assert not report.ok
    assert any("card snapshot mismatch" in error for error in report.errors)


def test_card_ids_must_match_snapshots_and_retrieval_ids(
    repo_fixture, tmp_path: Path
):
    repo_root, mapping = repo_fixture
    _first_case(mapping)["retrieval_relevant_ids"] = ["card::9999"]
    _seal_case(_first_case(mapping))

    report = validate_suite(_load_mapping(tmp_path, mapping), repo_root)
    assert not report.ok
    assert any("retrieval_relevant_ids" in error for error in report.errors)


def test_packet_hash_mismatch_is_reported(repo_fixture, tmp_path: Path):
    repo_root, mapping = repo_fixture
    _first_case(mapping)["question"] = "Hash hesaplandıktan sonra değiştirildi."

    report = validate_suite(_load_mapping(tmp_path, mapping), repo_root)
    assert not report.ok
    assert any("packet_sha256 mismatch" in error for error in report.errors)
