import json
from pathlib import Path

import pytest

from eval.rag_reliability_schema import (
    ReliabilityValidationError,
    load_reliability_suite,
    validate_repository_sources,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "eval/fixtures/rag_reliability_v1.json"


def test_frozen_suite_shape_and_categories_are_exact():
    suite = load_reliability_suite(FIXTURE)
    assert suite.schema_version == 1
    assert len(suite.cases) == 55
    assert [case.id for case in suite.cases[:3]] == ["O01", "O02", "O03"]
    assert [case.id for case in suite.cases[15:18]] == ["P01", "P02", "P03"]
    assert [case.id for case in suite.cases[30:33]] == ["OD01", "OD02", "OD03"]
    assert [case.id for case in suite.cases[45:48]] == ["NT01", "NT02", "NT03"]
    assert len(suite.repeat_case_ids) == 6
    assert set(suite.repeat_case_ids) == {"NT01", "P03", "O08", "P13", "O14", "P01"}
    assert [case.expected_status for case in suite.smoke_cases] == [
        "answerable", "partial", "answerable", "unsupported",
    ]


def test_positive_pairs_share_expected_contract_and_negatives_are_unsupported():
    suite = load_reliability_suite(FIXTURE)
    by_id = {case.id: case for case in suite.cases}
    for index in range(1, 16):
        original = by_id[f"O{index:02d}"]
        paraphrase = by_id[f"P{index:02d}"]
        assert original.pair_id == paraphrase.pair_id
        assert original.expected_status == paraphrase.expected_status
        assert original.required_central_verdicts == paraphrase.required_central_verdicts
        assert original.required_evidence_group_ids == paraphrase.required_evidence_group_ids
        assert original.retrieval_mode == "clean"
        assert paraphrase.retrieval_mode == "clean_with_card_hints"
    for case in suite.cases[30:]:
        assert case.expected_status == "unsupported"
        assert case.required_evidence_group_ids == ()


def test_every_source_hash_window_and_excerpt_matches_repository():
    suite = load_reliability_suite(FIXTURE)
    assert validate_repository_sources(suite, ROOT) == ()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda data: data["cases"][0].update(question="bad\ufffdtext"), "clean NFC"),
        (lambda data: data["cases"][1].update(id="O01"), "case IDs"),
        (lambda data: data["cases"][30].update(expected_status="answerable"), "negative"),
        (lambda data: data.update(repeat_case_ids=["NT01"]), "six repeat"),
    ],
)
def test_mutated_fixture_is_rejected(tmp_path, mutation, match):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    mutation(data)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ReliabilityValidationError, match=match):
        load_reliability_suite(path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda data: (
                data["evidence_groups"].pop(0),
                data["cases"][0].update(required_evidence_group_ids=data["cases"][0]["required_evidence_group_ids"][1:]),
                data["cases"][15].update(required_evidence_group_ids=data["cases"][15]["required_evidence_group_ids"][1:]),
            ),
            "evidence group IDs",
        ),
        (lambda data: data["cases"][30].update(category="fabricated_near_topic"), "off-domain"),
        (lambda data: data["cases"][30].update(retrieval_mode="clean"), "negative categories"),
        (lambda data: data["cases"][45].update(category="off_domain"), "fabricated-near-topic"),
        (lambda data: data["smoke_cases"][0].update(id="SMOKE-ALTERED"), "smoke IDs"),
        (lambda data: data["smoke_cases"][0].update(category="original"), "smoke categories"),
    ],
)
def test_frozen_evidence_and_negative_metadata_cannot_be_weakened(tmp_path, mutation, match):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    mutation(data)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ReliabilityValidationError, match=match):
        load_reliability_suite(path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda data: (
                data["cases"][1].update(required_central_verdicts=["entailed"]),
                data["cases"][16].update(required_central_verdicts=["entailed"]),
            ),
            "positive pair verdicts",
        ),
        (
            lambda data: data["smoke_cases"][1].update(required_central_verdicts=["not_found"]),
            "smoke semantic",
        ),
        (
            lambda data: data["smoke_cases"][1].update(required_evidence_group_ids=[]),
            "smoke semantic",
        ),
    ],
)
def test_frozen_pair_and_smoke_semantic_contracts_cannot_be_weakened(tmp_path, mutation, match):
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    mutation(data)
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ReliabilityValidationError, match=match):
        load_reliability_suite(path)
