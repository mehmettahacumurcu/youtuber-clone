"""Shared test fixtures: a Settings object backed by an isolated temp data dir."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pipeline.config import load_settings


# These tests intentionally validate the unpublished corpus against frozen hashes.
# Preserve their strict checks when any local corpus is present; only the public
# source snapshot without either corpus directory skips them.
_PRIVATE_CORPUS_TESTS = {
    "test_report_locks_semantics_then_allows_style_without_changing_lock",
    "test_missing_review_is_incomplete_but_invalid_review_is_blocked",
    "test_locked_missing_review_stays_incomplete_and_turkish_round_trips",
    "test_critical_failure_is_failed_not_blocked_and_keeps_exact_reasons",
    "test_report_contract_is_lexicographic_and_path_stable",
    "test_production_gate_accepts_exactly_42_of_45_with_two_per_question",
    "test_production_gate_rejects_question_below_two_of_three",
    "test_offline_validate_never_constructs_ollama_client",
    "test_worker_refreshes_gates_then_makes_15_frozen_ordered_calls_and_is_deterministic",
    "test_retrieval_defects_preserve_existing_snapshot",
    "test_stale_prerequisite_blocks_before_retrieval_and_preserves_snapshot",
    "test_load_snapshot_rejects_tampered_order",
    "test_load_snapshot_refreshes_fixed_gates_before_accepting_identity",
    "test_positive_record_requires_status_verdicts_groups_and_exact_reconstruction",
    "test_every_source_hash_window_and_excerpt_matches_repository",
    "test_v1_fixture_has_corrected_bindings_and_complete_provenance",
}


def pytest_collection_modifyitems(items):
    root = Path(__file__).resolve().parents[1]
    if (root / "data/clean").exists() or (root / "data/cards").exists():
        return
    for item in items:
        if item.nodeid.startswith("tests/eval/") and item.originalname in _PRIVATE_CORPUS_TESTS:
            item.add_marker(pytest.mark.skip(
                reason="private clean/card corpus omitted from source publication"
            ))


def pytest_addoption(parser):
    parser.addoption(
        "--voice-root",
        type=Path,
        default=None,
        help="real RVC artifact root used only by tests marked slow",
    )


@pytest.fixture
def voice_root(request) -> Path | None:
    return request.config.getoption("--voice-root")


@pytest.fixture
def settings(tmp_path):
    """Return a Settings whose paths point inside a fresh tmp_path/data tree."""
    data = tmp_path / "data"
    for sub in ("audio", "meta", "vad", "diarize", "identify", "transcribe", "filter", "clean", "dataset", "review", "reference"):
        (data / sub).mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        textwrap.dedent(
            f"""
            paths:
              data_dir: {data.as_posix()}
              audio_dir: {(data / "audio").as_posix()}
              meta_dir: {(data / "meta").as_posix()}
              vad_dir: {(data / "vad").as_posix()}
              diarize_dir: {(data / "diarize").as_posix()}
              identify_dir: {(data / "identify").as_posix()}
              transcribe_dir: {(data / "transcribe").as_posix()}
              filter_dir: {(data / "filter").as_posix()}
              clean_dir: {(data / "clean").as_posix()}
              dataset_dir: {(data / "dataset").as_posix()}
              reference_clip: {(data / "reference" / "speaker_reference.wav").as_posix()}
              review_dir: {(data / "review").as_posix()}
              decisions_db: {(data / "decisions.sqlite").as_posix()}
              logs_dir: {(tmp_path / "logs").as_posix()}
            """
        ),
        encoding="utf-8",
    )
    return load_settings(cfg)
