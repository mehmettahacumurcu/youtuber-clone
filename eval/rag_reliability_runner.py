"""Resumable live execution of the frozen production RAG reliability suite."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path

import requests

from eval.rag_reliability_schema import ReliabilitySuite
from pipeline.config import Settings
from rag.card_manifest import validate_card_hint_manifest
from rag.evidence_pipeline import EvidenceDecision, assess_question
from rag.verifier import EvidenceVerifier


class RunIdentityMismatch(RuntimeError):
    """Resume inputs differ from the original run."""


def run_suite(
    *,
    suite: ReliabilitySuite,
    suite_path: Path,
    settings: Settings,
    run_dir: Path,
    assess_fn: Callable[..., EvidenceDecision] = assess_question,
    verifier_factory: Callable[..., object] = EvidenceVerifier.from_config,
    runtime_identity_fn: Callable[..., Mapping[str, object]] | None = None,
    resume: bool,
) -> None:
    """Run each frozen case once, persisting atomically for safe resume."""
    run_dir = Path(run_dir)
    identity = _build_identity(
        suite_path,
        settings,
        runtime_identity_fn or runtime_identity,
    )
    identity_path = run_dir / "run.json"
    if resume:
        try:
            existing = json.loads(identity_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            existing = None
        if type(existing) is not dict:
            if _has_result_records(run_dir):
                raise RunIdentityMismatch(
                    "cannot resume existing records without a valid run identity"
                )
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_atomic(identity_path, identity)
        elif existing != identity:
            raise RunIdentityMismatch("fixture, config, manifest, or verifier identity changed")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_atomic(identity_path, identity)

    if any(case.retrieval_mode == "clean_with_card_hints" for case in suite.cases):
        status = validate_card_hint_manifest(
            catalog_path=settings.rag.card_catalog_path,
            manifest_path=settings.rag.card_manifest_path,
            collection=settings.rag.card_collection,
            embedder=settings.rag.embedder,
            store_path=settings.rag.store_path,
        )
        if not status.usable and runtime_identity_fn is None:
            raise RuntimeError(
                f"live hint acceptance requires a valid card manifest: {status.reason}"
            )

    verifier = verifier_factory(settings.rag)
    for case in suite.cases:
        _run_one(
            case,
            run_dir / "cases" / f"{case.id}.json",
            settings,
            verifier,
            assess_fn,
            resume,
        )
    by_id = {case.id: case for case in suite.cases}
    for case_id in suite.repeat_case_ids:
        _run_one(
            by_id[case_id],
            run_dir / "repeats" / f"{case_id}.json",
            settings,
            verifier,
            assess_fn,
            resume,
        )


def runtime_identity(config) -> Mapping[str, object]:
    """Pin the live verifier to its local Ollama version and model digest."""
    version = requests.get("http://127.0.0.1:11434/api/version", timeout=10)
    version.raise_for_status()
    tags = requests.get("http://127.0.0.1:11434/api/tags", timeout=30)
    tags.raise_for_status()
    models = tags.json().get("models", [])
    matching = [
        model for model in models
        if model.get("name") == config.verifier_model
        or model.get("model") == config.verifier_model
        or str(model.get("name", "")).startswith(config.verifier_model + ":")
    ]
    if len(matching) != 1 or not matching[0].get("digest"):
        raise RuntimeError(f"cannot resolve one installed digest for {config.verifier_model}")
    return {
        "ollama_version": version.json().get("version"),
        "model_digest": matching[0]["digest"],
    }


def decision_to_record(decision: EvidenceDecision) -> dict[str, object]:
    """Serialize every decision field required for deterministic offline gating."""
    return {
        "status": decision.status,
        "claims": [claim.model_dump(mode="json") for claim in decision.claims],
        "selected_spans": [
            {
                "id": span.id, "video_id": span.video_id, "title": span.title,
                "start": span.start, "end": span.end, "text": span.text,
                "retrieval_score": span.retrieval_score,
                "source_chunk_ids": list(span.source_chunk_ids),
                "query_origins": list(span.query_origins),
            }
            for span in decision.selected_spans
        ],
        "hint_hits": [
            {
                "card_id": hint.card_id, "question": hint.question, "take": hint.take,
                "retrieval_score": hint.retrieval_score,
            }
            for hint in decision.hint_hits
        ],
        "answer_question": decision.answer_question,
        "unsupported_claims": list(decision.unsupported_claims),
        "failure_reason": decision.failure_reason,
        "diagnostics": dict(decision.diagnostics),
    }


def _run_one(case, path, settings, verifier, assess_fn, resume) -> None:
    if resume and path.is_file():
        return
    started = time.monotonic()
    decision = assess_fn(
        case.question,
        settings,
        verifier=verifier,
        retrieval_mode=case.retrieval_mode,
    )
    record = {
        "case_id": case.id,
        "question": case.question,
        "retrieval_mode": case.retrieval_mode,
        "elapsed_s": time.monotonic() - started,
        "decision": decision_to_record(decision),
    }
    _write_atomic(path, record)
    print(f"{case.id}: {decision.status} ({record['elapsed_s']:.1f}s)", flush=True)


def _build_identity(suite_path, settings, identity_fn) -> dict[str, object]:
    config = settings.rag
    projection = {
        field: str(getattr(config, field)) if field.endswith("_path") or field == "store_path"
        else getattr(config, field)
        for field in (
            "store_path", "evidence_collection", "retrieval_mode", "card_collection",
            "card_catalog_path", "card_manifest_path", "card_hint_top_n", "card_hint_min_score",
            "original_candidate_n", "hint_candidate_n", "verifier_candidate_max",
            "verifier_span_char_cap", "verifier_span_seconds_cap", "answer_span_max",
            "answer_span_char_cap", "answer_context_char_cap", "verifier_model",
            "verifier_temperature", "verifier_seed", "verifier_num_ctx",
            "verifier_num_predict", "verifier_timeout_s",
        )
    }
    manifest_sha = (
        hashlib.sha256(config.card_manifest_path.read_bytes()).hexdigest()
        if config.card_manifest_path.is_file() else None
    )
    return {
        "schema_version": 1,
        "fixture_sha256": hashlib.sha256(Path(suite_path).read_bytes()).hexdigest(),
        "config": projection,
        "card_manifest_sha256": manifest_sha,
        "runtime": dict(identity_fn(config)),
    }


def _write_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _has_result_records(run_dir: Path) -> bool:
    return any(
        path.is_file()
        for directory in (run_dir / "cases", run_dir / "repeats")
        for path in directory.glob("*.json")
    )
