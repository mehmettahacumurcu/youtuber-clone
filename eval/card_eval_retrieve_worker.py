"""Freeze one strictly-identified live card-retrieval packet per eval case.

This module deliberately runs retrieval in a short-lived process.  It validates
both fixed semantic gates and the current card-index manifest before the first
call that can load BGE, the reranker, or local Qdrant.
"""
from __future__ import annotations

import argparse
import json
import math
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.card_eval import generate_report
from eval.card_eval_ollama import (
    ModelIdentity,
    OFFICIAL_MODEL_DIGEST,
    OFFICIAL_MODEL_METADATA,
)
from eval.card_eval_schema import (
    EvalSuite,
    canonical_sha256,
    load_suite,
    validate_suite,
)
from pipeline.config import load_settings
from rag.index_cards import DENSE_DIM, CardCatalog, load_card_catalog
from rag.retrieve import retrieve
from rag.store import VectorStore
from rag.types import Hit


REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_SCHEMA = 1
RETRIEVAL_TOP_K = 20
RETRIEVAL_TOP_N = 3
RETRIEVAL_MAX_PER_VIDEO = 0
RETRIEVAL_WINDOW_CHARS = None

_RUN_FIELDS = {
    "run_schema",
    "suite_path",
    "suite_id",
    "suite_sha256",
    "model_identity",
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
_MANIFEST_FIELDS = {
    "manifest_schema",
    "catalog_sha256",
    "card_count",
    "collection",
    "embedder",
    "dense_dim",
    "store_path",
    "index_fingerprint",
    "built_at",
}
_IMMUTABLE_MANIFEST_FIELDS = _MANIFEST_FIELDS - {
    "index_fingerprint",
    "built_at",
}
_GATE_FIELDS = {
    "gate_schema",
    "stage",
    "suite_sha256",
    "model_identity_sha256",
    "input_fingerprint",
    "gate",
}
_GATE_RESULT_FIELDS = {
    "gate_id",
    "status",
    "expected_count",
    "result_count",
    "reviewed_count",
    "full_pass_count",
    "missing_result_keys",
    "missing_review_keys",
    "failed_answer_keys",
    "critical_failure_counts",
    "styled_regressions",
    "per_question",
    "semantic_passed",
    "style_status",
    "style_mean",
    "reasons",
    "style_reasons",
}
_QUESTION_GATE_FIELDS = {
    "question_id",
    "expected_count",
    "reviewed_count",
    "full_pass_count",
    "critical_failures",
    "style_scores",
    "style_median",
}
_SNAPSHOT_FIELDS = {
    "snapshot_schema",
    "suite_id",
    "suite_sha256",
    "model_identity_sha256",
    "fixed_gates",
    "index",
    "retrieval_settings",
    "packets",
    "micro_recall_at_3",
}
_FIXED_GATE_ID_FIELDS = {
    "diagnostic_input_fingerprint",
    "production_input_fingerprint",
}
_RETRIEVAL_SETTING_FIELDS = {
    "top_k",
    "top_n",
    "max_per_video",
    "window_chars",
    "collection",
    "embedder",
    "reranker",
    "store_path",
}
_PACKET_FIELDS = {
    "question_id",
    "question",
    "expected_ids",
    "returned_ids",
    "returned_expected_ids",
    "missing_expected_ids",
    "recall_at_3",
    "hits",
}
_HIT_FIELDS = {
    "rank",
    "card_id",
    "take",
    "dense_score",
    "sparse_score",
    "rerank_score",
}
_MICRO_FIELDS = {"expected_count", "returned_count", "missing_count", "recall"}
_MOJIBAKE_MARKERS = (
    "\u00c3",
    "\u00c2",
    "\u00c5",
    "\u00c4",
    "\u00e2\u20ac",
    "\u00f0\u0178",
    "\u00ef\u00bb\u00bf",
)


@dataclass(frozen=True)
class RetrievalSettings:
    cards_path: Path
    index_manifest_path: Path
    store_path: Path
    collection: str
    embedder: str
    reranker: str

    def __post_init__(self) -> None:
        for name in ("cards_path", "index_manifest_path", "store_path"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                raise ValueError(f"{name} must be a Path")
        for name in ("collection", "embedder", "reranker"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be non-empty text")


@dataclass(frozen=True)
class LivePrerequisites:
    run_dir: Path
    model_identity: ModelIdentity
    model_identity_sha256: str
    diagnostic_input_fingerprint: str
    production_input_fingerprint: str
    index_manifest: Mapping[str, object]
    catalog: CardCatalog


@dataclass(frozen=True)
class RetrievalHit:
    rank: int
    card_id: str
    take: str
    dense_score: float
    sparse_score: float
    rerank_score: float

    def to_json(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "card_id": self.card_id,
            "take": self.take,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "rerank_score": self.rerank_score,
        }


@dataclass(frozen=True)
class RetrievalPacket:
    question_id: str
    question: str
    expected_ids: tuple[str, ...]
    returned_ids: tuple[str, ...]
    returned_expected_ids: tuple[str, ...]
    missing_expected_ids: tuple[str, ...]
    recall_at_3: float
    hits: tuple[RetrievalHit, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "expected_ids": list(self.expected_ids),
            "returned_ids": list(self.returned_ids),
            "returned_expected_ids": list(self.returned_expected_ids),
            "missing_expected_ids": list(self.missing_expected_ids),
            "recall_at_3": self.recall_at_3,
            "hits": [hit.to_json() for hit in self.hits],
        }


@dataclass(frozen=True)
class RetrievalSnapshot:
    suite_id: str
    suite_sha256: str
    model_identity_sha256: str
    fixed_gates: Mapping[str, str]
    index: Mapping[str, object]
    retrieval_settings: Mapping[str, object]
    packets: tuple[RetrievalPacket, ...]
    micro_expected_count: int
    micro_returned_count: int

    def to_json(self) -> dict[str, object]:
        missing = self.micro_expected_count - self.micro_returned_count
        recall = (
            self.micro_returned_count / self.micro_expected_count
            if self.micro_expected_count
            else 1.0
        )
        return {
            "snapshot_schema": SNAPSHOT_SCHEMA,
            "suite_id": self.suite_id,
            "suite_sha256": self.suite_sha256,
            "model_identity_sha256": self.model_identity_sha256,
            "fixed_gates": dict(self.fixed_gates),
            "index": dict(self.index),
            "retrieval_settings": dict(self.retrieval_settings),
            "packets": [packet.to_json() for packet in self.packets],
            "micro_recall_at_3": {
                "expected_count": self.micro_expected_count,
                "returned_count": self.micro_returned_count,
                "missing_count": missing,
                "recall": recall,
            },
        }


def retrieve_live_packets(
    suite: EvalSuite,
    settings: RetrievalSettings,
    output_path: Path,
) -> RetrievalSnapshot:
    """Retrieve all 15 cases and atomically replace one deterministic snapshot."""

    prerequisites = validate_live_prerequisites(
        suite, settings, output_path, refresh_report=True
    )
    catalog_by_id = prerequisites.catalog.by_id
    store = VectorStore(
        path=str(settings.store_path),
        collection=settings.collection,
        dense_dim=DENSE_DIM,
    )
    packets: list[RetrievalPacket] = []
    for case in suite.cases:
        raw_hits = retrieve(
            case.question,
            store_path=str(settings.store_path),
            collection=settings.collection,
            embedder_name=settings.embedder,
            reranker_name=settings.reranker,
            top_k=RETRIEVAL_TOP_K,
            top_n=RETRIEVAL_TOP_N,
            max_per_video=RETRIEVAL_MAX_PER_VIDEO,
            window_chars=RETRIEVAL_WINDOW_CHARS,
            store=store,
        )
        hits = _validate_hits(case.id, raw_hits, catalog_by_id)
        returned_ids = tuple(hit.card_id for hit in hits)
        expected_ids = tuple(case.retrieval_relevant_ids)
        expected = set(expected_ids)
        returned_expected_ids = tuple(
            card_id for card_id in returned_ids if card_id in expected
        )
        returned_expected = set(returned_expected_ids)
        missing_expected_ids = tuple(
            card_id for card_id in expected_ids if card_id not in returned_expected
        )
        packets.append(
            RetrievalPacket(
                question_id=case.id,
                question=case.question,
                expected_ids=expected_ids,
                returned_ids=returned_ids,
                returned_expected_ids=returned_expected_ids,
                missing_expected_ids=missing_expected_ids,
                recall_at_3=(
                    len(returned_expected_ids) / len(expected_ids)
                    if expected_ids
                    else 1.0
                ),
                hits=hits,
            )
        )

    immutable_index = {
        key: prerequisites.index_manifest[key]
        for key in (
            "manifest_schema",
            "catalog_sha256",
            "card_count",
            "collection",
            "embedder",
            "dense_dim",
            "store_path",
            "index_fingerprint",
        )
    }
    retrieval_settings = _retrieval_settings_json(settings)
    snapshot = RetrievalSnapshot(
        suite_id=suite.suite_id,
        suite_sha256=suite.sha256,
        model_identity_sha256=prerequisites.model_identity_sha256,
        fixed_gates={
            "diagnostic_input_fingerprint": (
                prerequisites.diagnostic_input_fingerprint
            ),
            "production_input_fingerprint": (
                prerequisites.production_input_fingerprint
            ),
        },
        index=immutable_index,
        retrieval_settings=retrieval_settings,
        packets=tuple(packets),
        micro_expected_count=sum(len(packet.expected_ids) for packet in packets),
        micro_returned_count=sum(
            len(packet.returned_expected_ids) for packet in packets
        ),
    )
    _write_atomic_json(Path(output_path), snapshot.to_json())
    return snapshot


def validate_live_prerequisites(
    suite: EvalSuite,
    settings: RetrievalSettings,
    output_path: Path,
    *,
    refresh_report: bool = True,
) -> LivePrerequisites:
    """Validate fixed gates, run identity, and card index without loading retrieval."""

    if not isinstance(suite, EvalSuite):
        raise TypeError("suite must be an EvalSuite")
    if not isinstance(settings, RetrievalSettings):
        raise TypeError("settings must be RetrievalSettings")
    if len(suite.cases) != 15:
        raise ValueError("live retrieval requires the frozen 15-case suite")
    validation = validate_suite(suite, REPO_ROOT)
    if not validation.ok:
        raise ValueError(
            "suite provenance is not current: " + "; ".join(validation.errors)
        )
    output_path = Path(output_path)
    if (
        output_path.name != "retrieval_snapshot.json"
        or output_path.parent.name != "live"
    ):
        raise ValueError(
            "output path must be <run-dir>/live/retrieval_snapshot.json"
        )
    run_dir = output_path.parents[1]
    run, identity = _load_run_manifest(run_dir / "run.json", suite)
    if refresh_report:
        # This is the authoritative currentness check. It recomputes both fixed
        # input fingerprints from present results/reviews/semantic locks.
        generate_report(run_dir)

    identity_sha256 = canonical_sha256(identity.to_json())
    diagnostic = _load_passing_gate(
        run_dir / "fixed" / "diagnostic-gate.json",
        stage="fixed-diagnostic",
        suite=suite,
        model_identity_sha256=identity_sha256,
    )
    production = _load_passing_gate(
        run_dir / "fixed" / "production-gate.json",
        stage="fixed-production",
        suite=suite,
        model_identity_sha256=identity_sha256,
    )
    catalog = load_card_catalog(settings.cards_path)
    manifest = load_index_manifest(settings.index_manifest_path)
    _validate_index_manifest(manifest, catalog, settings)
    # Keep the manifest/run values tied to the exact objects just validated.
    if run["suite_sha256"] != suite.sha256:
        raise ValueError("run fixture identity is stale")
    return LivePrerequisites(
        run_dir=run_dir,
        model_identity=identity,
        model_identity_sha256=identity_sha256,
        diagnostic_input_fingerprint=diagnostic["input_fingerprint"],
        production_input_fingerprint=production["input_fingerprint"],
        index_manifest=manifest,
        catalog=catalog,
    )


def load_index_manifest(path: Path) -> dict[str, object]:
    """Strictly decode an index manifest, including its immutable fingerprint."""

    raw = _read_strict_json(path)
    if set(raw) != _MANIFEST_FIELDS or raw.get("manifest_schema") != 1:
        raise ValueError("index manifest has missing or unexpected fields")
    for field in (
        "catalog_sha256",
        "collection",
        "embedder",
        "store_path",
        "index_fingerprint",
        "built_at",
    ):
        _strict_text(raw.get(field), f"index manifest {field}")
    _strict_sha256(raw["catalog_sha256"], "index catalog_sha256")
    _strict_sha256(raw["index_fingerprint"], "index fingerprint")
    _strict_positive_int(raw.get("card_count"), "index card_count")
    _strict_positive_int(raw.get("dense_dim"), "index dense_dim")
    try:
        built_at = datetime.fromisoformat(str(raw["built_at"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("index built_at must be an ISO-8601 timestamp") from exc
    if built_at.utcoffset() is None:
        raise ValueError("index built_at must be timezone-aware")
    immutable = {key: raw[key] for key in _IMMUTABLE_MANIFEST_FIELDS}
    if raw["index_fingerprint"] != canonical_sha256(immutable):
        raise ValueError("index fingerprint is stale")
    return raw


def load_retrieval_snapshot(
    path: Path,
    *,
    suite: EvalSuite,
    settings: RetrievalSettings,
) -> RetrievalSnapshot:
    """Load a snapshot and revalidate it against the current frozen identities."""

    raw = _read_strict_json(path)
    if set(raw) != _SNAPSHOT_FIELDS or raw.get("snapshot_schema") != SNAPSHOT_SCHEMA:
        raise ValueError("retrieval snapshot has missing or unexpected fields")
    prerequisites = validate_live_prerequisites(
        suite, settings, Path(path), refresh_report=True
    )
    if raw.get("suite_id") != suite.suite_id or raw.get("suite_sha256") != suite.sha256:
        raise ValueError("retrieval snapshot suite identity is stale")
    if raw.get("model_identity_sha256") != prerequisites.model_identity_sha256:
        raise ValueError("retrieval snapshot model identity is stale")
    fixed_gates = raw.get("fixed_gates")
    expected_gates = {
        "diagnostic_input_fingerprint": prerequisites.diagnostic_input_fingerprint,
        "production_input_fingerprint": prerequisites.production_input_fingerprint,
    }
    if type(fixed_gates) is not dict or set(fixed_gates) != _FIXED_GATE_ID_FIELDS:
        raise ValueError("retrieval snapshot fixed-gate identity is invalid")
    if fixed_gates != expected_gates:
        raise ValueError("retrieval snapshot fixed-gate identity is stale")
    expected_index = {
        key: prerequisites.index_manifest[key]
        for key in prerequisites.index_manifest
        if key != "built_at"
    }
    if raw.get("index") != expected_index:
        raise ValueError("retrieval snapshot index identity is stale")
    expected_settings = _retrieval_settings_json(settings)
    if raw.get("retrieval_settings") != expected_settings:
        raise ValueError("retrieval snapshot settings are stale")

    packet_values = raw.get("packets")
    if type(packet_values) is not list or len(packet_values) != len(suite.cases):
        raise ValueError("retrieval snapshot must contain exactly 15 packets")
    if [item.get("question_id") for item in packet_values if type(item) is dict] != [
        case.id for case in suite.cases
    ]:
        raise ValueError("retrieval snapshot packets are not in suite order")
    catalog_by_id = prerequisites.catalog.by_id
    packets = tuple(
        _packet_from_json(raw_packet, case, catalog_by_id)
        for raw_packet, case in zip(packet_values, suite.cases)
    )
    expected_count = sum(len(packet.expected_ids) for packet in packets)
    returned_count = sum(len(packet.returned_expected_ids) for packet in packets)
    micro = raw.get("micro_recall_at_3")
    expected_micro = {
        "expected_count": expected_count,
        "returned_count": returned_count,
        "missing_count": expected_count - returned_count,
        "recall": returned_count / expected_count if expected_count else 1.0,
    }
    if type(micro) is not dict or set(micro) != _MICRO_FIELDS or micro != expected_micro:
        raise ValueError("retrieval snapshot micro recall is inconsistent")
    return RetrievalSnapshot(
        suite_id=suite.suite_id,
        suite_sha256=suite.sha256,
        model_identity_sha256=prerequisites.model_identity_sha256,
        fixed_gates=dict(fixed_gates),
        index=dict(expected_index),
        retrieval_settings=dict(expected_settings),
        packets=packets,
        micro_expected_count=expected_count,
        micro_returned_count=returned_count,
    )


def _load_run_manifest(
    path: Path, suite: EvalSuite
) -> tuple[dict[str, object], ModelIdentity]:
    raw = _read_strict_json(path)
    if set(raw) != _RUN_FIELDS or raw.get("run_schema") != 1:
        raise ValueError("run manifest has missing or unexpected fields")
    if raw.get("suite_id") != suite.suite_id or raw.get("suite_sha256") != suite.sha256:
        raise ValueError("run manifest fixture identity is stale")
    suite_path = Path(_strict_text(raw.get("suite_path"), "run suite_path"))
    run_suite = load_suite(suite_path)
    if run_suite.sha256 != suite.sha256 or run_suite.suite_id != suite.suite_id:
        raise ValueError("run manifest suite_path points to different fixture bytes")
    identity_value = raw.get("model_identity")
    if type(identity_value) is not dict or set(identity_value) != _IDENTITY_FIELDS:
        raise ValueError("run model identity has missing or unexpected fields")
    identity = ModelIdentity(**identity_value)
    if identity.digest != OFFICIAL_MODEL_DIGEST:
        raise ValueError("run model is not the pinned official Qwen3 digest")
    for field, expected in OFFICIAL_MODEL_METADATA.items():
        if getattr(identity, field) != expected:
            raise ValueError(f"run model has unexpected {field}")
    if identity.name != suite.model_policy.required_model:
        raise ValueError("run model differs from suite model policy")
    return raw, identity


def _load_passing_gate(
    path: Path,
    *,
    stage: str,
    suite: EvalSuite,
    model_identity_sha256: str,
) -> dict[str, object]:
    raw = _read_strict_json(path)
    if set(raw) != _GATE_FIELDS or raw.get("gate_schema") != 1:
        raise ValueError(f"{stage} gate has missing or unexpected fields")
    if raw.get("stage") != stage or raw.get("suite_sha256") != suite.sha256:
        raise ValueError(f"{stage} gate fixture identity is stale")
    if raw.get("model_identity_sha256") != model_identity_sha256:
        raise ValueError(f"{stage} gate model identity is stale")
    _strict_sha256(raw.get("input_fingerprint"), f"{stage} input fingerprint")
    gate = raw.get("gate")
    if type(gate) is not dict or set(gate) != _GATE_RESULT_FIELDS:
        raise ValueError(f"{stage} gate result has missing or unexpected fields")
    if (
        gate.get("gate_id") != stage
        or gate.get("status") != "passed"
        or gate.get("semantic_passed") is not True
    ):
        raise ValueError(f"{stage} semantic gate is not passed")
    expected_per_question = 2 if stage == "fixed-diagnostic" else 3
    expected_total = len(suite.cases) * expected_per_question
    for field in ("expected_count", "result_count", "reviewed_count"):
        if gate.get(field) != expected_total:
            raise ValueError(f"{stage} gate {field} is incomplete")
    full_pass_count = gate.get("full_pass_count")
    minimum = expected_total if stage == "fixed-diagnostic" else 42
    if type(full_pass_count) is not int or full_pass_count < minimum:
        raise ValueError(f"{stage} gate full-pass threshold is not met")
    for field in (
        "missing_result_keys",
        "missing_review_keys",
        "critical_failure_counts",
        "styled_regressions",
        "reasons",
    ):
        if gate.get(field) != []:
            raise ValueError(f"{stage} passing gate contains {field}")
    per_question = gate.get("per_question")
    if type(per_question) is not list or len(per_question) != len(suite.cases):
        raise ValueError(f"{stage} gate per-question results are incomplete")
    if [item.get("question_id") for item in per_question if type(item) is dict] != [
        case.id for case in suite.cases
    ]:
        raise ValueError(f"{stage} gate questions are not in suite order")
    per_question_passes = 0
    for item in per_question:
        if type(item) is not dict or set(item) != _QUESTION_GATE_FIELDS:
            raise ValueError(f"{stage} gate has invalid per-question fields")
        if (
            item.get("expected_count") != expected_per_question
            or item.get("reviewed_count") != expected_per_question
            or type(item.get("full_pass_count")) is not int
            or item["full_pass_count"] < 2
            or item.get("critical_failures") != []
        ):
            raise ValueError(f"{stage} per-question threshold is not met")
        per_question_passes += item["full_pass_count"]
    if per_question_passes != full_pass_count:
        raise ValueError(f"{stage} gate full-pass count is inconsistent")
    return raw


def _validate_index_manifest(
    manifest: Mapping[str, object],
    catalog: CardCatalog,
    settings: RetrievalSettings,
) -> None:
    expected = {
        "catalog_sha256": catalog.sha256,
        "card_count": len(catalog.cards),
        "collection": settings.collection,
        "embedder": settings.embedder,
        "dense_dim": DENSE_DIM,
        "store_path": str(settings.store_path),
    }
    mismatches = [
        field for field, expected_value in expected.items() if manifest[field] != expected_value
    ]
    if mismatches:
        raise ValueError(
            "index manifest is stale for: " + ", ".join(sorted(mismatches))
        )


def _validate_hits(
    question_id: str,
    raw_hits: object,
    catalog_by_id: Mapping[str, object],
) -> tuple[RetrievalHit, ...]:
    if type(raw_hits) is not list:
        raise ValueError(f"{question_id} retrieval must return a list")
    if len(raw_hits) > RETRIEVAL_TOP_N:
        raise ValueError(f"{question_id} returned more than top_n hits")
    seen: set[str] = set()
    records: list[RetrievalHit] = []
    previous_score: float | None = None
    for rank, hit in enumerate(raw_hits, 1):
        if not isinstance(hit, Hit):
            raise ValueError(f"{question_id} returned a non-Hit value")
        card_id = hit.chunk.id
        if card_id in seen:
            raise ValueError(f"{question_id} returned duplicate card ID {card_id}")
        seen.add(card_id)
        canonical = catalog_by_id.get(card_id)
        if canonical is None:
            raise ValueError(f"{question_id} returned unknown card ID {card_id}")
        if hit.chunk.text != canonical.take:
            raise ValueError(
                f"{question_id} retrieved text differs from canonical card {card_id}"
            )
        dense_score = _strict_finite(hit.dense_score, f"{question_id} dense score")
        sparse_score = _strict_finite(hit.sparse_score, f"{question_id} sparse score")
        rerank_score = _strict_finite(hit.rerank_score, f"{question_id} rerank score")
        if previous_score is not None and rerank_score > previous_score:
            raise ValueError(f"{question_id} hits are not sorted by rerank score")
        previous_score = rerank_score
        records.append(
            RetrievalHit(
                rank=rank,
                card_id=card_id,
                take=canonical.take,
                dense_score=dense_score,
                sparse_score=sparse_score,
                rerank_score=rerank_score,
            )
        )
    return tuple(records)


def _packet_from_json(
    value: object,
    case,
    catalog_by_id: Mapping[str, object],
) -> RetrievalPacket:
    if type(value) is not dict or set(value) != _PACKET_FIELDS:
        raise ValueError(f"{case.id} packet has missing or unexpected fields")
    if value.get("question_id") != case.id or value.get("question") != case.question:
        raise ValueError(f"{case.id} packet question identity is stale")
    expected_ids = _strict_string_list(value.get("expected_ids"), "expected_ids")
    if expected_ids != case.retrieval_relevant_ids:
        raise ValueError(f"{case.id} packet expected IDs are stale")
    hit_values = value.get("hits")
    if type(hit_values) is not list or len(hit_values) > RETRIEVAL_TOP_N:
        raise ValueError(f"{case.id} packet hits are invalid")
    hits: list[RetrievalHit] = []
    seen: set[str] = set()
    previous: float | None = None
    for expected_rank, hit in enumerate(hit_values, 1):
        if type(hit) is not dict or set(hit) != _HIT_FIELDS:
            raise ValueError(f"{case.id} packet hit has invalid fields")
        if hit.get("rank") != expected_rank:
            raise ValueError(f"{case.id} packet hit ranks are not contiguous")
        card_id = _strict_text(hit.get("card_id"), f"{case.id} card_id")
        if card_id in seen:
            raise ValueError(f"{case.id} packet contains duplicate IDs")
        seen.add(card_id)
        canonical = catalog_by_id.get(card_id)
        if canonical is None:
            raise ValueError(f"{case.id} packet contains an unknown card")
        if hit.get("take") != canonical.take:
            raise ValueError(f"{case.id} packet take differs from canonical catalog")
        dense = _strict_finite(hit.get("dense_score"), f"{case.id} dense score")
        sparse = _strict_finite(hit.get("sparse_score"), f"{case.id} sparse score")
        rerank = _strict_finite(hit.get("rerank_score"), f"{case.id} rerank score")
        if previous is not None and rerank > previous:
            raise ValueError(f"{case.id} packet hits are not sorted")
        previous = rerank
        hits.append(
            RetrievalHit(
                rank=expected_rank,
                card_id=card_id,
                take=canonical.take,
                dense_score=dense,
                sparse_score=sparse,
                rerank_score=rerank,
            )
        )
    returned_ids = tuple(hit.card_id for hit in hits)
    expected = set(expected_ids)
    returned_expected_ids = tuple(item for item in returned_ids if item in expected)
    returned_expected = set(returned_expected_ids)
    missing = tuple(item for item in expected_ids if item not in returned_expected)
    expected_fields = {
        "returned_ids": returned_ids,
        "returned_expected_ids": returned_expected_ids,
        "missing_expected_ids": missing,
    }
    for field, expected_value in expected_fields.items():
        if _strict_string_list(value.get(field), field) != expected_value:
            raise ValueError(f"{case.id} packet {field} is inconsistent")
    recall = len(returned_expected_ids) / len(expected_ids) if expected_ids else 1.0
    if _strict_finite(value.get("recall_at_3"), "recall_at_3") != recall:
        raise ValueError(f"{case.id} packet recall is inconsistent")
    return RetrievalPacket(
        question_id=case.id,
        question=case.question,
        expected_ids=expected_ids,
        returned_ids=returned_ids,
        returned_expected_ids=returned_expected_ids,
        missing_expected_ids=missing,
        recall_at_3=recall,
        hits=tuple(hits),
    )


def _retrieval_settings_json(settings: RetrievalSettings) -> dict[str, object]:
    return {
        "top_k": RETRIEVAL_TOP_K,
        "top_n": RETRIEVAL_TOP_N,
        "max_per_video": RETRIEVAL_MAX_PER_VIDEO,
        "window_chars": RETRIEVAL_WINDOW_CHARS,
        "collection": settings.collection,
        "embedder": settings.embedder,
        "reranker": settings.reranker,
        "store_path": str(settings.store_path),
    }


def _read_strict_json(path: Path) -> dict[str, Any]:
    payload = Path(path).read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
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
    _validate_unicode_tree(raw)
    return raw


def _write_atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"
        temp.write_text(payload, encoding="utf-8", newline="\n")
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def _object_without_duplicate_keys(pairs):
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON value {value}")


def _validate_unicode_tree(value: object, path: str = "$") -> None:
    if type(value) is str:
        _strict_text(value, path)
    elif type(value) is list:
        for index, item in enumerate(value):
            _validate_unicode_tree(item, f"{path}[{index}]")
    elif type(value) is dict:
        for key, item in value.items():
            _strict_text(key, f"{path}.<key>")
            _validate_unicode_tree(item, f"{path}.{key}")


def _strict_text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    if (
        not unicodedata.is_normalized("NFC", value)
        or "\ufffd" in value
        or any(marker in value for marker in _MOJIBAKE_MARKERS)
        or any(0xD800 <= ord(character) <= 0xDFFF for character in value)
    ):
        raise ValueError(f"{label} contains corrupt Unicode")
    return value


def _strict_string_list(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not list:
        raise ValueError(f"{label} must be an array")
    result = tuple(_strict_text(item, label) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _strict_finite(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _strict_positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _strict_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--cards", type=Path, default=Path("data/cards/speaker_cards.json")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/cards/speaker_cards.index_manifest.json"),
    )
    parser.add_argument("--collection", default="speaker_cards")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    rag = load_settings().rag
    suite = load_suite(args.suite)
    settings = RetrievalSettings(
        cards_path=args.cards,
        index_manifest_path=args.manifest,
        store_path=rag.store_path,
        collection=args.collection,
        embedder=rag.embedder,
        reranker=rag.reranker,
    )
    snapshot = retrieve_live_packets(
        suite, settings, args.run_dir / "live" / "retrieval_snapshot.json"
    )
    micro = snapshot.to_json()["micro_recall_at_3"]
    print(
        f"Retrieved {len(snapshot.packets)} packets; "
        f"micro recall@3 {micro['returned_count']}/{micro['expected_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
