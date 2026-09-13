from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eval.card_eval_ollama import (
    ModelIdentity,
    OFFICIAL_MODEL_DIGEST,
    OFFICIAL_MODEL_METADATA,
)
from eval.card_eval_schema import canonical_sha256, load_suite
from rag.types import Chunk, Hit


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_PATH = REPO_ROOT / "eval" / "fixtures" / "verified_card_eval_v1.json"
CATALOG_PATH = REPO_ROOT / "data" / "cards" / "speaker_cards.json"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _catalog_card(card_id: str, *, take: str = "Kanıt metni") -> dict[str, object]:
    return {
        "id": card_id,
        "q": "Soru nedir?",
        "take": take,
        "n_takes": 1,
        "alt_qs": [],
        "sources": [
            {
                "video_id": "video-1",
                "title": "Başlık",
                "start": 12.5,
                "score": 3.25,
            }
        ],
        "support": 3.25,
        "n_strong": 1,
        "date": "20260711",
    }


class _FakeEmbedder:
    def encode(self, texts: list[str]):
        return SimpleNamespace(
            dense=np.zeros((len(texts), 1024), dtype=np.float32),
            sparse=[{1: 0.5} for _ in texts],
        )


class _BrokenEmbedder:
    def __init__(self, defect: str):
        self.defect = defect

    def encode(self, texts: list[str]):
        count = len(texts)
        dense = np.zeros((count, 1024), dtype=np.float32)
        sparse = [{1: 0.5} for _ in texts]
        if self.defect == "dense_rows":
            dense = dense[:-1]
        elif self.defect == "dense_dim":
            dense = np.zeros((count, 8), dtype=np.float32)
        elif self.defect == "dense_nonfinite":
            dense[0, 0] = np.nan
        elif self.defect == "sparse_rows":
            sparse = sparse[:-1]
        else:
            sparse[0] = {1: math.inf}
        return SimpleNamespace(dense=dense, sparse=sparse)


class _RecordingStore:
    instances: list["_RecordingStore"] = []
    manifest_path: Path | None = None
    fail_upsert = False

    def __init__(self, path: str, collection: str, dense_dim: int):
        self.path = path
        self.collection = collection
        self.dense_dim = dense_dim
        self.ensure_calls = 0
        self.upserts: list[tuple[list[Chunk], np.ndarray, list[dict[int, float]]]] = []
        self.__class__.instances.append(self)

    def ensure_collection(self) -> None:
        assert self.manifest_path is not None
        assert not self.manifest_path.exists(), "old manifest must be invalidated first"
        self.ensure_calls += 1

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        dense: np.ndarray,
        sparse: list[dict[int, float]],
    ) -> None:
        if self.fail_upsert:
            raise RuntimeError("upsert failed")
        self.upserts.append((chunks, dense, sparse))


def test_build_card_index_invalidates_old_manifest_and_fingerprints_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from rag import index_cards

    cards_path = tmp_path / "cards.json"
    manifest_path = tmp_path / "cards.index_manifest.json"
    cards = [_catalog_card("card::0001"), _catalog_card("card::0002")]
    _write_json(cards_path, cards)
    manifest_path.write_text("stale", encoding="utf-8")
    store_path = (tmp_path / "qdrant").resolve()
    rag = SimpleNamespace(embedder="BAAI/bge-m3", store_path=store_path)
    monkeypatch.setattr(index_cards, "load_settings", lambda: SimpleNamespace(rag=rag))
    monkeypatch.setattr(index_cards, "get_embedder", lambda _name: _FakeEmbedder())
    _RecordingStore.instances.clear()
    _RecordingStore.manifest_path = manifest_path
    _RecordingStore.fail_upsert = False
    monkeypatch.setattr(index_cards, "VectorStore", _RecordingStore)

    manifest = index_cards.build_card_index(
        cards_path, "speaker_cards_test", manifest_path
    )

    store = _RecordingStore.instances[-1]
    assert store.ensure_calls == 1
    assert [chunk.id for chunk in store.upserts[0][0]] == [
        "card::0001",
        "card::0002",
    ]
    assert [chunk.text for chunk in store.upserts[0][0]] == [
        "Kanıt metni",
        "Kanıt metni",
    ]
    assert manifest == json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["catalog_sha256"] == __import__("hashlib").sha256(
        cards_path.read_bytes()
    ).hexdigest()
    assert manifest["card_count"] == 2
    assert manifest["collection"] == "speaker_cards_test"
    assert manifest["embedder"] == "BAAI/bge-m3"
    assert manifest["dense_dim"] == 1024
    assert manifest["store_path"] == str(store_path)
    built_at = datetime.fromisoformat(manifest["built_at"].replace("Z", "+00:00"))
    assert built_at.utcoffset() is not None
    immutable = {
        key: value
        for key, value in manifest.items()
        if key not in {"built_at", "index_fingerprint"}
    }
    assert manifest["index_fingerprint"] == canonical_sha256(immutable)


def test_failed_rebuild_leaves_no_trustworthy_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from rag import index_cards

    cards_path = tmp_path / "cards.json"
    manifest_path = tmp_path / "cards.index_manifest.json"
    _write_json(cards_path, [_catalog_card("card::0001")])
    manifest_path.write_text("stale", encoding="utf-8")
    rag = SimpleNamespace(embedder="BAAI/bge-m3", store_path=tmp_path / "qdrant")
    monkeypatch.setattr(index_cards, "load_settings", lambda: SimpleNamespace(rag=rag))
    monkeypatch.setattr(index_cards, "get_embedder", lambda _name: _FakeEmbedder())
    _RecordingStore.instances.clear()
    _RecordingStore.manifest_path = manifest_path
    _RecordingStore.fail_upsert = True
    monkeypatch.setattr(index_cards, "VectorStore", _RecordingStore)

    with pytest.raises(RuntimeError, match="upsert failed"):
        index_cards.build_card_index(cards_path, "speaker_cards", manifest_path)

    assert not manifest_path.exists()
    assert not list(tmp_path.glob("*.tmp-*"))


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda cards: cards.append(cards[0].copy()), "duplicate card ID"),
        (lambda cards: cards[0].update(extra=True), "unexpected fields"),
        (lambda cards: cards[0].update(support=math.nan), "finite"),
        (lambda cards: cards[0].update(take="bozuk \ufffd"), "Unicode"),
    ],
)
def test_invalid_catalog_is_rejected_before_manifest_invalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
):
    from rag import index_cards

    cards = [_catalog_card("card::0001")]
    mutation(cards)
    cards_path = tmp_path / "cards.json"
    manifest_path = tmp_path / "cards.index_manifest.json"
    _write_json(cards_path, cards)
    manifest_path.write_text("still-current", encoding="utf-8")
    monkeypatch.setattr(
        index_cards,
        "load_settings",
        lambda: SimpleNamespace(
            rag=SimpleNamespace(embedder="BAAI/bge-m3", store_path=tmp_path / "qdrant")
        ),
    )

    with pytest.raises(ValueError, match=message):
        index_cards.build_card_index(cards_path, "speaker_cards", manifest_path)

    assert manifest_path.read_text(encoding="utf-8") == "still-current"


@pytest.mark.parametrize(
    "defect",
    [
        "dense_rows",
        "dense_dim",
        "dense_nonfinite",
        "sparse_rows",
        "sparse_nonfinite",
    ],
)
def test_invalid_embedding_batch_cannot_publish_full_count_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
):
    from rag import index_cards

    cards_path = tmp_path / "cards.json"
    manifest_path = tmp_path / "cards.index_manifest.json"
    _write_json(
        cards_path,
        [_catalog_card("card::0001"), _catalog_card("card::0002")],
    )
    manifest_path.write_text("stale", encoding="utf-8")
    rag = SimpleNamespace(embedder="BAAI/bge-m3", store_path=tmp_path / "qdrant")
    monkeypatch.setattr(index_cards, "load_settings", lambda: SimpleNamespace(rag=rag))
    monkeypatch.setattr(
        index_cards, "get_embedder", lambda _name: _BrokenEmbedder(defect)
    )
    _RecordingStore.instances.clear()
    _RecordingStore.manifest_path = manifest_path
    _RecordingStore.fail_upsert = False
    monkeypatch.setattr(index_cards, "VectorStore", _RecordingStore)

    with pytest.raises(ValueError, match="embedding"):
        index_cards.build_card_index(cards_path, "speaker_cards", manifest_path)

    assert not manifest_path.exists()
    assert _RecordingStore.instances[-1].upserts == []


def test_common_mojibake_is_rejected_by_catalog_and_snapshot_text_guards(
    tmp_path: Path
):
    from eval import card_eval_retrieve_worker as worker
    from rag import index_cards

    cards_path = tmp_path / "cards.json"
    _write_json(
        cards_path,
        [_catalog_card("card::0001", take="hakk\u00c4\u00b1nda")],
    )
    with pytest.raises(ValueError, match="Unicode"):
        index_cards.load_card_catalog(cards_path)
    with pytest.raises(ValueError, match="Unicode"):
        worker._strict_text("hakk\u00c4\u00b1nda", "snapshot text")


def _identity() -> ModelIdentity:
    return ModelIdentity(
        name="qwen3:14b",
        digest=OFFICIAL_MODEL_DIGEST,
        ollama_version="0.31.2",
        format=OFFICIAL_MODEL_METADATA["format"],
        family=OFFICIAL_MODEL_METADATA["family"],
        parameter_size=OFFICIAL_MODEL_METADATA["parameter_size"],
        quantization_level=OFFICIAL_MODEL_METADATA["quantization_level"],
        file_type=15,
        quantization_version=2,
    )


def _passing_gate(stage: str, suite, identity: ModelIdentity) -> dict[str, object]:
    is_diagnostic = stage == "fixed-diagnostic"
    per_question = [
        {
            "question_id": case.id,
            "expected_count": 2 if is_diagnostic else 3,
            "reviewed_count": 2 if is_diagnostic else 3,
            "full_pass_count": 2 if is_diagnostic else 3,
            "critical_failures": [],
            "style_scores": [],
            "style_median": None,
        }
        for case in suite.cases
    ]
    expected = 30 if is_diagnostic else 45
    fingerprint = canonical_sha256({"stage": stage, "current": True})
    return {
        "gate_schema": 1,
        "stage": stage,
        "suite_sha256": suite.sha256,
        "model_identity_sha256": canonical_sha256(identity.to_json()),
        "input_fingerprint": fingerprint,
        "gate": {
            "gate_id": stage,
            "status": "passed",
            "expected_count": expected,
            "result_count": expected,
            "reviewed_count": expected,
            "full_pass_count": expected,
            "missing_result_keys": [],
            "missing_review_keys": [],
            "failed_answer_keys": [],
            "critical_failure_counts": [],
            "styled_regressions": [],
            "per_question": per_question,
            "semantic_passed": True,
            "style_status": "passed",
            "style_mean": 2.5,
            "reasons": [],
            "style_reasons": [],
        },
    }


def _worker_environment(tmp_path: Path):
    from eval.card_eval_retrieve_worker import RetrievalSettings

    suite = load_suite(SUITE_PATH)
    identity = _identity()
    run_dir = tmp_path / "run"
    output_path = run_dir / "live" / "retrieval_snapshot.json"
    store_path = (tmp_path / "qdrant").resolve()
    manifest_path = tmp_path / "speaker_cards.index_manifest.json"
    raw = CATALOG_PATH.read_bytes()
    catalog = json.loads(raw.decode("utf-8"))
    immutable_manifest = {
        "manifest_schema": 1,
        "catalog_sha256": __import__("hashlib").sha256(raw).hexdigest(),
        "card_count": len(catalog),
        "collection": "speaker_cards",
        "embedder": "BAAI/bge-m3",
        "dense_dim": 1024,
        "store_path": str(store_path),
    }
    manifest = {
        **immutable_manifest,
        "index_fingerprint": canonical_sha256(immutable_manifest),
        "built_at": "2026-07-11T12:00:00Z",
    }
    _write_json(manifest_path, manifest)
    run = {
        "run_schema": 1,
        "suite_path": str(SUITE_PATH.resolve()),
        "suite_id": suite.suite_id,
        "suite_sha256": suite.sha256,
        "model_identity": identity.to_json(),
    }
    _write_json(run_dir / "run.json", run)
    diagnostic = _passing_gate("fixed-diagnostic", suite, identity)
    production = _passing_gate("fixed-production", suite, identity)
    _write_json(run_dir / "fixed" / "diagnostic-gate.json", diagnostic)
    _write_json(run_dir / "fixed" / "production-gate.json", production)
    settings = RetrievalSettings(
        cards_path=CATALOG_PATH,
        index_manifest_path=manifest_path,
        store_path=store_path,
        collection="speaker_cards",
        embedder="BAAI/bge-m3",
        reranker="BAAI/bge-reranker-v2-m3",
    )
    return suite, settings, output_path, diagnostic, production


def _catalog_take_map() -> dict[str, str]:
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return {card["id"]: card["take"] for card in raw}


def _successful_retriever(suite):
    takes = _catalog_take_map()
    fallback_ids = [card.id for case in suite.cases for card in case.cards]
    calls: list[tuple[str, dict[str, object]]] = []

    def fake_retrieve(query: str, **kwargs):
        calls.append((query, kwargs))
        case = suite.cases[(len(calls) - 1) % len(suite.cases)]
        ids = list(case.retrieval_relevant_ids)
        ids.extend(card_id for card_id in fallback_ids if card_id not in ids)
        ids = ids[:3]
        return [
            Hit(
                chunk=Chunk(
                    id=card_id,
                    video_id=f"video-{rank}",
                    start=float(rank),
                    end=0.0,
                    title=f"Başlık {rank}",
                    text=takes[card_id],
                ),
                dense_score=1.0 / rank,
                sparse_score=2.0 / rank,
                rerank_score=4.0 - rank,
            )
            for rank, card_id in enumerate(ids, 1)
        ]

    return calls, fake_retrieve


def _mock_worker_store(worker, monkeypatch: pytest.MonkeyPatch):
    stores = []

    def factory(**kwargs):
        store = SimpleNamespace(**kwargs)
        stores.append(store)
        return store

    monkeypatch.setattr(worker, "VectorStore", factory)
    return stores


def test_worker_refreshes_gates_then_makes_15_frozen_ordered_calls_and_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from eval import card_eval_retrieve_worker as worker

    suite, settings, output_path, diagnostic, production = _worker_environment(tmp_path)
    refresh_calls: list[Path] = []

    def fake_generate_report(run_dir: Path):
        refresh_calls.append(run_dir)
        return {"stages": []}

    calls, fake_retrieve = _successful_retriever(suite)
    monkeypatch.setattr(worker, "generate_report", fake_generate_report)
    monkeypatch.setattr(worker, "retrieve", fake_retrieve)
    created_stores = _mock_worker_store(worker, monkeypatch)

    snapshot = worker.retrieve_live_packets(suite, settings, output_path)
    first_bytes = output_path.read_bytes()
    second = worker.retrieve_live_packets(suite, settings, output_path)

    assert refresh_calls == [output_path.parents[1], output_path.parents[1]]
    assert len(calls) == 30
    assert [query for query, _ in calls[:15]] == [case.question for case in suite.cases]
    first_store = calls[0][1]["store"]
    second_store = calls[15][1]["store"]
    assert first_store is not second_store
    assert all(kwargs["store"] is first_store for _, kwargs in calls[:15])
    assert all(kwargs["store"] is second_store for _, kwargs in calls[15:])
    assert created_stores == [first_store, second_store]
    expected_args = {
        "store_path": str(settings.store_path),
        "collection": "speaker_cards",
        "embedder_name": "BAAI/bge-m3",
        "reranker_name": "BAAI/bge-reranker-v2-m3",
        "top_k": 20,
        "top_n": 3,
        "max_per_video": 0,
        "window_chars": None,
    }
    assert all(
        {key: value for key, value in kwargs.items() if key != "store"}
        == expected_args
        for _, kwargs in calls
    )
    assert [packet.question_id for packet in snapshot.packets] == [
        case.id for case in suite.cases
    ]
    assert snapshot.to_json() == second.to_json()
    assert output_path.read_bytes() == first_bytes
    raw_snapshot = json.loads(first_bytes.decode("utf-8"))
    assert "created_at" not in raw_snapshot
    assert "built_at" not in json.dumps(raw_snapshot)
    assert raw_snapshot["fixed_gates"] == {
        "diagnostic_input_fingerprint": diagnostic["input_fingerprint"],
        "production_input_fingerprint": production["input_fingerprint"],
    }
    first_hit = raw_snapshot["packets"][0]["hits"][0]
    assert first_hit["rank"] == 1
    assert first_hit["take"] == _catalog_take_map()[first_hit["card_id"]]
    assert first_hit["dense_score"] == 1.0
    assert first_hit["sparse_score"] == 2.0
    assert first_hit["rerank_score"] == 3.0
    assert raw_snapshot["packets"][0]["expected_ids"] == list(
        suite.cases[0].retrieval_relevant_ids
    )
    assert raw_snapshot["micro_recall_at_3"]["expected_count"] == sum(
        len(case.retrieval_relevant_ids) for case in suite.cases
    )
    assert raw_snapshot["micro_recall_at_3"]["missing_count"] == 0
    assert raw_snapshot["micro_recall_at_3"]["recall"] == 1.0


@pytest.mark.parametrize(
    "defect, message",
    [
        ("duplicate", "duplicate"),
        ("unknown", "unknown"),
        ("text", "canonical"),
        ("unsorted", "sorted"),
        ("nonfinite", "finite"),
    ],
)
def test_retrieval_defects_preserve_existing_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
    message: str,
):
    from eval import card_eval_retrieve_worker as worker

    suite, settings, output_path, _diagnostic, _production = _worker_environment(
        tmp_path
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"existing-snapshot")
    monkeypatch.setattr(worker, "generate_report", lambda _run_dir: {})
    _calls, good = _successful_retriever(suite)

    def broken(query: str, **kwargs):
        hits = good(query, **kwargs)
        if defect == "duplicate":
            hits[1].chunk = hits[0].chunk
        elif defect == "unknown":
            hits[0].chunk = Chunk(
                id="card::9999",
                video_id="v",
                start=0,
                end=0,
                title="t",
                text="x",
            )
        elif defect == "text":
            hits[0].chunk = Chunk(
                **{**asdict(hits[0].chunk), "text": "değiştirilmiş"}
            )
        elif defect == "unsorted":
            hits[0].rerank_score = -100.0
        else:
            hits[0].dense_score = math.nan
        return hits

    monkeypatch.setattr(worker, "retrieve", broken)
    _mock_worker_store(worker, monkeypatch)

    with pytest.raises(ValueError, match=message):
        worker.retrieve_live_packets(suite, settings, output_path)

    assert output_path.read_bytes() == b"existing-snapshot"


@pytest.mark.parametrize("stale", ["diagnostic", "production", "index"])
def test_stale_prerequisite_blocks_before_retrieval_and_preserves_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stale: str,
):
    from eval import card_eval_retrieve_worker as worker

    suite, settings, output_path, diagnostic, production = _worker_environment(tmp_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"existing-snapshot")
    if stale == "diagnostic":
        diagnostic["model_identity_sha256"] = "0" * 64
        _write_json(output_path.parents[1] / "fixed" / "diagnostic-gate.json", diagnostic)
    elif stale == "production":
        production["gate"]["status"] = "failed"
        production["gate"]["semantic_passed"] = False
        _write_json(output_path.parents[1] / "fixed" / "production-gate.json", production)
    else:
        manifest = json.loads(settings.index_manifest_path.read_text(encoding="utf-8"))
        manifest["catalog_sha256"] = "0" * 64
        _write_json(settings.index_manifest_path, manifest)
    monkeypatch.setattr(worker, "generate_report", lambda _run_dir: {})
    retrieve_calls = 0

    def forbidden_retrieve(*args, **kwargs):
        nonlocal retrieve_calls
        retrieve_calls += 1
        raise AssertionError("retrieval must stay gated")

    monkeypatch.setattr(worker, "retrieve", forbidden_retrieve)

    with pytest.raises(ValueError):
        worker.retrieve_live_packets(suite, settings, output_path)

    assert retrieve_calls == 0
    assert output_path.read_bytes() == b"existing-snapshot"


def test_load_snapshot_rejects_tampered_order(tmp_path: Path, monkeypatch):
    from eval import card_eval_retrieve_worker as worker

    suite, settings, output_path, _diagnostic, _production = _worker_environment(
        tmp_path
    )
    monkeypatch.setattr(worker, "generate_report", lambda _run_dir: {})
    _calls, fake_retrieve = _successful_retriever(suite)
    monkeypatch.setattr(worker, "retrieve", fake_retrieve)
    _mock_worker_store(worker, monkeypatch)
    worker.retrieve_live_packets(suite, settings, output_path)
    raw = json.loads(output_path.read_text(encoding="utf-8"))
    raw["packets"] = list(reversed(raw["packets"]))
    _write_json(output_path, raw)

    with pytest.raises(ValueError, match="suite order"):
        worker.load_retrieval_snapshot(output_path, suite=suite, settings=settings)


def test_load_snapshot_refreshes_fixed_gates_before_accepting_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from eval import card_eval_retrieve_worker as worker

    suite, settings, output_path, diagnostic, _production = _worker_environment(
        tmp_path
    )
    refresh_count = 0

    def refresh(_run_dir: Path):
        nonlocal refresh_count
        refresh_count += 1
        if refresh_count == 2:
            diagnostic["input_fingerprint"] = "f" * 64
            _write_json(
                output_path.parents[1] / "fixed" / "diagnostic-gate.json",
                diagnostic,
            )
        return {}

    monkeypatch.setattr(worker, "generate_report", refresh)
    _calls, fake_retrieve = _successful_retriever(suite)
    monkeypatch.setattr(worker, "retrieve", fake_retrieve)
    _mock_worker_store(worker, monkeypatch)
    worker.retrieve_live_packets(suite, settings, output_path)

    with pytest.raises(ValueError, match="fixed-gate identity is stale"):
        worker.load_retrieval_snapshot(
            output_path, suite=suite, settings=settings
        )
    assert refresh_count == 2


def test_retrieve_accepts_shared_store_without_constructing_another(monkeypatch):
    import importlib

    retrieve_module = importlib.import_module("rag.retrieve")

    class Embedder:
        def encode(self, _texts):
            return SimpleNamespace(dense=np.zeros((1, 4)), sparse=[{}])

    class Reranker:
        def score(self, _query, texts):
            return [float(len(texts) - index) for index, _ in enumerate(texts)]

    class SharedStore:
        def __init__(self):
            self.calls = 0

        def hybrid_search(self, **_kwargs):
            self.calls += 1
            return [
                Hit(
                    chunk=Chunk(
                        id="card::0001",
                        video_id="v",
                        start=0.0,
                        end=0.0,
                        title="t",
                        text="kanıt",
                    )
                )
            ]

    monkeypatch.setattr(retrieve_module, "get_embedder", lambda _name: Embedder())
    monkeypatch.setattr(retrieve_module, "get_reranker", lambda _name: Reranker())
    monkeypatch.setattr(
        retrieve_module,
        "VectorStore",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("a second store must not be constructed")
        ),
    )
    store = SharedStore()

    hits = retrieve_module.retrieve(
        "soru",
        store_path="unused",
        collection="unused",
        embedder_name="embedder",
        reranker_name="reranker",
        top_k=20,
        top_n=3,
        store=store,
    )

    assert store.calls == 1
    assert [hit.chunk.id for hit in hits] == ["card::0001"]
