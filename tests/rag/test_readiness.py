from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag.readiness import approved_identity, default_store_probe


def _settings(tmp_path: Path):
    rag = SimpleNamespace(
        store_path=tmp_path / "qdrant", evidence_collection="speaker_clean",
        card_collection="speaker_cards", card_catalog_path=tmp_path / "cards.json",
        card_manifest_path=tmp_path / "cards.manifest.json", embedder="BAAI/bge-m3", reranker="reranker",
    )
    return SimpleNamespace(rag=rag)


def test_readiness_rejects_missing_clean_manifest_before_opening_store(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        approved_identity(_settings(tmp_path), card_validator=lambda **_kwargs: None, store_probe=lambda *_args: {})


def test_readiness_returns_only_manifest_identities_and_counts(tmp_path: Path):
    manifest = {
        "schema": "youtuber.rag.v1", "corpus_version": "2026.08.01",
        "corpus_fingerprint": "a" * 64, "index_fingerprint": "b" * 64,
        "clean_collection": "speaker_clean", "clean_count": 3,
    }
    path = tmp_path / "rebuild_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    cards = SimpleNamespace(
        usable=True, manifest_fingerprint="c" * 64,
        catalog=SimpleNamespace(cards=(object(), object())),
    )

    identity = approved_identity(
        _settings(tmp_path), clean_manifest_path=path, card_validator=lambda **_kwargs: cards,
        store_probe=lambda *_args: {"speaker_clean": 3, "speaker_cards": 2},
    )

    assert identity == {
        "corpus_version": "2026.08.01", "index_version": "2026.08.01",
        "corpus_fingerprint": "a" * 64, "index_fingerprint": "b" * 64,
        "card_fingerprint": "c" * 64, "clean_count": 3, "card_count": 2,
        "embedder": "BAAI/bge-m3", "reranker": "reranker",
    }


def test_readiness_rejects_exact_store_count_mismatch(tmp_path: Path):
    manifest = {
        "schema": "youtuber.rag.v1", "corpus_version": "2026.08.01",
        "corpus_fingerprint": "a" * 64, "index_fingerprint": "b" * 64,
        "clean_collection": "speaker_clean", "clean_count": 3,
    }
    path = tmp_path / "rebuild_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    cards = SimpleNamespace(usable=True, manifest_fingerprint="c" * 64, catalog=SimpleNamespace(cards=(object(), object())))

    with pytest.raises(RuntimeError, match="count differs"):
        approved_identity(
            _settings(tmp_path), clean_manifest_path=path, card_validator=lambda **_kwargs: cards,
            store_probe=lambda *_args: {"speaker_clean": 2, "speaker_cards": 2},
        )


def test_default_store_probe_closes_its_qdrant_reader(monkeypatch, tmp_path: Path):
    events = []

    class Client:
        def __init__(self, *, path):
            events.append(("open", path))

        def collection_exists(self, collection):
            return True

        def count(self, collection, *, exact):
            return SimpleNamespace(count=3 if collection == "clean" else 2)

        def close(self):
            events.append(("close",))

    monkeypatch.setitem(sys.modules, "qdrant_client", SimpleNamespace(QdrantClient=Client))

    counts = default_store_probe(tmp_path / "qdrant", "clean", "cards")

    assert counts == {"clean": 3, "cards": 2}
    assert events[-1] == ("close",)
