import hashlib
import json
from pathlib import Path

import pytest

from rag.card_manifest import (
    DENSE_DIM,
    MANIFEST_SCHEMA,
    canonical_sha256,
    relocate_card_manifest,
    validate_card_hint_manifest,
)


def _write_catalog(path: Path) -> str:
    value = [{
        "id": "card::0001",
        "q": "Soru?",
        "take": "Sentetik arama ipucu.",
        "n_takes": 1,
        "alt_qs": [],
        "sources": [],
        "support": 1.0,
        "n_strong": 1,
        "date": "20260713",
    }]
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, catalog_sha: str, store: Path, **updates: object) -> dict:
    immutable = {
        "manifest_schema": MANIFEST_SCHEMA,
        "catalog_sha256": catalog_sha,
        "card_count": 1,
        "collection": "speaker_cards",
        "embedder": "BAAI/bge-m3",
        "dense_dim": DENSE_DIM,
        "store_path": str(store.resolve()),
    }
    immutable.update(updates)
    value = {
        **immutable,
        "index_fingerprint": canonical_sha256(immutable),
        "built_at": "2026-07-13T10:00:00Z",
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return value


def _validate(tmp_path: Path, **manifest_updates: object):
    catalog = tmp_path / "cards.json"
    manifest = tmp_path / "manifest.json"
    store = tmp_path / "qdrant"
    sha = _write_catalog(catalog)
    _write_manifest(manifest, sha, store, **manifest_updates)
    return validate_card_hint_manifest(
        catalog_path=catalog,
        manifest_path=manifest,
        collection="speaker_cards",
        embedder="BAAI/bge-m3",
        store_path=store,
    )


def test_valid_manifest_returns_canonical_catalog(tmp_path):
    status = _validate(tmp_path)
    assert status.usable
    assert status.reason is None
    assert status.catalog is not None
    assert status.catalog.by_id["card::0001"].take == "Sentetik arama ipucu."


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("manifest_schema", 2, "manifest_schema_mismatch"),
        ("catalog_sha256", "0" * 64, "catalog_hash_mismatch"),
        ("card_count", 2, "card_count_mismatch"),
        ("collection", "wrong", "collection_mismatch"),
        ("embedder", "wrong", "embedder_mismatch"),
        ("dense_dim", 7, "dense_dim_mismatch"),
    ],
)
def test_manifest_mismatch_degrades_to_typed_unusable(tmp_path, field, value, reason):
    status = _validate(tmp_path, **{field: value})
    assert not status.usable
    assert status.catalog is None
    assert status.reason == reason


def test_missing_and_malformed_manifest_are_unusable(tmp_path):
    catalog = tmp_path / "cards.json"
    _write_catalog(catalog)
    missing = validate_card_hint_manifest(
        catalog_path=catalog,
        manifest_path=tmp_path / "missing.json",
        collection="speaker_cards",
        embedder="BAAI/bge-m3",
        store_path=tmp_path / "qdrant",
    )
    assert missing.reason == "manifest_missing"
    bad = tmp_path / "bad.json"
    bad.write_text('{"manifest_schema": NaN}', encoding="utf-8")
    malformed = validate_card_hint_manifest(
        catalog_path=catalog,
        manifest_path=bad,
        collection="speaker_cards",
        embedder="BAAI/bge-m3",
        store_path=tmp_path / "qdrant",
    )
    assert malformed.reason == "manifest_invalid"


def test_catalog_must_match_the_full_canonical_card_schema(tmp_path):
    catalog = tmp_path / "cards.json"
    catalog.write_text(
        json.dumps([{"id": "card::0001", "q": "Soru?", "take": "take"}]),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, hashlib.sha256(catalog.read_bytes()).hexdigest(), tmp_path / "qdrant")
    status = validate_card_hint_manifest(
        catalog_path=catalog,
        manifest_path=manifest,
        collection="speaker_cards",
        embedder="BAAI/bge-m3",
        store_path=tmp_path / "qdrant",
    )
    assert status.reason == "catalog_invalid"


def test_catalog_rejects_lone_producer_mojibake_marker(tmp_path):
    catalog = tmp_path / "cards.json"
    _write_catalog(catalog)
    value = json.loads(catalog.read_text(encoding="utf-8"))
    value[0]["take"] = f"Lone marker: {chr(0x00C3)}"
    catalog.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, hashlib.sha256(catalog.read_bytes()).hexdigest(), tmp_path / "qdrant")

    status = validate_card_hint_manifest(
        catalog_path=catalog,
        manifest_path=manifest,
        collection="speaker_cards",
        embedder="BAAI/bge-m3",
        store_path=tmp_path / "qdrant",
    )

    assert status.reason == "catalog_invalid"


def test_bad_fingerprint_and_store_path_are_unusable(tmp_path):
    catalog = tmp_path / "cards.json"
    manifest = tmp_path / "manifest.json"
    sha = _write_catalog(catalog)
    value = _write_manifest(manifest, sha, tmp_path / "qdrant")
    value["index_fingerprint"] = "0" * 64
    manifest.write_text(json.dumps(value), encoding="utf-8")
    bad_fingerprint = validate_card_hint_manifest(
        catalog_path=catalog,
        manifest_path=manifest,
        collection="speaker_cards",
        embedder="BAAI/bge-m3",
        store_path=tmp_path / "qdrant",
    )
    assert bad_fingerprint.reason == "fingerprint_mismatch"

    _write_manifest(manifest, sha, tmp_path / "other")
    mismatch = validate_card_hint_manifest(
        catalog_path=catalog,
        manifest_path=manifest,
        collection="speaker_cards",
        embedder="BAAI/bge-m3",
        store_path=tmp_path / "expected",
    )
    assert mismatch.reason == "store_path_mismatch"


def test_relocation_rewrites_only_store_identity_and_fingerprint(tmp_path):
    catalog = tmp_path / "cards.json"
    manifest = tmp_path / "manifest.json"
    sha = _write_catalog(catalog)
    before = _write_manifest(manifest, sha, tmp_path / "source-store")
    relocate_card_manifest(
        catalog_path=catalog,
        manifest_path=manifest,
        destination_store_path=tmp_path / "dest-store",
        collection="speaker_cards",
        embedder="BAAI/bge-m3",
    )
    after = json.loads(manifest.read_text(encoding="utf-8"))
    assert after["store_path"] == str((tmp_path / "dest-store").resolve())
    assert after["built_at"] == before["built_at"]
    assert after["catalog_sha256"] == before["catalog_sha256"]
    assert after["index_fingerprint"] != before["index_fingerprint"]
