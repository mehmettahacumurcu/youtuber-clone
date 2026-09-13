"""Strict, non-content readiness checks for the approved RAG artifact pair."""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path

from pipeline.config import Settings
from rag.card_manifest import validate_card_hint_manifest


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = {"schema", "corpus_version", "corpus_fingerprint", "index_fingerprint", "clean_collection", "clean_count"}


def load_clean_manifest(path: Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(value) is not dict or set(value) != _FIELDS or value["schema"] != "youtuber.rag.v1":
        raise ValueError("clean RAG manifest is invalid")
    if type(value["clean_count"]) is not int or value["clean_count"] <= 0:
        raise ValueError("clean RAG manifest count is invalid")
    for key in ("corpus_version", "clean_collection"):
        if type(value[key]) is not str or not value[key]:
            raise ValueError("clean RAG manifest text is invalid")
    for key in ("corpus_fingerprint", "index_fingerprint"):
        if type(value[key]) is not str or _SHA256.fullmatch(value[key]) is None:
            raise ValueError("clean RAG manifest fingerprint is invalid")
    return value


def default_store_probe(path: Path, clean_collection: str, card_collection: str) -> Mapping[str, int]:
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(path))
    try:
        result: dict[str, int] = {}
        for collection in (clean_collection, card_collection):
            if not client.collection_exists(collection):
                raise RuntimeError("approved RAG collection is missing")
            count = client.count(collection, exact=True).count
            if not isinstance(count, int) or count <= 0:
                raise RuntimeError("approved RAG collection is empty")
            result[collection] = count
        return result
    finally:
        client.close()


def approved_identity(
    settings: Settings,
    *,
    clean_manifest_path: Path | None = None,
    card_validator: Callable[..., object] = validate_card_hint_manifest,
    store_probe: Callable[[Path, str, str], Mapping[str, int]] | None = None,
) -> dict[str, object]:
    """Validate manifests and open both collections without reading corpus text."""
    rag = settings.rag
    clean = load_clean_manifest(clean_manifest_path or (Path(rag.store_path).parent / "rebuild_manifest.json"))
    if clean["clean_collection"] != rag.evidence_collection:
        raise RuntimeError("clean RAG collection differs from approved manifest")
    cards = card_validator(
        catalog_path=rag.card_catalog_path, manifest_path=rag.card_manifest_path,
        collection=rag.card_collection, embedder=rag.embedder, store_path=rag.store_path,
    )
    if not getattr(cards, "usable", False) or not getattr(cards, "manifest_fingerprint", None):
        raise RuntimeError("card RAG manifest is unusable")
    counts = (store_probe or default_store_probe)(Path(rag.store_path), rag.evidence_collection, rag.card_collection)
    clean_count = counts.get(rag.evidence_collection, 0)
    card_count = counts.get(rag.card_collection, 0)
    catalog = getattr(cards, "catalog", None)
    expected_card_count = len(getattr(catalog, "cards", ()))
    if clean_count <= 0 or card_count <= 0 or expected_card_count <= 0:
        raise RuntimeError("approved RAG collection is empty")
    if clean_count != clean["clean_count"] or card_count != expected_card_count:
        raise RuntimeError("approved RAG collection count differs from manifest")
    return {
        "corpus_version": clean["corpus_version"],
        "index_version": clean["corpus_version"],
        "corpus_fingerprint": clean["corpus_fingerprint"],
        "index_fingerprint": clean["index_fingerprint"],
        "card_fingerprint": cards.manifest_fingerprint,
        "clean_count": clean_count,
        "card_count": card_count,
        "embedder": rag.embedder, "reranker": rag.reranker,
    }
