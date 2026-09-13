"""Strict, dependency-light validation for optional card search hints."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = 1
DENSE_DIM = 1024
_CARD_ID = re.compile(r"card::[0-9]{4}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CARD_FIELDS = {
    "id", "q", "take", "n_takes", "alt_qs", "sources", "support", "n_strong", "date",
}
_SOURCE_FIELDS = {"video_id", "title", "start", "score"}
_MOJIBAKE_MARKERS = (
    "\u00c3",
    "\u00c2",
    "\u00c5",
    "\u00c4",
    "\u00e2\u20ac",
    "\u00f0\u0178",
    "\u00ef\u00bb\u00bf",
)
_MANIFEST_FIELDS = {
    "manifest_schema", "catalog_sha256", "card_count", "collection", "embedder",
    "dense_dim", "store_path", "index_fingerprint", "built_at",
}


@dataclass(frozen=True)
class CatalogCard:
    id: str
    question: str
    take: str


@dataclass(frozen=True)
class CardCatalog:
    sha256: str
    cards: tuple[CatalogCard, ...]

    @property
    def by_id(self) -> dict[str, CatalogCard]:
        return {card.id: card for card in self.cards}


@dataclass(frozen=True)
class CardHintStatus:
    usable: bool
    catalog: CardCatalog | None
    reason: str | None
    manifest_fingerprint: str | None = None


def canonical_sha256(value: object) -> str:
    normalized = _normalize(value)
    payload = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_card_catalog(path: Path) -> CardCatalog:
    raw = Path(path).read_bytes()
    value = _decode_strict_json(raw)
    if type(value) is not list or not value:
        raise ValueError("catalog must be a non-empty array")
    cards: list[CatalogCard] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if type(item) is not dict or set(item) != _CARD_FIELDS:
            raise ValueError(f"catalog item {index} fields do not match the canonical schema")
        card_id = _strict_text(item.get("id"), f"catalog item {index} id")
        question = _strict_text(item.get("q"), f"{card_id} question")
        take = _strict_text(item.get("take"), f"{card_id} take")
        if _CARD_ID.fullmatch(card_id) is None or card_id in seen:
            raise ValueError(f"invalid or duplicate card id {card_id}")
        seen.add(card_id)
        if type(item["n_takes"]) is not int or item["n_takes"] <= 0:
            raise ValueError(f"{card_id} n_takes must be positive")
        if type(item["n_strong"]) is not int or item["n_strong"] < 0:
            raise ValueError(f"{card_id} n_strong must be non-negative")
        if type(item["support"]) not in (int, float) or not math.isfinite(item["support"]):
            raise ValueError(f"{card_id} support must be finite")
        _strict_text(item["date"], f"{card_id} date")
        if type(item["alt_qs"]) is not list:
            raise ValueError(f"{card_id} alt_qs must be an array")
        for alt_index, alternate in enumerate(item["alt_qs"]):
            _strict_text(alternate, f"{card_id} alt_qs[{alt_index}]")
        if type(item["sources"]) is not list:
            raise ValueError(f"{card_id} sources must be an array")
        for source_index, source in enumerate(item["sources"]):
            if type(source) is not dict or set(source) != _SOURCE_FIELDS:
                raise ValueError(f"{card_id} source {source_index} fields are invalid")
            _strict_text(source["video_id"], f"{card_id} source video_id")
            _strict_text(source["title"], f"{card_id} source title")
            for field in ("start", "score"):
                if type(source[field]) not in (int, float) or not math.isfinite(source[field]):
                    raise ValueError(f"{card_id} source {field} must be finite")
            if source["start"] < 0:
                raise ValueError(f"{card_id} source start must be non-negative")
        cards.append(CatalogCard(card_id, question, take))
    return CardCatalog(hashlib.sha256(raw).hexdigest(), tuple(cards))


def validate_card_hint_manifest(
    *,
    catalog_path: Path,
    manifest_path: Path,
    collection: str,
    embedder: str,
    store_path: Path,
) -> CardHintStatus:
    catalog_path = Path(catalog_path)
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return CardHintStatus(False, None, "manifest_missing")
    try:
        catalog = load_card_catalog(catalog_path)
    except (OSError, UnicodeError, ValueError):
        return CardHintStatus(False, None, "catalog_invalid")
    try:
        value = _load_manifest(manifest_path)
    except (OSError, UnicodeError, ValueError):
        return CardHintStatus(False, None, "manifest_invalid")
    expected = {
        "manifest_schema": MANIFEST_SCHEMA,
        "catalog_sha256": catalog.sha256,
        "card_count": len(catalog.cards),
        "collection": collection,
        "embedder": embedder,
        "dense_dim": DENSE_DIM,
        "store_path": str(Path(store_path).resolve()),
    }
    reason_by_field = {
        "manifest_schema": "manifest_schema_mismatch",
        "catalog_sha256": "catalog_hash_mismatch",
        "card_count": "card_count_mismatch",
        "collection": "collection_mismatch",
        "embedder": "embedder_mismatch",
        "dense_dim": "dense_dim_mismatch",
    }
    for field, reason in reason_by_field.items():
        if value[field] != expected[field]:
            return CardHintStatus(False, None, reason)
    if _path_identity(value["store_path"]) != _path_identity(expected["store_path"]):
        return CardHintStatus(False, None, "store_path_mismatch")
    immutable = {key: value[key] for key in expected}
    if value["index_fingerprint"] != canonical_sha256(immutable):
        return CardHintStatus(False, None, "fingerprint_mismatch")
    return CardHintStatus(True, catalog, None, value["index_fingerprint"])


def relocate_card_manifest(
    *,
    catalog_path: Path,
    manifest_path: Path,
    destination_store_path: Path,
    collection: str,
    embedder: str,
) -> None:
    catalog = load_card_catalog(catalog_path)
    value = _load_manifest(manifest_path)
    immutable_before = {
        key: value[key]
        for key in (
            "manifest_schema", "catalog_sha256", "card_count", "collection", "embedder",
            "dense_dim", "store_path",
        )
    }
    if value["index_fingerprint"] != canonical_sha256(immutable_before):
        raise ValueError("cannot relocate a manifest with an invalid fingerprint")
    if value["catalog_sha256"] != catalog.sha256 or value["card_count"] != len(catalog.cards):
        raise ValueError("cannot relocate a manifest for a different catalog")
    if value["manifest_schema"] != MANIFEST_SCHEMA or value["dense_dim"] != DENSE_DIM:
        raise ValueError("cannot relocate a manifest with a different schema or dense dimension")
    if value["collection"] != collection or value["embedder"] != embedder:
        raise ValueError("cannot relocate a manifest for different index settings")
    value["store_path"] = str(Path(destination_store_path).resolve())
    immutable_after = {key: value[key] for key in immutable_before}
    value["index_fingerprint"] = canonical_sha256(immutable_after)
    _write_atomic_json(Path(manifest_path), value)


def _load_manifest(path: Path) -> dict[str, object]:
    value = _decode_strict_json(Path(path).read_bytes())
    if type(value) is not dict or set(value) != _MANIFEST_FIELDS:
        raise ValueError("manifest fields do not match schema 1")
    if type(value["manifest_schema"]) is not int:
        raise ValueError("manifest_schema must be an integer")
    if type(value["card_count"]) is not int or value["card_count"] <= 0:
        raise ValueError("card_count must be positive")
    if type(value["dense_dim"]) is not int or value["dense_dim"] <= 0:
        raise ValueError("dense_dim must be positive")
    for field in ("catalog_sha256", "collection", "embedder", "store_path", "index_fingerprint", "built_at"):
        _strict_text(value[field], field)
    if _SHA256.fullmatch(value["catalog_sha256"]) is None:
        raise ValueError("catalog_sha256 is invalid")
    if _SHA256.fullmatch(value["index_fingerprint"]) is None:
        raise ValueError("index_fingerprint is invalid")
    stamp = str(value["built_at"]).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(stamp)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("built_at must include a timezone")
    return value


def _decode_strict_json(raw: bytes) -> object:
    text = raw.decode("utf-8")
    return json.loads(
        text,
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=lambda value: (_raise(ValueError(f"non-finite JSON value {value}"))),
    )


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _strict_text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    if (
        not unicodedata.is_normalized("NFC", value)
        or "\ufffd" in value
        or any(marker in value for marker in _MOJIBAKE_MARKERS)
    ):
        raise ValueError(f"{label} is not clean NFC text")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{label} contains a surrogate")
    return value


def _normalize(value: object) -> object:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite value")
    return value


def _path_identity(value: object) -> str:
    return os.path.normcase(str(Path(str(value)).resolve()))


def _write_atomic_json(path: Path, value: object) -> None:
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


def _raise(error: Exception) -> None:
    raise error
