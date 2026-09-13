"""Build the immutable, fingerprinted ``speaker_cards`` Qdrant collection.

Run with every process using the local Qdrant directory stopped.  Rebuilding is
destructive, so the previous manifest is removed before the collection is
recreated and a new manifest is published only after every upsert succeeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from eval.card_eval_schema import canonical_sha256
from pipeline.config import load_settings
from rag.embed import get_embedder
from rag.store import VectorStore
from rag.types import Chunk


DEFAULT_CARDS = Path("data/cards/speaker_cards.json")
DEFAULT_COLLECTION = "speaker_cards"
DEFAULT_MANIFEST = Path("data/cards/speaker_cards.index_manifest.json")
DENSE_DIM = 1024
MANIFEST_SCHEMA = 1

_CARD_FIELDS = {
    "id",
    "q",
    "take",
    "n_takes",
    "alt_qs",
    "sources",
    "support",
    "n_strong",
    "date",
}
_SOURCE_FIELDS = {"video_id", "title", "start", "score"}
_CARD_ID_RE = re.compile(r"card::[0-9]{4}\Z")
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
class CatalogSource:
    video_id: str
    title: str
    start: float
    score: float


@dataclass(frozen=True)
class CatalogCard:
    id: str
    q: str
    take: str
    sources: tuple[CatalogSource, ...]


@dataclass(frozen=True)
class CardCatalog:
    sha256: str
    cards: tuple[CatalogCard, ...]

    @property
    def by_id(self) -> dict[str, CatalogCard]:
        return {card.id: card for card in self.cards}


def load_card_catalog(path: Path) -> CardCatalog:
    """Load and strictly validate the canonical card catalog."""

    path = Path(path)
    raw_bytes = path.read_bytes()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"card catalog {path} is not valid UTF-8") from exc
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"card catalog {path} is not strict JSON: {exc}") from exc
    if type(raw) is not list or not raw:
        raise ValueError("card catalog must be a non-empty JSON array")

    cards: list[CatalogCard] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(raw):
        label = f"card catalog entry {index}"
        if type(value) is not dict:
            raise ValueError(f"{label} must be an object")
        if set(value) != _CARD_FIELDS:
            raise ValueError(f"{label} has missing or unexpected fields")

        card_id = _strict_text(value["id"], f"{label} id")
        if _CARD_ID_RE.fullmatch(card_id) is None:
            raise ValueError(f"{label} has an invalid card ID")
        if card_id in seen_ids:
            raise ValueError(f"duplicate card ID {card_id}")
        seen_ids.add(card_id)
        question = _strict_text(value["q"], f"{card_id} question")
        take = _strict_text(value["take"], f"{card_id} take")
        _strict_positive_int(value["n_takes"], f"{card_id} n_takes")
        _strict_nonnegative_int(value["n_strong"], f"{card_id} n_strong")
        _strict_finite(value["support"], f"{card_id} support")
        _strict_text(value["date"], f"{card_id} date")

        alt_questions = value["alt_qs"]
        if type(alt_questions) is not list:
            raise ValueError(f"{card_id} alt_qs must be an array")
        for alt_index, alternate in enumerate(alt_questions):
            _strict_text(alternate, f"{card_id} alt_qs[{alt_index}]")

        source_values = value["sources"]
        if type(source_values) is not list:
            raise ValueError(f"{card_id} sources must be an array")
        sources: list[CatalogSource] = []
        for source_index, source in enumerate(source_values):
            source_label = f"{card_id} source {source_index}"
            if type(source) is not dict or set(source) != _SOURCE_FIELDS:
                raise ValueError(
                    f"{source_label} has missing or unexpected fields"
                )
            start = _strict_finite(source["start"], f"{source_label} start")
            if start < 0:
                raise ValueError(f"{source_label} start must be non-negative")
            sources.append(
                CatalogSource(
                    video_id=_strict_text(
                        source["video_id"], f"{source_label} video_id"
                    ),
                    title=_strict_text(source["title"], f"{source_label} title"),
                    start=start,
                    score=_strict_finite(source["score"], f"{source_label} score"),
                )
            )
        cards.append(
            CatalogCard(
                id=card_id,
                q=question,
                take=take,
                sources=tuple(sources),
            )
        )
    return CardCatalog(
        sha256=hashlib.sha256(raw_bytes).hexdigest(), cards=tuple(cards)
    )


def build_card_index(
    cards_path: Path, collection: str, manifest_path: Path
) -> dict[str, object]:
    """Destructively rebuild one card collection and publish its manifest."""

    cards_path = Path(cards_path)
    manifest_path = Path(manifest_path)
    if type(collection) is not str or not collection.strip():
        raise ValueError("collection must be a non-empty string")
    catalog = load_card_catalog(cards_path)
    rag_settings = load_settings().rag
    store_path = str(rag_settings.store_path)

    chunks: list[Chunk] = []
    embed_texts: list[str] = []
    for card in catalog.cards:
        source = card.sources[0] if card.sources else None
        chunks.append(
            Chunk(
                id=card.id,
                video_id=(
                    source.video_id if source is not None else f"nocite::{card.id}"
                ),
                start=source.start if source is not None else 0.0,
                end=0.0,
                title=source.title if source is not None else card.q[:80],
                text=card.take,
            )
        )
        embed_texts.append(f"{card.q}\n{card.take}")

    embedder = get_embedder(rag_settings.embedder)
    # From this point a failure must leave no manifest claiming the old index is current.
    if manifest_path.exists():
        manifest_path.unlink()
    store = VectorStore(
        path=store_path, collection=collection, dense_dim=DENSE_DIM
    )
    store.ensure_collection()
    for start in range(0, len(chunks), 64):
        part = chunks[start : start + 64]
        output = embedder.encode(embed_texts[start : start + 64])
        dense, sparse = _validate_embedding_batch(output, len(part))
        store.upsert_chunks(
            part,
            dense,
            sparse,
        )

    immutable_manifest: dict[str, object] = {
        "manifest_schema": MANIFEST_SCHEMA,
        "catalog_sha256": catalog.sha256,
        "card_count": len(catalog.cards),
        "collection": collection,
        "embedder": rag_settings.embedder,
        "dense_dim": DENSE_DIM,
        "store_path": store_path,
    }
    manifest = {
        **immutable_manifest,
        "index_fingerprint": canonical_sha256(immutable_manifest),
        "built_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _write_atomic_json(manifest_path, manifest)
    return manifest


def _validate_embedding_batch(
    output: object, expected_count: int
) -> tuple[np.ndarray, list[dict[int, float]]]:
    """Reject any batch that could make zip-based upsert silently truncate."""

    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("embedding batch expected_count must be positive")
    try:
        dense = np.asarray(output.dense)
        sparse_value = output.sparse
    except AttributeError as exc:
        raise ValueError("embedding output is missing dense or sparse values") from exc
    if dense.shape != (expected_count, DENSE_DIM):
        raise ValueError(
            "embedding dense shape must be "
            f"({expected_count}, {DENSE_DIM}), got {dense.shape}"
        )
    if not np.issubdtype(dense.dtype, np.number) or not np.isfinite(dense).all():
        raise ValueError("embedding dense values must be finite numbers")
    if type(sparse_value) is not list or len(sparse_value) != expected_count:
        raise ValueError("embedding sparse cardinality differs from the batch")
    sparse: list[dict[int, float]] = []
    for row_index, row in enumerate(sparse_value):
        if not isinstance(row, Mapping):
            raise ValueError(f"embedding sparse row {row_index} must be a mapping")
        normalized: dict[int, float] = {}
        for key, value in row.items():
            if isinstance(key, bool) or not isinstance(key, (int, np.integer)) or key < 0:
                raise ValueError(
                    f"embedding sparse row {row_index} has an invalid index"
                )
            if isinstance(value, (bool, str)):
                raise ValueError(
                    f"embedding sparse row {row_index} has a non-finite value"
                )
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"embedding sparse row {row_index} has a non-finite value"
                ) from exc
            if not math.isfinite(numeric):
                raise ValueError(
                    f"embedding sparse row {row_index} has a non-finite value"
                )
            normalized[int(key)] = numeric
        sparse.append(normalized)
    return dense, sparse


def _write_atomic_json(path: Path, value: object) -> None:
    path = Path(path)
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


def _strict_finite(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _strict_positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _strict_nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, default=DEFAULT_CARDS)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = build_card_index(args.cards, args.collection, args.manifest)
    print(
        f"Indexed {manifest['card_count']} cards into {manifest['collection']}; "
        f"fingerprint {manifest['index_fingerprint']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
