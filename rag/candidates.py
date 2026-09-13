"""Broad clean retrieval with optional, isolated card search hints."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from pipeline.config import RagConfig
from rag.card_manifest import CardHintStatus, validate_card_hint_manifest
from rag.retrieve import retrieve
from rag.types import Candidate, CandidatePacket, CardHint, Hit


class CandidateRetrievalError(RuntimeError):
    """Original clean evidence retrieval failed."""


def collect_candidates(
    question: str,
    settings: RagConfig,
    *,
    retrieve_fn: Callable[..., list[Hit]] = retrieve,
    card_manifest_validator: Callable[..., CardHintStatus] = validate_card_hint_manifest,
) -> CandidatePacket:
    question = question.strip()
    if not question:
        raise CandidateRetrievalError("question must be non-empty")
    common = {
        "store_path": str(settings.store_path),
        "embedder_name": settings.embedder,
        "reranker_name": settings.reranker,
        "top_k": settings.retrieve_top_k,
        "top_n": settings.retrieve_top_k,
        "max_per_video": None,
        "window_chars": None,
    }
    try:
        original_hits = retrieve_fn(question, collection=settings.evidence_collection, **common)
    except Exception as exc:
        raise CandidateRetrievalError(f"original clean retrieval failed: {exc}") from exc

    original = _dedupe_hits(original_hits, "original")
    diagnostics: dict[str, object] = {
        "original_hit_count": len(original_hits),
        "original_unique_count": len(original),
        "card_hint_status": "disabled" if settings.retrieval_mode == "clean" else "unchecked",
    }
    if settings.retrieval_mode == "clean":
        return CandidatePacket(
            tuple(original[: settings.verifier_candidate_max]), (), diagnostics,
        )

    try:
        status = card_manifest_validator(
            catalog_path=settings.card_catalog_path,
            manifest_path=settings.card_manifest_path,
            collection=settings.card_collection,
            embedder=settings.embedder,
            store_path=settings.store_path,
        )
        if not status.usable or status.catalog is None:
            diagnostics["card_hint_status"] = status.reason or "invalid"
            return CandidatePacket(
                tuple(original[: settings.verifier_candidate_max]), (), diagnostics,
            )

        by_id = status.catalog.by_id
        card_ids = set(by_id)
        original = [candidate for candidate in original if candidate.chunk.id not in card_ids]
        card_hits = retrieve_fn(
            question,
            collection=settings.card_collection,
            **{**common, "top_n": settings.card_hint_top_n},
        )
        eligible: list[tuple[Hit, object]] = []
        for hit in card_hits[: settings.card_hint_top_n]:
            card = by_id.get(hit.chunk.id)
            if card is None:
                raise ValueError(f"card index returned unknown id {hit.chunk.id}")
            if hit.rerank_score >= settings.card_hint_min_score:
                eligible.append((hit, card))

        base = original[: settings.original_candidate_n]
        ordered = {candidate.chunk.id: candidate for candidate in base}
        order = [candidate.chunk.id for candidate in base]
        hints: list[CardHint] = []
        for card_hit, card in eligible:
            origin = card.id
            hints.append(CardHint(card.id, card.question, card.take, card_hit.rerank_score))
            hint_query = f"{question}\n\n{card.take[:800]}"
            hint_hits = retrieve_fn(
                hint_query, collection=settings.evidence_collection, **common,
            )
            added = 0
            for hit in hint_hits:
                chunk_id = hit.chunk.id
                if chunk_id in card_ids:
                    continue
                if chunk_id in ordered:
                    ordered[chunk_id] = _merge_candidate(ordered[chunk_id], hit, origin)
                    continue
                if added >= settings.hint_candidate_n:
                    continue
                ordered[chunk_id] = Candidate(hit.chunk, hit.rerank_score, (origin,))
                order.append(chunk_id)
                added += 1

        for candidate in original:
            if len(order) >= settings.verifier_candidate_max:
                break
            if candidate.chunk.id in ordered:
                ordered[candidate.chunk.id] = _merge_candidate(
                    ordered[candidate.chunk.id],
                    Hit(candidate.chunk, rerank_score=candidate.retrieval_score),
                    "original",
                )
                continue
            ordered[candidate.chunk.id] = candidate
            order.append(candidate.chunk.id)
        diagnostics["card_hint_status"] = "enabled"
        diagnostics["card_manifest_fingerprint"] = status.manifest_fingerprint
        diagnostics["eligible_hint_count"] = len(hints)
        return CandidatePacket(
            tuple(ordered[chunk_id] for chunk_id in order[: settings.verifier_candidate_max]),
            tuple(hints),
            diagnostics,
        )
    except Exception as exc:
        diagnostics["card_hint_status"] = "retrieval_error"
        diagnostics["card_hint_error"] = f"{type(exc).__name__}: {exc}"
        return CandidatePacket(
            tuple(original[: settings.verifier_candidate_max]), (), diagnostics,
        )


def _dedupe_hits(hits: list[Hit], origin: str) -> list[Candidate]:
    ordered: dict[str, Candidate] = {}
    for hit in hits:
        if hit.chunk.id in ordered:
            ordered[hit.chunk.id] = _merge_candidate(ordered[hit.chunk.id], hit, origin)
        else:
            ordered[hit.chunk.id] = Candidate(hit.chunk, hit.rerank_score, (origin,))
    return list(ordered.values())


def _merge_candidate(candidate: Candidate, hit: Hit, origin: str) -> Candidate:
    origins = candidate.query_origins
    if origin not in origins:
        origins = (*origins, origin)
    return replace(
        candidate,
        retrieval_score=max(candidate.retrieval_score, hit.rerank_score),
        query_origins=origins,
    )
