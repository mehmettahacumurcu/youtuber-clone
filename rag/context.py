"""Resolve clean retrieval candidates into exact, bounded transcript spans."""
from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from rag.types import Candidate, EvidenceSpan


class EvidenceSourceError(RuntimeError):
    """A clean source cannot satisfy the exact-span contract."""


@dataclass(frozen=True)
class _Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class _Draft:
    span: EvidenceSpan
    first_index: int
    last_index: int


def resolve_evidence_spans(
    question: str,
    candidates: Sequence[Candidate],
    clean_dir: Path,
    *,
    score_segments: Callable[[str, list[str]], Sequence[float]],
    max_spans: int = 12,
    max_chars_per_span: int = 1800,
    max_seconds_per_span: float = 180.0,
) -> tuple[EvidenceSpan, ...]:
    if max_spans <= 0 or max_chars_per_span <= 0 or max_seconds_per_span <= 0:
        raise EvidenceSourceError("span limits must be positive")
    clean_dir = Path(clean_dir)
    cache: dict[str, tuple[str, tuple[_Segment, ...]]] = {}
    score_cache: dict[tuple[str, int], float] = {}
    drafts: list[_Draft] = []
    for candidate in candidates:
        if candidate.chunk.id.startswith("card::"):
            raise EvidenceSourceError("card chunks cannot become evidence")
        video_id = candidate.chunk.video_id
        if video_id not in cache:
            cache[video_id] = _load_clean(clean_dir, video_id)
        title, segments = cache[video_id]
        draft = _resolve_one(
            question, candidate, title, segments, score_segments, score_cache,
            max_chars_per_span, max_seconds_per_span,
        )
        merged = False
        for index, existing in enumerate(drafts):
            if _overlaps_half(existing.span, draft.span):
                drafts[index] = _merge_metadata(existing, draft)
                merged = True
                break
        if not merged:
            drafts.append(draft)
        if len(drafts) >= max_spans:
            break
    return tuple(draft.span for draft in drafts)


def _load_clean(clean_dir: Path, video_id: str) -> tuple[str, tuple[_Segment, ...]]:
    path = clean_dir / f"{video_id}.json"
    if not path.is_file():
        raise EvidenceSourceError(f"missing clean source {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceSourceError(f"invalid clean source {path}: {exc}") from exc
    if type(value) is not dict or value.get("video_id") != video_id:
        raise EvidenceSourceError(f"clean source video_id mismatch for {video_id}")
    raw_segments = value.get("segments")
    if type(raw_segments) is not list or not raw_segments:
        raise EvidenceSourceError(f"clean source has no segments for {video_id}")
    segments: list[_Segment] = []
    previous_end = -math.inf
    for index, raw in enumerate(raw_segments):
        if type(raw) is not dict:
            raise EvidenceSourceError(f"segment {index} is not an object")
        start, end, text = raw.get("start"), raw.get("end"), raw.get("text")
        if type(start) not in (int, float) or type(end) not in (int, float):
            raise EvidenceSourceError(f"segment {index} has invalid timing")
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or start >= end:
            raise EvidenceSourceError(f"segment {index} has invalid timing")
        if start < previous_end:
            raise EvidenceSourceError(f"segment {index} is not chronological")
        if type(text) is not str or not text.strip():
            raise EvidenceSourceError(f"segment {index} has empty text")
        normalized = unicodedata.normalize("NFC", text.strip())
        segments.append(_Segment(start, end, normalized))
        previous_end = end
    return str(value.get("title") or video_id), tuple(segments)


def _resolve_one(
    question: str,
    candidate: Candidate,
    title: str,
    segments: tuple[_Segment, ...],
    scorer: Callable[[str, list[str]], Sequence[float]],
    score_cache: dict[tuple[str, int], float],
    char_cap: int,
    seconds_cap: float,
) -> _Draft:
    chunk = candidate.chunk
    if not all(math.isfinite(value) for value in (chunk.start, chunk.end, candidate.retrieval_score)):
        raise EvidenceSourceError(f"candidate {chunk.id} has non-finite values")
    if chunk.start >= chunk.end:
        raise EvidenceSourceError(f"candidate {chunk.id} has invalid timing")
    overlap = [
        index for index, segment in enumerate(segments)
        if segment.start < chunk.end and segment.end > chunk.start
    ]
    if not overlap:
        raise EvidenceSourceError(f"candidate {chunk.id} has no overlapping segment")
    _ensure_scores(question, chunk.video_id, overlap, segments, scorer, score_cache)
    anchor = max(overlap, key=lambda index: (score_cache[(chunk.video_id, index)], -index))
    anchor_text = _join(segments, anchor, anchor)
    anchor_duration = segments[anchor].end - segments[anchor].start
    if len(anchor_text) > char_cap or anchor_duration > seconds_cap:
        raise EvidenceSourceError(f"candidate {chunk.id} anchor exceeds span cap")
    left = right = anchor
    while left > 0 or right + 1 < len(segments):
        choices: list[int] = []
        if left > 0:
            choices.append(left - 1)
        if right + 1 < len(segments):
            choices.append(right + 1)
        _ensure_scores(question, chunk.video_id, choices, segments, scorer, score_cache)
        chosen = max(choices, key=lambda index: (score_cache[(chunk.video_id, index)], -index))
        next_left = chosen if chosen < left else left
        next_right = chosen if chosen > right else right
        text = _join(segments, next_left, next_right)
        duration = segments[next_right].end - segments[next_left].start
        if len(text) > char_cap or duration > seconds_cap:
            break
        left, right = next_left, next_right
    text = _join(segments, left, right)
    start, end = segments[left].start, segments[right].end
    span = EvidenceSpan(
        id=f"span::{chunk.video_id}::{int(round(start * 1000))}::{int(round(end * 1000))}",
        video_id=chunk.video_id,
        title=title or chunk.title,
        start=start,
        end=end,
        text=text,
        retrieval_score=candidate.retrieval_score,
        source_chunk_ids=(chunk.id,),
        query_origins=candidate.query_origins,
    )
    return _Draft(span, left, right)


def _ensure_scores(question, video_id, indices, segments, scorer, score_cache) -> None:
    unseen = [index for index in indices if (video_id, index) not in score_cache]
    if not unseen:
        return
    values = list(scorer(question, [segments[index].text for index in unseen]))
    if len(values) != len(unseen):
        raise EvidenceSourceError("segment score count differs from input count")
    for index, value in zip(unseen, values):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise EvidenceSourceError("segment scorer returned a non-finite value")
        score_cache[(video_id, index)] = numeric


def _join(segments: tuple[_Segment, ...], first: int, last: int) -> str:
    return unicodedata.normalize(
        "NFC", " ".join(segment.text.strip() for segment in segments[first:last + 1] if segment.text.strip()),
    )


def _overlaps_half(first: EvidenceSpan, second: EvidenceSpan) -> bool:
    if first.video_id != second.video_id:
        return False
    intersection = max(0.0, min(first.end, second.end) - max(first.start, second.start))
    shorter = min(first.end - first.start, second.end - second.start)
    return shorter > 0 and intersection / shorter >= 0.5


def _merge_metadata(first: _Draft, second: _Draft) -> _Draft:
    source_ids = tuple(dict.fromkeys((*first.span.source_chunk_ids, *second.span.source_chunk_ids)))
    origins = tuple(dict.fromkeys((*first.span.query_origins, *second.span.query_origins)))
    return replace(
        first,
        span=replace(
            first.span,
            retrieval_score=max(first.span.retrieval_score, second.span.retrieval_score),
            source_chunk_ids=source_ids,
            query_origins=origins,
        ),
    )
