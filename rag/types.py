"""Shared RAG data contracts."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class Chunk:
    id: str
    video_id: str
    start: float
    end: float
    title: str
    text: str


@dataclass
class Hit:
    chunk: Chunk
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0


@dataclass(frozen=True)
class Candidate:
    chunk: Chunk
    retrieval_score: float
    query_origins: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_origins", tuple(self.query_origins))


@dataclass(frozen=True)
class CardHint:
    card_id: str
    question: str
    take: str
    retrieval_score: float


@dataclass(frozen=True)
class CandidatePacket:
    candidates: tuple[Candidate, ...]
    hint_hits: tuple[CardHint, ...]
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "hint_hits", tuple(self.hint_hits))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True)
class EvidenceSpan:
    id: str
    video_id: str
    title: str
    start: float
    end: float
    text: str
    retrieval_score: float
    source_chunk_ids: tuple[str, ...]
    query_origins: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_chunk_ids", tuple(self.source_chunk_ids))
        object.__setattr__(self, "query_origins", tuple(self.query_origins))
