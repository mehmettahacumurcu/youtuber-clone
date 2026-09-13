import json
import math
import unicodedata
from pathlib import Path

import pytest

from rag.context import EvidenceSourceError, resolve_evidence_spans
from rag.types import Candidate, Chunk


def _write_clean(path: Path, video_id: str, segments: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{video_id}.json").write_text(
        json.dumps({"video_id": video_id, "title": "Başlık", "segments": segments}, ensure_ascii=False),
        encoding="utf-8",
    )


def _candidate(video="v1", start=10.0, end=20.0, chunk_id="v1::0", score=2.0):
    return Candidate(
        Chunk(chunk_id, video, start, end, "Başlık", "indexed chunk"), score, ("original",),
    )


def _scorer(question, texts):
    return [float(text.split(":", 1)[0]) for text in texts]


def test_anchor_expansion_uses_complete_segments_ties_earlier_and_stable_id(tmp_path):
    segments = [
        {"start": 0.0, "end": 5.0, "text": "1: sol iki"},
        {"start": 5.0, "end": 10.0, "text": "4: sol bir"},
        {"start": 10.0, "end": 15.0, "text": "9: çapa"},
        {"start": 15.0, "end": 20.0, "text": "4: sağ bir"},
        {"start": 20.0, "end": 25.0, "text": "1: sağ iki"},
    ]
    _write_clean(tmp_path, "v1", segments)
    spans = resolve_evidence_spans(
        "soru", (_candidate(),), tmp_path, score_segments=_scorer,
        max_spans=12, max_chars_per_span=200, max_seconds_per_span=180.0,
    )
    assert len(spans) == 1
    span = spans[0]
    assert span.id == "span::v1::0::25000"
    assert span.start == 0.0 and span.end == 25.0
    assert span.text == unicodedata.normalize("NFC", " ".join(s["text"] for s in segments))
    assert span.source_chunk_ids == ("v1::0",)


def test_anchor_tie_chooses_earliest_overlapping_segment(tmp_path):
    _write_clean(tmp_path, "v1", [
        {"start": 10.0, "end": 12.0, "text": "5: erken"},
        {"start": 12.0, "end": 14.0, "text": "5: geç"},
    ])
    spans = resolve_evidence_spans(
        "soru", (_candidate(start=10.0, end=14.0),), tmp_path,
        score_segments=_scorer, max_chars_per_span=8,
    )
    assert spans[0].text == "5: erken"
    assert spans[0].id == "span::v1::10000::12000"


def test_preferred_neighbor_that_exceeds_cap_stops_without_slicing(tmp_path):
    _write_clean(tmp_path, "v1", [
        {"start": 0.0, "end": 2.0, "text": "1: kısa sol"},
        {"start": 2.0, "end": 4.0, "text": "9: çapa"},
        {"start": 4.0, "end": 6.0, "text": "8: " + "x" * 80},
    ])
    spans = resolve_evidence_spans(
        "soru", (_candidate(start=2.0, end=4.0),), tmp_path,
        score_segments=_scorer, max_chars_per_span=30,
    )
    assert spans[0].text == "9: çapa"
    assert spans[0].start == 2.0 and spans[0].end == 4.0


def test_fifty_percent_overlap_keeps_higher_ranked_span_and_merges_metadata(tmp_path):
    _write_clean(tmp_path, "v1", [
        {"start": 0.0, "end": 10.0, "text": "9: bir"},
        {"start": 10.0, "end": 20.0, "text": "8: iki"},
        {"start": 20.0, "end": 30.0, "text": "7: üç"},
    ])
    first = _candidate(start=0.0, end=20.0, chunk_id="v1::0", score=3.0)
    second = Candidate(
        Chunk("v1::1", "v1", 10.0, 30.0, "Başlık", "chunk"), 2.0, ("card::0001",),
    )
    spans = resolve_evidence_spans(
        "soru", (first, second), tmp_path, score_segments=_scorer,
        max_chars_per_span=15,
    )
    assert len(spans) == 1
    assert spans[0].id == "span::v1::0::20000"
    assert spans[0].source_chunk_ids == ("v1::0", "v1::1")
    assert spans[0].query_origins == ("original", "card::0001")
    assert spans[0].retrieval_score == 3.0


@pytest.mark.parametrize(
    "candidate, message",
    [
        (_candidate(video="missing"), "missing clean source"),
        (_candidate(chunk_id="card::0001"), "card chunks cannot become evidence"),
    ],
)
def test_invalid_sources_raise_typed_error(tmp_path, candidate, message):
    if candidate.chunk.video_id != "missing":
        _write_clean(tmp_path, candidate.chunk.video_id, [
            {"start": 10.0, "end": 20.0, "text": "9: metin"},
        ])
    with pytest.raises(EvidenceSourceError, match=message):
        resolve_evidence_spans("soru", (candidate,), tmp_path, score_segments=_scorer)


def test_scorer_cardinality_and_nonfinite_values_are_rejected(tmp_path):
    _write_clean(tmp_path, "v1", [{"start": 10.0, "end": 20.0, "text": "metin"}])
    with pytest.raises(EvidenceSourceError, match="score count"):
        resolve_evidence_spans(
            "soru", (_candidate(),), tmp_path, score_segments=lambda question, texts: [],
        )
    with pytest.raises(EvidenceSourceError, match="non-finite"):
        resolve_evidence_spans(
            "soru", (_candidate(),), tmp_path,
            score_segments=lambda question, texts: [math.nan for _ in texts],
        )


def test_anchor_larger_than_cap_and_no_overlap_are_source_errors(tmp_path):
    _write_clean(tmp_path, "v1", [{"start": 0.0, "end": 10.0, "text": "9: " + "x" * 30}])
    with pytest.raises(EvidenceSourceError, match="anchor exceeds"):
        resolve_evidence_spans(
            "soru", (_candidate(start=0.0, end=10.0),), tmp_path,
            score_segments=_scorer, max_chars_per_span=10,
        )
    with pytest.raises(EvidenceSourceError, match="overlapping segment"):
        resolve_evidence_spans(
            "soru", (_candidate(start=20.0, end=21.0),), tmp_path,
            score_segments=_scorer,
        )


def test_anchor_larger_than_duration_cap_is_a_source_error(tmp_path):
    _write_clean(tmp_path, "v1", [{"start": 0.0, "end": 181.0, "text": "9: kısa"}])
    with pytest.raises(EvidenceSourceError, match="anchor exceeds"):
        resolve_evidence_spans(
            "soru", (_candidate(start=0.0, end=181.0),), tmp_path,
            score_segments=_scorer, max_chars_per_span=1800, max_seconds_per_span=180.0,
        )
