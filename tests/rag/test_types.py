from dataclasses import FrozenInstanceError

import pytest

from rag.types import Candidate, CandidatePacket, CardHint, Chunk, EvidenceSpan, Hit


def test_chunk_fields():
    c = Chunk(id="v1::0", video_id="v1", start=12.0, end=33.5, title="Test", text="merhaba")
    assert c.video_id == "v1"
    assert c.start == 12.0 and c.end == 33.5


def test_hit_carries_scores_and_chunk():
    c = Chunk(id="v1::0", video_id="v1", start=0.0, end=1.0, title="T", text="x")
    h = Hit(chunk=c, dense_score=0.8, sparse_score=0.3, rerank_score=2.1)
    assert h.chunk.id == "v1::0"
    assert h.rerank_score == 2.1


def test_evidence_contracts_are_frozen_and_keep_tuple_metadata():
    chunk = Chunk("v::0", "v", 1.0, 4.0, "Başlık", "temiz metin")
    candidate = Candidate(chunk, 1.25, ("original", "card::0001"))
    hint = CardHint("card::0001", "soru", "sentetik take", 0.5)
    packet = CandidatePacket((candidate,), (hint,), {"card_hints": "enabled"})
    span = EvidenceSpan(
        "span::v::1000::4000", "v", "Başlık", 1.0, 4.0, "temiz metin",
        1.25, ("v::0",), ("original", "card::0001"),
    )
    assert packet.candidates == (candidate,)
    assert packet.hint_hits == (hint,)
    assert span.source_chunk_ids == ("v::0",)
    with pytest.raises(FrozenInstanceError):
        candidate.retrieval_score = 2.0


def test_evidence_contracts_copy_mutable_container_inputs():
    chunk = Chunk("v::0", "v", 1.0, 4.0, "Başlık", "temiz metin")
    origins = ["original"]
    source_chunk_ids = ["v::0"]
    candidates = [Candidate(chunk, 1.25, origins)]
    hint_hits = [CardHint("card::0001", "soru", "sentetik take", 0.5)]
    diagnostics = {"card_hints": "enabled"}
    packet = CandidatePacket(candidates, hint_hits, diagnostics)
    span = EvidenceSpan(
        "span::v::1000::4000", "v", "Başlık", 1.0, 4.0, "temiz metin",
        1.25, source_chunk_ids, origins,
    )

    origins.append("card::0001")
    source_chunk_ids.append("v::1")
    candidates.clear()
    hint_hits.clear()
    diagnostics["card_hints"] = "disabled"

    assert packet.candidates == (Candidate(chunk, 1.25, ("original",)),)
    assert packet.hint_hits == (CardHint("card::0001", "soru", "sentetik take", 0.5),)
    assert packet.diagnostics == {"card_hints": "enabled"}
    assert span.source_chunk_ids == ("v::0",)
    assert span.query_origins == ("original",)
    with pytest.raises(TypeError):
        packet.diagnostics["card_hints"] = "disabled"
