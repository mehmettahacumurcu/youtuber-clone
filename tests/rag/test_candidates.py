import pytest

from pipeline.config import RagConfig
from rag.candidates import CandidateRetrievalError, collect_candidates
from rag.card_manifest import CardCatalog, CardHintStatus, CatalogCard
from rag.types import Chunk, Hit


def _hit(chunk_id: str, score: float, text: str | None = None) -> Hit:
    return Hit(
        Chunk(chunk_id, chunk_id.split("::")[0], 1.0, 2.0, chunk_id, text or chunk_id),
        rerank_score=score,
    )


class RecordingRetriever:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def __call__(self, query: str, **kwargs):
        self.calls.append((query, kwargs))
        value = self.values[(kwargs["collection"], query)]
        if isinstance(value, Exception):
            raise value
        return value


def _settings(mode="clean_with_card_hints") -> RagConfig:
    return RagConfig(retrieval_mode=mode)


def _valid_status() -> CardHintStatus:
    catalog = CardCatalog(
        "0" * 64,
        (
            CatalogCard("card::0001", "kart sorusu", "A" * 900),
            CatalogCard("card::0002", "ikinci kart", "ikinci take"),
        ),
    )
    return CardHintStatus(True, catalog, None, "1" * 64)


def test_clean_mode_returns_twelve_unique_originals_and_never_touches_cards():
    original = [_hit(f"v{i}::0", 20 - i) for i in range(13)]
    retriever = RecordingRetriever({("speaker_clean", "soru"): original})
    validator_calls = []
    packet = collect_candidates(
        "soru",
        _settings("clean"),
        retrieve_fn=retriever,
        card_manifest_validator=lambda **kwargs: validator_calls.append(kwargs),
    )
    assert [c.chunk.id for c in packet.candidates] == [f"v{i}::0" for i in range(12)]
    assert packet.hint_hits == ()
    assert validator_calls == []
    assert retriever.calls[0][1]["top_k"] == 20
    assert retriever.calls[0][1]["top_n"] == 20
    assert retriever.calls[0][1]["max_per_video"] is None
    assert retriever.calls[0][1]["window_chars"] is None


def test_invalid_manifest_degrades_to_clean_only_with_reason():
    original = [_hit(f"v{i}::0", 20 - i) for i in range(12)]
    retriever = RecordingRetriever({("speaker_clean", "soru"): original})
    packet = collect_candidates(
        "soru",
        _settings(),
        retrieve_fn=retriever,
        card_manifest_validator=lambda **kwargs: CardHintStatus(False, None, "manifest_missing"),
    )
    assert len(retriever.calls) == 1
    assert len(packet.candidates) == 12
    assert packet.diagnostics["card_hint_status"] == "manifest_missing"


def test_hint_quota_origins_backfill_and_take_cap_are_exact():
    original = [_hit(f"o{i}::0", 20 - i) for i in range(12)]
    hint_one_query = "soru\n\n" + "A" * 800
    hint_two_query = "soru\n\nikinci take"
    retriever = RecordingRetriever({
        ("speaker_clean", "soru"): original,
        ("speaker_cards", "soru"): [
            _hit("card::0001", -1.0, "untrusted-index-text"),
            _hit("card::0002", 0.5, "other-untrusted-text"),
        ],
        ("speaker_clean", hint_one_query): [
            _hit("o0::0", 99.0), _hit("h1::0", 8.0), _hit("h2::0", 7.0), _hit("h3::0", 6.0),
        ],
        ("speaker_clean", hint_two_query): [
            _hit("h2::0", 10.0), _hit("h4::0", 9.0), _hit("h5::0", 8.0),
        ],
    })
    packet = collect_candidates(
        "soru", _settings(), retrieve_fn=retriever,
        card_manifest_validator=lambda **kwargs: _valid_status(),
    )
    ids = [candidate.chunk.id for candidate in packet.candidates]
    assert ids[:8] == [f"o{i}::0" for i in range(8)]
    assert ids == [f"o{i}::0" for i in range(8)] + ["h1::0", "h2::0", "h4::0", "h5::0"]
    assert packet.candidates[0].retrieval_score == 99.0
    assert packet.candidates[0].query_origins == ("original", "card::0001")
    assert packet.candidates[9].query_origins == ("card::0001", "card::0002")
    assert packet.hint_hits[0].take == "A" * 900
    assert all(candidate.chunk.id not in {"card::0001", "card::0002"} for candidate in packet.candidates)
    assert hint_one_query in [query for query, _ in retriever.calls]


def test_secondary_clean_response_never_promotes_a_catalog_card_to_evidence():
    original = [_hit(f"o{i}::0", 20 - i) for i in range(12)]
    hint_query = "soru\n\n" + "A" * 800
    retriever = RecordingRetriever({
        ("speaker_clean", "soru"): original,
        ("speaker_cards", "soru"): [_hit("card::0001", 0.0)],
        ("speaker_clean", hint_query): [
            _hit("card::0001", 99.0, "stale card index text"),
            _hit("h1::0", 8.0),
            _hit("h2::0", 7.0),
        ],
    })
    packet = collect_candidates(
        "soru", _settings(), retrieve_fn=retriever,
        card_manifest_validator=lambda **kwargs: _valid_status(),
    )
    assert [candidate.chunk.id for candidate in packet.candidates] == [
        *[f"o{i}::0" for i in range(8)], "h1::0", "h2::0", *[f"o{i}::0" for i in range(8, 10)],
    ]
    assert all(candidate.chunk.id != "card::0001" for candidate in packet.candidates)


def test_below_floor_card_does_not_issue_secondary_search():
    original = [_hit(f"o{i}::0", 20 - i) for i in range(12)]
    retriever = RecordingRetriever({
        ("speaker_clean", "soru"): original,
        ("speaker_cards", "soru"): [_hit("card::0001", -1.0001)],
    })
    packet = collect_candidates(
        "soru", _settings(), retrieve_fn=retriever,
        card_manifest_validator=lambda **kwargs: _valid_status(),
    )
    assert len(retriever.calls) == 2
    assert packet.hint_hits == ()


def test_card_branch_exception_discards_hints_but_keeps_originals():
    original = [_hit(f"o{i}::0", 20 - i) for i in range(12)]
    retriever = RecordingRetriever({
        ("speaker_clean", "soru"): original,
        ("speaker_cards", "soru"): RuntimeError("card store unavailable"),
    })
    packet = collect_candidates(
        "soru", _settings(), retrieve_fn=retriever,
        card_manifest_validator=lambda **kwargs: _valid_status(),
    )
    assert [c.chunk.id for c in packet.candidates] == [f"o{i}::0" for i in range(12)]
    assert packet.diagnostics["card_hint_status"] == "retrieval_error"


def test_manifest_validation_exception_degrades_to_original_only_with_diagnostics():
    original = [_hit(f"o{i}::0", 20 - i) for i in range(12)]
    retriever = RecordingRetriever({("speaker_clean", "soru"): original})

    def raising_validator(**kwargs):
        raise OSError("manifest unreadable")

    packet = collect_candidates(
        "soru", _settings(), retrieve_fn=retriever,
        card_manifest_validator=raising_validator,
    )
    assert [candidate.chunk.id for candidate in packet.candidates] == [f"o{i}::0" for i in range(12)]
    assert packet.hint_hits == ()
    assert packet.diagnostics["card_hint_status"] == "retrieval_error"
    assert packet.diagnostics["card_hint_error"] == "OSError: manifest unreadable"
    assert len(retriever.calls) == 1


def test_original_retrieval_exception_is_fatal():
    retriever = RecordingRetriever({("speaker_clean", "soru"): RuntimeError("clean store down")})
    with pytest.raises(CandidateRetrievalError, match="original clean retrieval failed"):
        collect_candidates(
            "soru", _settings(), retrieve_fn=retriever,
            card_manifest_validator=lambda **kwargs: _valid_status(),
        )
