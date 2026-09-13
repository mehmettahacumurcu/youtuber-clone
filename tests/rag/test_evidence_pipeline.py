import pytest

from pipeline.config import load_settings
from rag.candidates import CandidateRetrievalError
from rag.context import EvidenceSourceError
from rag.evidence_pipeline import (
    EvidenceDecision,
    assess_question,
    build_partial_question,
    derive_status,
    select_evidence,
)
from rag.types import Candidate, CandidatePacket, CardHint, Chunk, EvidenceSpan
from rag.verifier import (
    ClaimVerdict,
    VerificationResult,
    VerifierClaim,
    VerifierInvalidOutput,
    VerifierOutput,
    VerifierTransportError,
)


def _span(index: int) -> EvidenceSpan:
    return EvidenceSpan(
        f"span::v{index}::0::1000", f"v{index}", f"T{index}", 0.0, 1.0,
        f"evidence {index}", float(20 - index), (f"v{index}::0",), ("original",),
    )


def _claim(index, verdict, evidence=(), central=True, text=None):
    return VerifierClaim(
        claim_id=f"C{index}", claim=text or f"claim {index}", central=central,
        verdict=verdict, evidence_ids=list(evidence), reason="reason",
    )


class FakeVerifier:
    def __init__(self, claims=None, error=None):
        self.claims = claims
        self.error = error
        self.calls = []

    def verify(self, question, spans):
        self.calls.append((question, tuple(spans)))
        if self.error is not None:
            raise self.error
        output = VerifierOutput(claims=self.claims)
        return VerificationResult(output, {"wall_s": 1.0})


def _packet():
    candidate = Candidate(
        Chunk("v1::0", "v1", 0.0, 1.0, "T", "chunk"), 2.0, ("original",),
    )
    return CandidatePacket(
        (candidate,), (CardHint("card::0001", "q", "take", 1.0),), {"x": 1},
    )


def _run(claims, spans=(_span(1), _span(2)), packet=None):
    verifier = FakeVerifier(claims)
    decision = assess_question(
        "original question",
        load_settings(),
        verifier=verifier,
        collect_fn=lambda *args, **kwargs: packet or _packet(),
        resolve_fn=lambda *args, **kwargs: spans,
        score_segments=lambda question, texts: [0.0 for _ in texts],
    )
    return decision, verifier


def test_status_derivation_is_deterministic():
    assert derive_status([_claim(1, ClaimVerdict.ENTAILED, (_span(1).id,))]) == "answerable"
    assert derive_status([
        _claim(1, ClaimVerdict.CONTRADICTED, (_span(1).id,)),
        _claim(2, ClaimVerdict.NOT_FOUND),
    ]) == "partial"
    assert derive_status([_claim(1, ClaimVerdict.NOT_FOUND)]) == "unsupported"
    assert derive_status([
        _claim(1, ClaimVerdict.EXPLICITLY_UNRESOLVED, (_span(1).id,)),
    ]) == "answerable"


def test_answerable_selects_all_cited_spans_in_packet_rank_order():
    claims = [
        _claim(1, ClaimVerdict.ENTAILED, (_span(2).id, _span(1).id)),
        _claim(2, ClaimVerdict.CONTRADICTED, (_span(1).id,)),
    ]
    decision, verifier = _run(claims)
    assert decision.status == "answerable"
    assert decision.answer_question == "original question"
    assert [span.id for span in decision.selected_spans] == [_span(1).id, _span(2).id]
    assert decision.hint_hits[0].card_id == "card::0001"
    assert len(verifier.calls) == 1


def test_partial_scope_excludes_unsupported_wording_and_preserves_notice():
    supported = "Demirören krediyi Ziraat Bankası'ndan aldı."
    unsupported = "Uğur Mumcu'ya altı sayfalık mektup gönderdi."
    claims = [
        _claim(1, ClaimVerdict.ENTAILED, (_span(1).id,), text=supported),
        _claim(2, ClaimVerdict.NOT_FOUND, text=unsupported),
    ]
    decision, _ = _run(claims)
    assert decision.status == "partial"
    assert supported in decision.answer_question
    assert unsupported not in decision.answer_question
    assert decision.unsupported_claims == (unsupported,)


def test_empty_candidates_and_empty_spans_are_unsupported_without_verifier_call():
    for packet, spans in [(CandidatePacket((), (), {}), (_span(1),)), (_packet(), ())]:
        verifier = FakeVerifier([])
        decision = assess_question(
            "question", load_settings(), verifier=verifier,
            collect_fn=lambda *args, packet=packet, **kwargs: packet,
            resolve_fn=lambda *args, spans=spans, **kwargs: spans,
            score_segments=lambda question, texts: [0.0 for _ in texts],
        )
        assert decision.status == "unsupported"
        assert decision.selected_spans == ()
        assert decision.answer_question is None
        assert verifier.calls == []


@pytest.mark.parametrize(
    ("stage", "error", "reason"),
    [
        ("collect", CandidateRetrievalError("down"), "retrieval_error"),
        ("resolve", EvidenceSourceError("bad source"), "source_error"),
        ("verify", VerifierTransportError("timeout"), "verifier_transport_error"),
        ("verify", VerifierInvalidOutput("bad json"), "verifier_invalid_output"),
    ],
)
def test_stage_failures_map_to_fail_closed_decisions(stage, error, reason):
    verifier = FakeVerifier([_claim(1, ClaimVerdict.ENTAILED, (_span(1).id,))])
    if stage == "verify":
        verifier.error = error
    decision = assess_question(
        "question", load_settings(), verifier=verifier,
        collect_fn=(lambda *args, **kwargs: (_ for _ in ()).throw(error)) if stage == "collect"
        else (lambda *args, **kwargs: _packet()),
        resolve_fn=(lambda *args, **kwargs: (_ for _ in ()).throw(error)) if stage == "resolve"
        else (lambda *args, **kwargs: (_span(1),)),
        score_segments=lambda question, texts: [0.0 for _ in texts],
    )
    assert decision.status == "error"
    assert decision.failure_reason == reason
    assert decision.selected_spans == ()
    assert decision.answer_question is None


def test_unexpected_verifier_error_fails_closed_with_generic_verifier_reason():
    verifier = FakeVerifier(error=RuntimeError("unexpected verifier failure"))
    decision = assess_question(
        "question", load_settings(), verifier=verifier,
        collect_fn=lambda *args, **kwargs: _packet(),
        resolve_fn=lambda *args, **kwargs: (_span(1),),
        score_segments=lambda question, texts: [0.0 for _ in texts],
    )
    assert decision.status == "error"
    assert decision.failure_reason == "verifier_error"
    assert decision.selected_spans == ()
    assert decision.answer_question is None


def test_verifier_output_without_central_claim_fails_closed_as_invalid_output():
    verifier = FakeVerifier([_claim(1, ClaimVerdict.NOT_FOUND, central=False)])
    decision = assess_question(
        "question", load_settings(), verifier=verifier,
        collect_fn=lambda *args, **kwargs: _packet(),
        resolve_fn=lambda *args, **kwargs: (_span(1),),
        score_segments=lambda question, texts: [0.0 for _ in texts],
    )
    assert decision.status == "error"
    assert decision.failure_reason == "verifier_invalid_output"
    assert decision.selected_spans == ()
    assert decision.answer_question is None


def test_more_than_five_unique_verifier_citations_is_budget_error():
    spans = tuple(_span(index) for index in range(1, 7))
    claims = [
        _claim(index, ClaimVerdict.ENTAILED, (spans[index - 1].id,), central=index <= 5)
        for index in range(1, 7)
    ]
    decision, _ = _run(claims, spans=spans)
    assert decision.status == "error"
    assert decision.failure_reason == "evidence_budget_error"
    assert decision.selected_spans == ()


def test_request_mode_override_does_not_mutate_global_settings():
    settings = load_settings()
    seen = []
    verifier = FakeVerifier([_claim(1, ClaimVerdict.NOT_FOUND)])
    assess_question(
        "question", settings, verifier=verifier, retrieval_mode="clean",
        collect_fn=lambda question, config, **kwargs: (
            seen.append(config.retrieval_mode) or CandidatePacket((), (), {})
        ),
        score_segments=lambda question, texts: [0.0 for _ in texts],
    )
    assert seen == ["clean"]
    assert settings.rag.retrieval_mode == "clean_with_card_hints"
