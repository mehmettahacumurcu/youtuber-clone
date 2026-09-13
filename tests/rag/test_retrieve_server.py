from rag.evidence_pipeline import EvidenceDecision
from rag.retrieve_server import (
    _probe_retrieval_models,
    build_ready_payload,
    decision_to_response,
    handle_retrieve,
    require_rag_bearer,
    resolve_retrieval_mode,
)
from rag.types import CardHint, EvidenceSpan
from rag.verifier import ClaimVerdict, VerifierClaim


def test_warmup_failure_prints_a_traceback_for_packaged_diagnostics(monkeypatch, capsys):
    import rag.retrieve_server as server

    previous_state = dict(server._STATE)
    previous_identity = server._IDENTITY
    monkeypatch.setattr(
        server,
        "approved_identity",
        lambda settings: (_ for _ in ()).throw(OSError("could not get source code")),
    )
    try:
        server._warm()
        captured = capsys.readouterr()
        assert "Traceback (most recent call last)" in captured.err
        assert "OSError: could not get source code" in captured.err
        assert server._STATE["status"] == "error"
    finally:
        server._STATE.clear()
        server._STATE.update(previous_state)
        server._IDENTITY = previous_identity


def _decision(status="answerable"):
    span = EvidenceSpan(
        "span::v::1000::2000", "v", "Başlık", 1.0, 2.0, "kanıt", 2.5,
        ("v::0",), ("original",),
    )
    claim = VerifierClaim(
        claim_id="C1", claim="iddia", central=True, verdict=ClaimVerdict.ENTAILED,
        evidence_ids=[span.id], reason="kanıtlıyor",
    )
    grounded = status in {"answerable", "partial"}
    return EvidenceDecision(
        status=status,
        claims=(claim,) if grounded else (),
        selected_spans=(span,) if grounded else (),
        hint_hits=(CardHint("card::0001", "kart sorusu", "kart take", 0.5),),
        answer_question="soru" if grounded else None,
        unsupported_claims=("eksik",) if status == "partial" else (),
        failure_reason="retrieval_error" if status == "error" else None,
        diagnostics={"trace": "x"},
    )


def test_response_v2_has_typed_evidence_hints_and_legacy_aliases():
    payload = decision_to_response(_decision())
    assert payload["schema_version"] == 2
    assert payload["status"] == "answerable"
    assert payload["grounded"] is True
    assert payload["evidence_hits"][0]["id"].startswith("span::")
    assert payload["evidence_hits"][0]["score"] == 2.5
    assert payload["hint_hits"][0]["card_id"] == "card::0001"
    assert payload["hits"] == payload["evidence_hits"]
    assert payload["expanded_video"] is None
    assert payload["expanded_title"] is None
    assert payload["expanded_spans"] == []
    assert payload["threshold"] is None


def test_mode_and_collection_compatibility_are_deterministic():
    assert resolve_retrieval_mode({"mode": ["clean"]}, "clean_with_card_hints")[0] == "clean"
    mode, diagnostics = resolve_retrieval_mode(
        {"collection": ["speaker_cards"], "top_n": ["6"]}, "clean",
    )
    assert mode == "clean_with_card_hints"
    assert diagnostics == {
        "deprecated_collection": "speaker_cards",
        "ignored_top_n": "6",
    }


def test_unknown_or_conflicting_modes_are_rejected():
    for params in (
        {"mode": ["unknown"]},
        {"collection": ["unknown"]},
        {"mode": ["clean"], "collection": ["speaker_cards"]},
    ):
        try:
            resolve_retrieval_mode(params, "clean")
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {params}")


def test_handle_retrieve_passes_request_mode_and_status_code():
    calls = []
    code, payload = handle_retrieve(
        {"q": ["soru"], "mode": ["clean"]},
        state={"status": "ready", "error": None},
        default_mode="clean_with_card_hints",
        assess_fn=lambda question, retrieval_mode: (
            calls.append((question, retrieval_mode)) or _decision("partial")
        ),
    )
    assert calls == [("soru", "clean")]
    assert code == 200 and payload["status"] == "partial"


def test_warming_invalid_and_pipeline_error_are_full_fail_closed_v2_bodies():
    warming_code, warming = handle_retrieve(
        {"q": ["soru"]}, state={"status": "warming", "error": None},
        default_mode="clean", assess_fn=lambda question, retrieval_mode: _decision(),
    )
    assert warming_code == 503
    assert warming["schema_version"] == 2 and warming["status"] == "error"
    invalid_code, invalid = handle_retrieve(
        {"q": [""]}, state={"status": "ready", "error": None},
        default_mode="clean", assess_fn=lambda question, retrieval_mode: _decision(),
    )
    assert invalid_code == 400 and invalid["grounded"] is False
    error_code, error = handle_retrieve(
        {"q": ["soru"]}, state={"status": "ready", "error": None},
        default_mode="clean", assess_fn=lambda question, retrieval_mode: _decision("error"),
    )
    assert error_code == 500 and error["failure_reason"] == "retrieval_error"


def test_packaged_rag_ready_payload_is_stable_and_bearer_does_not_leak():
    secret = "s" * 64
    payload = build_ready_payload(
        {"status": "ready", "error": None},
        corpus_version="corpus-v1", index_version="index-v1",
        embedder="BAAI/bge-m3", reranker="BAAI/bge-reranker-v2-m3",
    )

    assert require_rag_bearer("Bearer " + secret, secret, packaged=True) is True
    assert require_rag_bearer("Bearer bad", secret, packaged=True) is False
    assert require_rag_bearer(None, secret, packaged=True) is False
    assert require_rag_bearer(None, secret, packaged=False) is True
    assert payload == {
        "status": "ready", "runtime_api": 1, "schema_version": 2,
        "corpus_version": "corpus-v1", "index_version": "index-v1",
        "embedder": "BAAI/bge-m3", "reranker": "BAAI/bge-reranker-v2-m3",
    }
    assert secret not in str(payload)


def test_model_readiness_probe_executes_real_encode_and_score_paths():
    calls = []

    class Embedder:
        def encode(self, texts):
            calls.append(("encode", texts))
            return type("Embedding", (), {"dense": [[1.0]], "sparse": [{1: 1.0}]})()

    class Reranker:
        def score(self, query, passages):
            calls.append(("score", query, passages))
            return [0.5]

    _probe_retrieval_models(Embedder(), Reranker())

    assert calls == [
        ("encode", ["readiness probe"]),
        ("score", "readiness probe", ["readiness probe"]),
    ]
