import json
from types import SimpleNamespace

import pytest

from rag.prompt import REFUSAL_TR
from ui.chat_runtime import (
    _EMPTY_OUTPUT_TR,
    _ERROR_TR,
    _FAILURE_REASONS,
    chat_turn,
    retrieve_decision,
    validate_v2_response,
)


class AnswerResponse:
    def __init__(self, content="cevap"):
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"message": {"content": self.content}, "eval_count": 10}


class CountingPost:
    def __init__(self):
        self.calls = []

    def __call__(self, method, url, *, payload, timeout_s):
        self.calls.append((url, payload, timeout_s))
        return {"message": {"content": "cevap"}, "eval_count": 10}


class SequencedPost(CountingPost):
    def __init__(self, *contents):
        super().__init__()
        self.contents = iter(contents)

    def __call__(self, method, url, *, payload, timeout_s):
        self.calls.append((url, payload, timeout_s))
        return {"message": {"content": next(self.contents)}, "eval_count": 10}


def _payload(status="answerable"):
    evidence = [{
        "id": "span::v::1000::2000", "video_id": "v", "title": "Başlık",
        "start": 1.0, "end": 2.0, "text": "temiz evidence", "retrieval_score": 2.0,
        "score": 2.0, "source_chunk_ids": ["v::0"], "query_origins": ["original"],
    }]
    grounded = status in {"answerable", "partial"}
    return {
        "schema_version": 2,
        "status": status,
        "grounded": grounded,
        "failure_reason": "verifier_transport_error" if status == "error" else None,
        "claims": ([{
            "claim_id": "C1", "claim": "desteklenen iddia", "central": True,
            "verdict": "entailed", "evidence_ids": ["span::v::1000::2000"], "reason": "r",
        }] + ([{
            "claim_id": "C2", "claim": "uydurma altı sayfalık mektup", "central": True,
            "verdict": "not_found", "evidence_ids": [], "reason": "r",
        }] if status == "partial" else [])) if grounded else [],
        "evidence_hits": evidence if grounded else [],
        "hint_hits": [{
            "card_id": "card::0001", "question": "hint q", "take": "UNTRUSTED CARD TAKE",
            "retrieval_score": 0.5,
        }],
        "answer_question": "desteklenen iddia" if grounded else None,
        "unsupported_claims": ["uydurma altı sayfalık mektup"] if status == "partial" else [],
        "diagnostics": {},
        "hits": evidence if grounded else [],
        "expanded_video": None,
        "expanded_title": None,
        "expanded_spans": [],
        "threshold": None,
    }


def _unsupported_payload():
    payload = _payload("unsupported")
    payload["unsupported_claims"] = ["yanıtsız iddia"]
    payload["claims"] = [{
        "claim_id": "C1",
        "claim": "yanıtsız iddia",
        "central": True,
        "verdict": "not_found",
        "evidence_ids": [],
        "reason": "r",
    }]
    return payload


def test_answerable_calls_model_with_clean_evidence_but_never_card_take():
    post = CountingPost()
    history, context, meta = chat_turn(
        "original", [], "speaker-model", "grounded_strict", "clean_with_card_hints",
        retrieve_fn=lambda query, mode: _payload("answerable"), answer_post=post,
        ensure_absent=lambda model: None,
    )
    assert len(post.calls) == 1
    serialized = json.dumps(post.calls[0][1], ensure_ascii=False)
    assert "temiz evidence" in serialized
    assert "UNTRUSTED CARD TAKE" not in serialized
    assert post.calls[0][1]["think"] is False
    assert post.calls[0][1]["options"]["num_ctx"] == 8192
    assert history[-1]["content"] == "cevap"
    assert "search hints — not evidence" in context
    assert "watch?v=v&t=1s" in context


def test_partial_model_sees_only_safe_question_and_ui_names_missing_claim():
    post = CountingPost()
    history, context, meta = chat_turn(
        "original includes uydurma altı sayfalık mektup",
        [],
        "speaker-model",
        "grounded_strict",
        "clean",
        retrieve_fn=lambda query, mode: _payload("partial"), answer_post=post,
        ensure_absent=lambda model: None,
    )
    serialized = json.dumps(post.calls[0][1], ensure_ascii=False)
    assert "desteklenen iddia" in serialized
    assert "uydurma altı sayfalık mektup" not in serialized
    assert history[-1]["content"] == "cevap"
    assert "uydurma altı sayfalık mektup" in meta


def test_unsupported_error_and_bad_schema_make_zero_answer_calls():
    for payload in (_payload("unsupported"), _payload("error"), {"schema_version": 1}):
        post = CountingPost()
        history, context, meta = chat_turn(
            "question", [], "speaker-model", "grounded_strict", "clean",
            retrieve_fn=lambda query, mode, payload=payload: payload, answer_post=post,
            ensure_absent=lambda model: None,
        )
        assert post.calls == []
        assert history[-1]["content"]


def test_internal_chat_result_exposes_the_exact_retrieval_used_for_generation():
    from ui.chat_runtime import chat_turn_with_trace

    post = CountingPost()
    result = chat_turn_with_trace(
        "question",
        [],
        "speaker-model",
        "grounded_strict",
        "clean",
        retrieve_fn=lambda _query, _mode: _payload("partial"),
        answer_post=post,
        ensure_absent=lambda _model: None,
    )

    assert result.trace == {"status": "partial", "evidence_count": 1}
    assert result.history[-1]["content"] == "cevap"
    assert len(post.calls) == 1


def test_fallback_generation_cannot_hide_failed_retrieval_from_internal_trace():
    from ui.chat_runtime import chat_turn_with_trace

    post = CountingPost()
    result = chat_turn_with_trace(
        "question",
        [],
        "speaker-model",
        "grounded_fallback",
        "clean",
        retrieve_fn=lambda _query, _mode: (_ for _ in ()).throw(ConnectionError("RAG down")),
        answer_post=post,
        ensure_absent=lambda _model: None,
    )

    assert result.trace == {"status": "error", "evidence_count": 0}
    assert result.history[-1]["content"] == "cevap"
    assert len(post.calls) == 1


def test_unsupported_claim_returns_refusal_without_calling_speaker():
    payload = _payload("unsupported")
    payload["unsupported_claims"] = ["uydurma altı sayfalık mektup"]
    post = CountingPost()

    history, context, meta = chat_turn(
        "question", [], "speaker-model", "grounded_strict", "clean",
        retrieve_fn=lambda query, mode: payload, answer_post=post,
        ensure_absent=lambda model: None,
    )

    assert post.calls == []
    assert history[-1]["content"] == REFUSAL_TR


def test_unsupported_diagnostic_span_claim_returns_refusal_without_calling_speaker():
    payload = _payload("unsupported")
    payload["unsupported_claims"] = ["yanıtsız merkezi iddia"]
    payload["claims"] = [
        {
            "claim_id": "C1", "claim": "yanıtsız merkezi iddia", "central": True,
            "verdict": "not_found", "evidence_ids": [], "reason": "r",
        },
        {
            "claim_id": "C2", "claim": "tanısal arka plan", "central": False,
            "verdict": "entailed", "evidence_ids": ["span::sample00017::167903::222189"], "reason": "r",
        },
    ]
    post = CountingPost()

    history, context, meta = chat_turn(
        "question", [], "speaker-model", "grounded_strict", "clean",
        retrieve_fn=lambda query, mode: payload, answer_post=post,
        ensure_absent=lambda model: None,
    )

    assert post.calls == []
    assert history[-1]["content"] == REFUSAL_TR


def test_unsupported_diagnostic_claim_rejects_malformed_span_id():
    payload = _payload("unsupported")
    payload["claims"] = [{
        "claim_id": "C1", "claim": "tanısal arka plan", "central": False,
        "verdict": "contradicted", "evidence_ids": ["not-a-span"], "reason": "r",
    }]

    with pytest.raises(ValueError, match="diagnostic span ID"):
        validate_v2_response(payload)


def test_empty_answer_retries_once_without_repeating_rag_work(monkeypatch):
    post = SequencedPost("", "second reply")
    events = []
    monkeypatch.setattr(
        "ui.chat_runtime.load_settings",
        lambda: SimpleNamespace(rag=SimpleNamespace(verifier_model="qwen3:4b")),
    )

    history, context, meta = chat_turn(
        "question", [], "speaker-model", "grounded_strict", "clean",
        answer_token_cap=1024,
        retrieve_fn=lambda query, mode: events.append(("retrieve", query, mode)) or _payload(),
        answer_post=lambda *args, **kwargs: events.append(("answer",)) or post(*args, **kwargs),
        ensure_absent=lambda model: events.append(("absent", model)),
    )

    assert len(post.calls) == 2
    assert all(call[1]["think"] is False for call in post.calls)
    assert all(call[1]["options"]["num_predict"] == 1024 for call in post.calls)
    assert post.calls[0][1] == post.calls[1][1]
    assert history[-1]["content"] == "second reply"
    assert events == [
        ("absent", "speaker-model"),
        ("retrieve", "question", "clean"),
        ("absent", "qwen3:4b"),
        ("answer",),
        ("answer",),
    ]


def test_second_empty_answer_returns_a_non_empty_turkish_error():
    post = SequencedPost("", "")

    history, context, meta = chat_turn(
        "free question", [], "speaker-model", "free", "clean", answer_post=post,
    )

    assert len(post.calls) == 2
    assert history[-1]["content"]
    assert "boş model çıktısı" in history[-1]["content"].lower()


def test_internal_result_marks_two_empty_model_replies_as_empty_output():
    from ui.chat_runtime import chat_turn_with_trace

    result = chat_turn_with_trace(
        "question",
        [],
        "speaker-model",
        "grounded_strict",
        "clean",
        retrieve_fn=lambda _query, _mode: _payload("answerable"),
        answer_post=SequencedPost("", ""),
        ensure_absent=lambda _model: None,
    )

    assert result.trace == {"status": "answerable", "evidence_count": 1}
    assert result.generation_status == "empty_output"
    assert result.history[-1]["content"] == _EMPTY_OUTPUT_TR


def test_internal_result_marks_model_exceptions_and_policy_refusals():
    from ui.chat_runtime import chat_turn_with_trace

    model_error = chat_turn_with_trace(
        "question",
        [],
        "speaker-model",
        "free",
        "clean",
        answer_post=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ConnectionError("Ollama unavailable")
        ),
    )
    refusal = chat_turn_with_trace(
        "question",
        [],
        "speaker-model",
        "grounded_strict",
        "clean",
        retrieve_fn=lambda _query, _mode: _payload("unsupported"),
        ensure_absent=lambda _model: None,
    )

    assert model_error.generation_status == "model_error"
    assert refusal.generation_status == "policy_refusal"


def test_free_mode_uses_the_bare_question_generation_path():
    post = CountingPost()
    history, context, meta = chat_turn(
        "free question", [], "speaker-model", "free", "clean",
        retrieve_fn=lambda query, mode: (_ for _ in ()).throw(AssertionError("no retrieval")),
        answer_post=post,
    )
    assert len(post.calls) == 1
    assert post.calls[0][1]["messages"][1]["content"] == "free question"
    assert context == "_RAG kapalı: serbest üretim._"


def test_v2_validation_rejects_card_chunks_and_oversized_context():
    card = _payload()
    card["evidence_hits"][0]["source_chunk_ids"] = ["card::0001"]
    try:
        validate_v2_response(card)
    except ValueError as exc:
        assert "card" in str(exc)
    else:
        raise AssertionError("card evidence was accepted")
    oversized = _payload()
    oversized["evidence_hits"][0]["text"] = "x" * 1801
    try:
        validate_v2_response(oversized)
    except ValueError as exc:
        assert "1800" in str(exc)
    else:
        raise AssertionError("oversized evidence was accepted")


def test_v2_validation_rejects_card_take_with_a_forged_clean_source_id():
    payload = _payload()
    payload["evidence_hits"][0]["text"] = "UNTRUSTED CARD TAKE"
    try:
        validate_v2_response(payload)
    except ValueError as exc:
        assert "card take" in str(exc)
    else:
        raise AssertionError("card take was accepted as evidence")


def test_malformed_or_inconsistent_grounded_payloads_make_zero_answer_calls():
    malformed = []

    bad_grounded = _payload()
    bad_grounded["grounded"] = False
    malformed.append(bad_grounded)

    bad_failure_reason = _payload()
    bad_failure_reason["failure_reason"] = "retrieval_error"
    malformed.append(bad_failure_reason)

    bad_claim_type = _payload()
    bad_claim_type["claims"][0]["central"] = "true"
    malformed.append(bad_claim_type)

    missing_claims = _payload()
    missing_claims["claims"] = []
    malformed.append(missing_claims)

    unsupported_answerable_claim = _payload()
    unsupported_answerable_claim["claims"][0]["verdict"] = "not_found"
    unsupported_answerable_claim["claims"][0]["evidence_ids"] = []
    malformed.append(unsupported_answerable_claim)

    bad_claim_reference = _payload()
    bad_claim_reference["claims"][0]["evidence_ids"] = ["span::missing::0::1"]
    malformed.append(bad_claim_reference)

    bad_span_metadata = _payload()
    bad_span_metadata["evidence_hits"][0]["start"] = "one"
    malformed.append(bad_span_metadata)

    bad_hint_type = _payload()
    bad_hint_type["hint_hits"][0]["take"] = 0
    malformed.append(bad_hint_type)

    bad_partial = _payload("partial")
    bad_partial["unsupported_claims"] = []
    malformed.append(bad_partial)

    for payload in malformed:
        post = CountingPost()
        history, context, meta = chat_turn(
            "question", [], "speaker-model", "grounded_strict", "clean",
            retrieve_fn=lambda query, mode, payload=payload: payload, answer_post=post,
            ensure_absent=lambda model: None,
        )
        assert post.calls == []
        assert history[-1]["content"]


def test_v2_validation_rejects_nonstring_status_with_a_value_error():
    payload = _payload()
    payload["status"] = {"answerable": True}
    try:
        validate_v2_response(payload)
    except ValueError as exc:
        assert "status" in str(exc)
    else:
        raise AssertionError("non-string status was accepted")


def test_partial_without_a_matching_central_not_found_claim_makes_zero_answer_calls():
    payload = _payload("partial")
    payload["claims"] = payload["claims"][:1]
    post = CountingPost()

    history, context, meta = chat_turn(
        "question", [], "speaker-model", "grounded_strict", "clean",
        retrieve_fn=lambda query, mode: payload, answer_post=post,
        ensure_absent=lambda model: None,
    )

    assert post.calls == []
    assert history[-1]["content"]


def test_grounded_strict_hands_off_speaker_then_configured_verifier_before_answer(monkeypatch):
    post = CountingPost()
    events = []
    monkeypatch.setattr(
        "ui.chat_runtime.load_settings",
        lambda: SimpleNamespace(rag=SimpleNamespace(verifier_model="qwen3:4b")),
    )

    chat_turn(
        "question", [], "speaker-model", "grounded_strict", "clean",
        retrieve_fn=lambda query, mode: events.append(("retrieve", query, mode)) or _payload(),
        answer_post=lambda *args, **kwargs: events.append(("answer",)) or post(*args, **kwargs),
        ensure_absent=lambda model: events.append(("absent", model)),
    )

    assert events == [
        ("absent", "speaker-model"),
        ("retrieve", "question", "clean"),
        ("absent", "qwen3:4b"),
        ("answer",),
    ]
    assert post.calls[0][1]["keep_alive"] == -1


def test_grounded_strict_speaker_unload_failure_returns_operational_error_without_answer_call():
    post = CountingPost()
    retrieved = []

    history, context, meta = chat_turn(
        "question", [], "speaker-model", "grounded_strict", "clean",
        retrieve_fn=lambda query, mode: retrieved.append((query, mode)) or _payload(),
        answer_post=post,
        ensure_absent=lambda model: (_ for _ in ()).throw(RuntimeError("unload failed")),
    )

    assert post.calls == []
    assert retrieved == []
    assert history[-1]["content"] == _ERROR_TR
    assert "RuntimeError: unload failed" in context
    assert meta == ""


def test_grounded_strict_verifier_unload_failure_returns_operational_error_without_answer_call(monkeypatch):
    post = CountingPost()
    calls = []
    monkeypatch.setattr(
        "ui.chat_runtime.load_settings",
        lambda: SimpleNamespace(rag=SimpleNamespace(verifier_model="qwen3:4b")),
    )

    def fail_on_verifier(model):
        calls.append(("absent", model))
        if model != "speaker-model":
            raise RuntimeError("verifier still loaded")

    history, context, meta = chat_turn(
        "question", [], "speaker-model", "grounded_strict", "clean",
        retrieve_fn=lambda query, mode: calls.append(("retrieve",)) or _payload(),
        answer_post=post,
        ensure_absent=fail_on_verifier,
    )

    assert calls == [("absent", "speaker-model"), ("retrieve",), ("absent", "qwen3:4b")]
    assert post.calls == []
    assert history[-1]["content"] == _ERROR_TR
    assert "RuntimeError: verifier still loaded" in context
    assert meta == ""


def test_free_mode_does_not_unload_speaker_before_direct_generation():
    post = CountingPost()

    history, context, meta = chat_turn(
        "free question", [], "speaker-model", "free", "clean",
        answer_post=post,
        ensure_absent=lambda model: (_ for _ in ()).throw(AssertionError("must not unload")),
    )

    assert history[-1]["content"] == "cevap"
    assert "RAG" in context
    assert meta
    assert post.calls[0][1]["keep_alive"] == -1


@pytest.mark.parametrize(
    ("payload", "warning"),
    [
        (_unsupported_payload(), "not supported by verified transcripts"),
        (_payload("error"), "transcript verification was unavailable"),
        ({"schema_version": 1}, "transcript verification was unavailable"),
    ],
)
def test_grounded_fallback_answers_without_evidence_and_marks_warning(
    monkeypatch, payload, warning
):
    post = CountingPost()
    events = []
    monkeypatch.setattr(
        "ui.chat_runtime.load_settings",
        lambda: SimpleNamespace(rag=SimpleNamespace(verifier_model="qwen3:4b")),
    )

    history, context, meta = chat_turn(
        "original question",
        [],
        "speaker-model",
        "grounded_fallback",
        "clean",
        retrieve_fn=lambda query, mode: payload,
        answer_post=lambda *args, **kwargs: events.append(("answer",))
        or post(*args, **kwargs),
        ensure_absent=lambda model: events.append(("absent", model)),
    )

    assert post.calls[0][1]["messages"][1]["content"] == "original question"
    assert history[-1]["content"] == "cevap"
    assert warning in meta
    assert "unverified fallback" in meta
    assert "watch?v=" not in context
    assert events[-2:] == [("absent", "qwen3:4b"), ("answer",)]


def test_grounded_fallback_answers_when_retrieval_raises(monkeypatch):
    monkeypatch.setattr(
        "ui.chat_runtime.load_settings",
        lambda: SimpleNamespace(rag=SimpleNamespace(verifier_model="qwen3:4b")),
    )
    post = CountingPost()

    history, context, meta = chat_turn(
        "original",
        [],
        "speaker-model",
        "grounded_fallback",
        "clean",
        retrieve_fn=lambda *_: (_ for _ in ()).throw(RuntimeError("server down")),
        answer_post=post,
        ensure_absent=lambda model: None,
    )

    assert history[-1]["content"] == "cevap"
    assert post.calls[0][1]["messages"][1]["content"] == "original"
    assert "verification was unavailable" in meta
    assert "RuntimeError: server down" in context


@pytest.mark.parametrize("reason", sorted(_FAILURE_REASONS))
def test_every_valid_evidence_error_reason_falls_back(monkeypatch, reason):
    payload = _payload("error")
    payload["failure_reason"] = reason
    monkeypatch.setattr(
        "ui.chat_runtime.load_settings",
        lambda: SimpleNamespace(rag=SimpleNamespace(verifier_model="qwen3:4b")),
    )
    post = CountingPost()

    history, context, meta = chat_turn(
        "original",
        [],
        "speaker-model",
        "grounded_fallback",
        "clean",
        retrieve_fn=lambda *_: payload,
        answer_post=post,
        ensure_absent=lambda model: None,
    )

    assert history[-1]["content"] == "cevap"
    assert len(post.calls) == 1
    assert post.calls[0][1]["messages"][1]["content"] == "original"
    assert "unverified fallback" in meta


def test_grounded_fallback_keeps_answerable_prompt_grounded():
    post = CountingPost()

    history, context, meta = chat_turn(
        "original",
        [],
        "speaker-model",
        "grounded_fallback",
        "clean",
        retrieve_fn=lambda *_: _payload("answerable"),
        answer_post=post,
        ensure_absent=lambda model: None,
    )

    serialized = json.dumps(post.calls[0][1], ensure_ascii=False)
    assert "temiz evidence" in serialized
    assert history[-1]["content"] == "cevap"
    assert "unverified fallback" not in meta


def test_grounded_fallback_keeps_partial_prompt_scoped():
    post = CountingPost()

    history, context, meta = chat_turn(
        "original includes uydurma altı sayfalık mektup",
        [],
        "speaker-model",
        "grounded_fallback",
        "clean",
        retrieve_fn=lambda *_: _payload("partial"),
        answer_post=post,
        ensure_absent=lambda model: None,
    )

    serialized = json.dumps(post.calls[0][1], ensure_ascii=False)
    assert "desteklenen iddia" in serialized
    assert "uydurma altı sayfalık mektup" not in serialized
    assert "unverified fallback" not in meta


def test_grounded_fallback_verifier_unload_failure_still_blocks_answer(monkeypatch):
    monkeypatch.setattr(
        "ui.chat_runtime.load_settings",
        lambda: SimpleNamespace(rag=SimpleNamespace(verifier_model="qwen3:4b")),
    )
    post = CountingPost()

    def ensure(model):
        if model == "qwen3:4b":
            raise RuntimeError("verifier still loaded")

    history, context, meta = chat_turn(
        "original",
        [],
        "speaker-model",
        "grounded_fallback",
        "clean",
        retrieve_fn=lambda *_: _unsupported_payload(),
        answer_post=post,
        ensure_absent=ensure,
    )

    assert post.calls == []
    assert history[-1]["content"] == _ERROR_TR
    assert "verifier still loaded" in context
    assert meta == ""


def test_fallback_warning_is_not_stored_in_assistant_history(monkeypatch):
    monkeypatch.setattr(
        "ui.chat_runtime.load_settings",
        lambda: SimpleNamespace(rag=SimpleNamespace(verifier_model="qwen3:4b")),
    )

    history, context, meta = chat_turn(
        "original",
        [],
        "speaker-model",
        "grounded_fallback",
        "clean",
        retrieve_fn=lambda *_: _unsupported_payload(),
        answer_post=CountingPost(),
        ensure_absent=lambda model: None,
    )

    assert "Unverified model answer" in meta
    assert history[-1]["content"] == "cevap"
    assert "Unverified model answer" not in history[-1]["content"]


def test_invalid_answer_mode_is_rejected_before_any_side_effect():
    with pytest.raises(ValueError, match="unknown answer mode"):
        chat_turn(
            "original",
            [],
            "speaker-model",
            True,
            "clean",
            retrieve_fn=lambda *_: (_ for _ in ()).throw(
                AssertionError("retrieval must not run")
            ),
            answer_post=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("answer must not run")
            ),
        )


@pytest.mark.parametrize("cap", [64, 640, 4096])
def test_valid_answer_token_cap_reaches_ollama(cap):
    post = CountingPost()

    history, context, meta = chat_turn(
        "question",
        [],
        "speaker-model",
        "free",
        "clean",
        answer_token_cap=cap,
        answer_post=post,
    )

    assert history[-1]["content"] == "cevap"
    assert post.calls[0][1]["options"]["num_predict"] == cap


def test_default_answer_token_cap_preserves_640():
    post = CountingPost()

    chat_turn(
        "question", [], "speaker-model", "free", "clean", answer_post=post
    )

    assert post.calls[0][1]["options"]["num_predict"] == 640


def test_packaged_retrieval_request_sends_the_launcher_bearer(monkeypatch):
    seen = {}

    class Response:
        def read(self):
            return json.dumps(_payload()).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        "ui.chat_runtime._RUNTIME_SETTINGS",
        SimpleNamespace(packaged=True, session_secret="s" * 64),
        raising=False,
    )

    def opener(request, *, timeout):
        seen["authorization"] = request.get_header("Authorization")
        return Response()

    assert retrieve_decision("soru", "clean", opener=opener)["status"] == "answerable"
    assert seen["authorization"] == "Bearer " + "s" * 64


@pytest.mark.parametrize("cap", [63, 4097, True, False, 640.0, "640", None])
def test_invalid_answer_token_cap_fails_before_side_effects(cap):
    with pytest.raises(ValueError, match="answer token cap"):
        chat_turn(
            "question",
            [],
            "speaker-model",
            "grounded_strict",
            "clean",
            answer_token_cap=cap,
            retrieve_fn=lambda *_: (_ for _ in ()).throw(
                AssertionError("retrieval must not run")
            ),
            answer_post=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("answer must not run")
            ),
            ensure_absent=lambda _model: (_ for _ in ()).throw(
                AssertionError("residency must not change")
            ),
        )
