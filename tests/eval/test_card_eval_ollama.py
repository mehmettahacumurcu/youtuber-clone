from __future__ import annotations

from collections import deque

import pytest
import requests

from eval.card_eval_ollama import (
    EmptyContentError,
    GenerationSettings,
    ModelPolicyError,
    OllamaClient,
    OllamaHTTPError,
    OllamaProtocolError,
    OllamaTransportError,
    PreflightRequiredError,
    ThinkingLeakError,
)


MODEL = "qwen3:14b"
DIGEST = "bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8"


class FakeResponse:
    def __init__(self, status_code=200, payload=None, *, json_error=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self._json_error = json_error
        self.text = text

    def json(self):
        if self._json_error is not None:
            raise self._json_error
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if not self.responses:
            raise AssertionError(f"unexpected request: {method} {url}")
        response = self.responses.popleft()
        if isinstance(response, BaseException):
            raise response
        return response


def _tag_row(*, name=MODEL, digest=DIGEST):
    return {
        "name": name,
        "model": name,
        "digest": digest,
        "size": 9_276_198_565,
        "details": {
            "format": "gguf",
            "family": "qwen3",
            "parameter_size": "14.8B",
            "quantization_level": "Q4_K_M",
        },
    }


def _metadata_responses(*, tags=None, show=None):
    if tags is None:
        tags = [_tag_row()]
    if show is None:
        show = {
            "details": {
                "format": "gguf",
                "family": "qwen3",
                "parameter_size": "14.8B",
                "quantization_level": "Q4_K_M",
            },
            "model_info": {
                "general.file_type": 15,
                "general.quantization_version": 2,
                "qwen3.context_length": 40960,
            },
            "capabilities": ["completion", "tools", "thinking"],
        }
    return [
        FakeResponse(payload={"version": "0.31.2"}),
        FakeResponse(payload={"models": tags}),
        FakeResponse(payload=show),
    ]


def _identity_check_responses(*, tags=None, version="0.31.2"):
    return [
        FakeResponse(payload={"version": version}),
        FakeResponse(payload={"models": tags or [_tag_row()]}),
    ]


def _settings(*, top_k=1):
    return GenerationSettings(
        temperature=0.0,
        seed=20260711,
        top_p=1.0,
        top_k=top_k,
        repeat_penalty=1.05,
        num_ctx=4096,
        num_predict=384,
    )


def _chat_payload(**overrides):
    payload = {
        "model": MODEL,
        "created_at": "2026-07-11T00:00:00Z",
        "message": {"role": "assistant", "content": "Doğru cevap.", "thinking": ""},
        "done": True,
        "done_reason": "stop",
        "total_duration": 100,
        "load_duration": 10,
        "prompt_eval_count": 42,
        "prompt_eval_duration": 20,
        "eval_count": 7,
        "eval_duration": 70,
    }
    payload.update(overrides)
    return payload


def _client_with_chat(chat_responses, *, sleeps=None):
    session = FakeSession(
        [
            *_metadata_responses(),
            *_identity_check_responses(),
            *chat_responses,
            *_identity_check_responses(),
        ]
    )
    recorded_sleeps = [] if sleeps is None else sleeps
    client = OllamaClient(session=session, sleep=recorded_sleeps.append)
    client.preflight(MODEL)
    return client, session, recorded_sleeps


def _chat_call_count(session):
    return len([call for call in session.calls if call[1].endswith("/api/chat")])


def test_preflight_records_exact_stock_model_identity_and_metadata_calls():
    session = FakeSession(_metadata_responses())
    client = OllamaClient(base_url="http://127.0.0.1:11434/", session=session)

    identity = client.preflight(MODEL)

    assert identity.to_json() == {
        "name": MODEL,
        "digest": DIGEST,
        "ollama_version": "0.31.2",
        "format": "gguf",
        "family": "qwen3",
        "parameter_size": "14.8B",
        "quantization_level": "Q4_K_M",
        "file_type": 15,
        "quantization_version": 2,
    }
    assert [(method, url) for method, url, _ in session.calls] == [
        ("GET", "http://127.0.0.1:11434/api/version"),
        ("GET", "http://127.0.0.1:11434/api/tags"),
        ("POST", "http://127.0.0.1:11434/api/show"),
    ]
    assert session.calls[2][2]["json"] == {"model": MODEL, "verbose": False}
    assert all(call[2]["timeout"] == (5, 30) for call in session.calls)


@pytest.mark.parametrize(
    "requested",
    [
        "qwen3:14b-latest",
        "QWEN3:14B",
        "huihui_ai/qwen3-abliterated:14b-v2",
    ],
)
def test_preflight_rejects_every_non_exact_or_abliterated_request(requested):
    session = FakeSession([])
    with pytest.raises(ModelPolicyError):
        OllamaClient(session=session).preflight(requested)
    assert not session.calls


def test_preflight_requires_one_exact_tag_and_rejects_conflicting_duplicates():
    missing = FakeSession(
        _metadata_responses(tags=[_tag_row(name="qwen3:8b")])[:2]
    )
    with pytest.raises(ModelPolicyError, match="not installed"):
        OllamaClient(session=missing).preflight(MODEL)

    other_digest = "a" * 64
    duplicate = FakeSession(
        _metadata_responses(tags=[_tag_row(), _tag_row(digest=other_digest)])[:2]
    )
    with pytest.raises(OllamaProtocolError, match="duplicate"):
        OllamaClient(session=duplicate).preflight(MODEL)


def test_preflight_rejects_well_formed_but_nonofficial_digest_and_metadata():
    retagged = FakeSession(
        _metadata_responses(tags=[_tag_row(digest="a" * 64)])[:2]
    )
    with pytest.raises(ModelPolicyError, match="official digest"):
        OllamaClient(session=retagged).preflight(MODEL)

    wrong_show = {
        "details": {
            "format": "safetensors",
            "family": "llama",
            "parameter_size": "14.8B",
            "quantization_level": "Q4_K_M",
        },
        "model_info": {},
    }
    malformed = FakeSession(_metadata_responses(show=wrong_show))
    with pytest.raises(ModelPolicyError, match="metadata"):
        OllamaClient(session=malformed).preflight(MODEL)


def test_preflight_rejects_malformed_digest_version_and_show_details():
    bad_version = FakeSession([FakeResponse(payload={"version": ""})])
    with pytest.raises(OllamaProtocolError, match="version"):
        OllamaClient(session=bad_version).preflight(MODEL)

    bad_digest = FakeSession(
        _metadata_responses(tags=[_tag_row(digest="not-a-digest")])[:2]
    )
    with pytest.raises(OllamaProtocolError, match="digest"):
        OllamaClient(session=bad_digest).preflight(MODEL)

    bad_show = FakeSession(
        _metadata_responses(show={"details": {}})
    )
    with pytest.raises(OllamaProtocolError, match="missing"):
        OllamaClient(session=bad_show).preflight(MODEL)


def test_failed_repreflight_clears_previous_identity():
    session = FakeSession(
        [*_metadata_responses(), FakeResponse(payload={"version": ""})]
    )
    client = OllamaClient(session=session)
    assert client.preflight(MODEL).digest == DIGEST

    with pytest.raises(OllamaProtocolError):
        client.preflight(MODEL)
    with pytest.raises(PreflightRequiredError):
        client.generate([{"role": "user", "content": "soru"}], _settings())


def test_generation_settings_freeze_top_level_policy_and_optional_top_k():
    assert _settings().to_ollama_options() == {
        "temperature": 0.0,
        "seed": 20260711,
        "top_p": 1.0,
        "top_k": 1,
        "repeat_penalty": 1.05,
        "num_ctx": 4096,
        "num_predict": 384,
    }
    production = _settings(top_k=None)
    assert "top_k" not in production.to_ollama_options()
    assert production.to_json() == {
        "think": False,
        "stream": False,
        "keep_alive": 0,
        "options": production.to_ollama_options(),
    }


def test_generate_requires_preflight_and_sends_exact_native_chat_request():
    unready = OllamaClient(session=FakeSession([]))
    with pytest.raises(PreflightRequiredError):
        unready.generate([{"role": "user", "content": "soru"}], _settings())

    client, session, _ = _client_with_chat(
        [FakeResponse(payload=_chat_payload())]
    )
    messages = [
        {"role": "system", "content": "Türkçe yanıtla."},
        {"role": "user", "content": "Soru ne?"},
    ]
    result = client.generate(messages, _settings())

    method, url, kwargs = next(
        call for call in session.calls if call[1].endswith("/api/chat")
    )
    assert (method, url, kwargs["timeout"]) == (
        "POST",
        "http://127.0.0.1:11434/api/chat",
        (5, 900),
    )
    assert kwargs["json"] == {
        "model": MODEL,
        "messages": messages,
        "think": False,
        "stream": False,
        "keep_alive": 0,
        "options": _settings().to_ollama_options(),
    }
    assert result.content == "Doğru cevap."
    assert result.done_reason == "stop"
    assert result.eval_count == 7


def test_generation_is_discarded_if_model_identity_changes_before_or_after_chat():
    before_session = FakeSession(
        [
            *_metadata_responses(),
            *_identity_check_responses(tags=[_tag_row(digest="a" * 64)]),
        ]
    )
    before = OllamaClient(session=before_session)
    before.preflight(MODEL)
    with pytest.raises(ModelPolicyError, match="changed"):
        before.generate([{"role": "user", "content": "soru"}], _settings())
    assert _chat_call_count(before_session) == 0

    after_session = FakeSession(
        [
            *_metadata_responses(),
            *_identity_check_responses(),
            FakeResponse(payload=_chat_payload()),
            *_identity_check_responses(tags=[_tag_row(digest="a" * 64)]),
        ]
    )
    after = OllamaClient(session=after_session)
    after.preflight(MODEL)
    with pytest.raises(ModelPolicyError, match="changed"):
        after.generate([{"role": "user", "content": "soru"}], _settings())
    assert _chat_call_count(after_session) == 1


def test_generate_preserves_length_stop_for_later_diagnostics():
    client, _, _ = _client_with_chat(
        [FakeResponse(payload=_chat_payload(done_reason="length", eval_count=384))]
    )
    result = client.generate([{"role": "user", "content": "soru"}], _settings())
    assert result.done_reason == "length"
    assert result.eval_count == 384


def test_generate_preserves_exact_nonblank_answer_text():
    payload = _chat_payload()
    payload["message"] = {
        "role": "assistant",
        "content": "  Doğru cevap. \n",
        "thinking": "",
    }
    client, _, _ = _client_with_chat([FakeResponse(payload=payload)])
    result = client.generate([{"role": "user", "content": "soru"}], _settings())
    assert result.content == "  Doğru cevap. \n"


@pytest.mark.parametrize(
    ("message", "error"),
    [
        ({"role": "assistant", "content": "", "thinking": ""}, EmptyContentError),
        (
            {"role": "assistant", "content": "yanıt", "thinking": "gizli"},
            ThinkingLeakError,
        ),
        (
            {"role": "assistant", "content": "yanıt", "thinking": "   "},
            ThinkingLeakError,
        ),
        (
            {"role": "assistant", "content": "yanıt", "thinking": 7},
            OllamaProtocolError,
        ),
        ({"role": "assistant"}, OllamaProtocolError),
        ({"content": "yanıt", "thinking": ""}, OllamaProtocolError),
        ({"role": "user", "content": "yanıt", "thinking": ""}, OllamaProtocolError),
    ],
)
def test_content_thinking_and_protocol_failures_are_not_retried(message, error):
    client, session, sleeps = _client_with_chat(
        [FakeResponse(payload=_chat_payload(message=message))]
    )
    with pytest.raises(error):
        client.generate([{"role": "user", "content": "soru"}], _settings())
    assert _chat_call_count(session) == 1
    assert sleeps == []


def test_malformed_json_and_response_model_mismatch_are_not_retried():
    client, session, _ = _client_with_chat(
        [FakeResponse(json_error=ValueError("bad json"))]
    )
    with pytest.raises(OllamaProtocolError, match="JSON"):
        client.generate([{"role": "user", "content": "soru"}], _settings())
    assert _chat_call_count(session) == 1

    client, session, _ = _client_with_chat(
        [FakeResponse(payload=_chat_payload(model="qwen3:8b"))]
    )
    with pytest.raises(OllamaProtocolError, match="model"):
        client.generate([{"role": "user", "content": "soru"}], _settings())
    assert _chat_call_count(session) == 1

    missing_model = _chat_payload()
    missing_model.pop("model")
    client, session, _ = _client_with_chat(
        [FakeResponse(payload=missing_model)]
    )
    with pytest.raises(OllamaProtocolError, match="model"):
        client.generate([{"role": "user", "content": "soru"}], _settings())
    assert _chat_call_count(session) == 1


@pytest.mark.parametrize("status", [502, 503, 504])
def test_transient_http_status_retries_twice_then_succeeds(status):
    sleeps = []
    client, session, _ = _client_with_chat(
        [
            FakeResponse(status_code=status),
            FakeResponse(status_code=status),
            FakeResponse(payload=_chat_payload()),
        ],
        sleeps=sleeps,
    )
    assert client.generate(
        [{"role": "user", "content": "soru"}], _settings()
    ).content
    assert _chat_call_count(session) == 3
    assert sleeps == [0.25, 0.5]


@pytest.mark.parametrize(
    "failure",
    [
        requests.ConnectionError("offline"),
        requests.ReadTimeout("slow"),
        requests.exceptions.ChunkedEncodingError("truncated"),
    ],
)
def test_transport_failures_retry_twice_and_exhaust_with_typed_error(failure):
    sleeps = []
    client, session, _ = _client_with_chat(
        [failure, failure, failure], sleeps=sleeps
    )
    with pytest.raises(OllamaTransportError):
        client.generate([{"role": "user", "content": "soru"}], _settings())
    assert _chat_call_count(session) == 3
    assert sleeps == [0.25, 0.5]


def test_third_transient_http_failure_exhausts_after_exactly_three_calls():
    sleeps = []
    client, session, _ = _client_with_chat(
        [
            FakeResponse(status_code=502),
            FakeResponse(status_code=502),
            FakeResponse(status_code=502),
        ],
        sleeps=sleeps,
    )
    with pytest.raises(OllamaHTTPError) as captured:
        client.generate([{"role": "user", "content": "soru"}], _settings())
    assert captured.value.status_code == 502
    assert _chat_call_count(session) == 3
    assert sleeps == [0.25, 0.5]


@pytest.mark.parametrize("status", [400, 404, 429, 500])
def test_non_retryable_http_status_fails_after_one_call(status):
    client, session, sleeps = _client_with_chat(
        [FakeResponse(status_code=status, text="nope")]
    )
    with pytest.raises(OllamaHTTPError) as captured:
        client.generate([{"role": "user", "content": "soru"}], _settings())
    assert captured.value.status_code == status
    assert _chat_call_count(session) == 1
    assert sleeps == []


def test_transient_status_followed_by_bad_json_stops_without_third_attempt():
    sleeps = []
    client, session, _ = _client_with_chat(
        [
            FakeResponse(status_code=502),
            FakeResponse(json_error=ValueError("bad json")),
            FakeResponse(payload=_chat_payload()),
        ],
        sleeps=sleeps,
    )
    with pytest.raises(OllamaProtocolError):
        client.generate([{"role": "user", "content": "soru"}], _settings())
    assert _chat_call_count(session) == 2
    assert sleeps == [0.25]
