import pytest
import requests

from rag.ollama_residency import ModelResidencyError, ensure_model_absent
from runtime.ollama_http import OllamaTransportError


class FakeResponse:
    def __init__(self, payload=None, *, status=200):
        self.payload = payload
        self.status = status
        self.status_code = status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"HTTP {self.status}")

    def json(self):
        return self.payload


class RequestRecorder:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def __call__(self, method, url, *, payload=None, timeout_s):
        self.calls.append((method, url, payload, timeout_s))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        if 300 <= response.status < 400:
            raise OllamaTransportError("Ollama redirect or changed origin was rejected")
        response.raise_for_status()
        return response.payload


def _transport(post, get):
    def request_json(method, url, *, payload=None, timeout_s):
        return (post if method == "POST" else get)(method, url, payload=payload, timeout_s=timeout_s)
    return request_json


def test_returns_when_generate_succeeds_and_status_immediately_confirms_absence():
    post = RequestRecorder([FakeResponse({})])
    get = RequestRecorder([FakeResponse({"models": []})])
    sleeps = []

    ensure_model_absent(
        "qwen3:14b",
        base_url="http://127.0.0.1:11434/",
        timeout_s=10.0,
        poll_interval_s=1.0,
        request_json=_transport(post, get),
        clock=lambda: 0.0,
        sleep=sleeps.append,
    )

    assert post.calls == [("POST", "http://127.0.0.1:11434/api/generate", {"model": "qwen3:14b", "keep_alive": 0}, 10.0)]
    assert get.calls == [("GET", "http://127.0.0.1:11434/api/ps", None, 10.0)]
    assert sleeps == []


def test_residency_uses_the_injected_bounded_json_transport_for_unload_and_status():
    calls = []

    def request_json(method, url, *, payload=None, timeout_s):
        calls.append((method, url, payload, timeout_s))
        return {"models": []}

    ensure_model_absent(
        "qwen3:14b", base_url="http://127.0.0.1:11434", timeout_s=10.0,
        poll_interval_s=1.0, request_json=request_json, clock=lambda: 0.0, sleep=lambda _: None,
    )

    assert calls == [
        ("POST", "http://127.0.0.1:11434/api/generate", {"model": "qwen3:14b", "keep_alive": 0}, 10.0),
        ("GET", "http://127.0.0.1:11434/api/ps", None, 10.0),
    ]


def test_polls_after_delay_until_status_confirms_absence():
    post = RequestRecorder([FakeResponse({})])
    get = RequestRecorder([
        FakeResponse({"models": [{"name": "qwen3:14b"}]}),
        FakeResponse({"models": []}),
    ])
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    ensure_model_absent(
        "qwen3:14b",
                base_url="http://127.0.0.1:11434",
        timeout_s=10.0,
        poll_interval_s=2.0,
        request_json=_transport(post, get),
        clock=lambda: now[0],
        sleep=sleep,
    )

    assert len(get.calls) == 2
    assert now == [2.0]


def test_generate_transport_failure_raises_typed_error_without_polling():
    post = RequestRecorder([requests.ConnectionError("down")])
    get = RequestRecorder([])

    with pytest.raises(ModelResidencyError, match="unload request failed"):
        ensure_model_absent(
            "qwen3:14b",
            base_url="http://127.0.0.1:11434",
            timeout_s=10.0,
            poll_interval_s=1.0,
            request_json=_transport(post, get),
            clock=lambda: 0.0,
            sleep=lambda _: None,
        )

    assert get.calls == []


def test_malformed_status_fails_closed():
    post = RequestRecorder([FakeResponse({})])
    get = RequestRecorder([FakeResponse({"models": [{"name": ""}]})])

    with pytest.raises(ModelResidencyError, match="invalid Ollama residency status"):
        ensure_model_absent(
            "qwen3:14b",
            base_url="http://127.0.0.1:11434",
            timeout_s=10.0,
            poll_interval_s=1.0,
            request_json=_transport(post, get),
            clock=lambda: 0.0,
            sleep=lambda _: None,
        )


def test_redirect_response_fails_closed_before_the_status_request():
    post = RequestRecorder([FakeResponse({}, status=307)])
    get = RequestRecorder([])

    with pytest.raises(ModelResidencyError, match="redirect"):
        ensure_model_absent(
            "qwen3:14b",
            base_url="http://127.0.0.1:11434",
            timeout_s=10.0,
            poll_interval_s=1.0,
            request_json=_transport(post, get),
            clock=lambda: 0.0,
            sleep=lambda _: None,
        )

    assert get.calls == []


def test_raises_timeout_when_target_remains_loaded_at_deadline():
    post = RequestRecorder([FakeResponse({})])
    get = RequestRecorder([FakeResponse({"models": [{"name": "qwen3:14b"}]})])
    clock_values = iter([0.0, 0.0, 0.0, 0.0, 1.0])

    with pytest.raises(ModelResidencyError, match="timed out waiting"):
        ensure_model_absent(
            "qwen3:14b",
            base_url="http://127.0.0.1:11434",
            timeout_s=1.0,
            poll_interval_s=0.25,
            request_json=_transport(post, get),
            clock=lambda: next(clock_values),
            sleep=lambda _: pytest.fail("must not sleep after the deadline"),
        )


def test_model_identifier_also_prevents_absence_confirmation():
    post = RequestRecorder([FakeResponse({})])
    get = RequestRecorder([
        FakeResponse({"models": [{"name": "display-name", "model": "qwen3:14b"}]}),
    ])
    clock_values = iter([0.0, 0.0, 0.0, 0.0, 1.0])

    with pytest.raises(ModelResidencyError, match="timed out waiting"):
        ensure_model_absent(
            "qwen3:14b",
            base_url="http://127.0.0.1:11434",
            timeout_s=1.0,
            poll_interval_s=0.25,
            request_json=_transport(post, get),
            clock=lambda: next(clock_values),
            sleep=lambda _: pytest.fail("must not sleep after the deadline"),
        )


def test_does_not_return_absent_when_deadline_expires_during_status_request():
    now = [0.0]
    request_timeouts = []

    def request_json(method, url, *, payload=None, timeout_s):
        request_timeouts.append((url, timeout_s))
        now[0] += 0.25
        if method == "POST":
            return {}
        now[0] += 0.5
        return {"models": []}

    with pytest.raises(ModelResidencyError, match="timed out waiting"):
        ensure_model_absent(
            "qwen3:14b",
            base_url="http://127.0.0.1:11434",
            timeout_s=1.0,
            poll_interval_s=0.25,
            request_json=request_json,
            clock=lambda: now[0],
            sleep=lambda _: pytest.fail("no sleep is allowed after expiration"),
        )

    assert request_timeouts == [
        ("http://127.0.0.1:11434/api/generate", 1.0),
        ("http://127.0.0.1:11434/api/ps", 0.75),
    ]


@pytest.mark.parametrize(
    ("timeout_s", "poll_interval_s"),
    [
        (0.0, 1.0),
        (-1.0, 1.0),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
        (1.0, float("nan")),
        (1.0, float("inf")),
    ],
)
def test_rejects_nonfinite_or_nonpositive_time_settings(timeout_s, poll_interval_s):
    with pytest.raises(ValueError):
        ensure_model_absent(
            "qwen3:14b",
            base_url="http://127.0.0.1:11434",
            timeout_s=timeout_s,
            poll_interval_s=poll_interval_s,
            request_json=lambda *args, **kwargs: pytest.fail("invalid inputs must not issue requests"),
            clock=lambda: 0.0,
            sleep=lambda _: pytest.fail("invalid inputs must not sleep"),
        )
