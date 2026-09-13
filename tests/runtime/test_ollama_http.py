from __future__ import annotations

import json

import pytest

from runtime.ollama_http import OllamaTransportError, ollama_json


class FakeResponse:
    def __init__(self, chunks, *, content_length="2", status_code=200, url="http://127.0.0.1:11434/api/chat"):
        self.headers = {"Content-Length": content_length} if content_length is not None else {}
        self.chunks = chunks
        self.status_code = status_code
        self.url = url
        self.closed = 0
        self.read_sizes = []

    def iter_content(self, chunk_size):
        self.read_sizes.append(chunk_size)
        yield from self.chunks

    def close(self):
        self.closed += 1

    def raise_for_status(self):
        return None


def test_ollama_transport_streams_bounded_json_and_closes_once():
    response = FakeResponse([b'{"ok":true}'], content_length="11")
    calls = []

    def request(*args, **kwargs):
        calls.append((args, kwargs))
        return response

    assert ollama_json("POST", "http://127.0.0.1:11434/api/chat", payload={"q": "x"}, request=request) == {"ok": True}
    assert calls == [
        (("POST", "http://127.0.0.1:11434/api/chat"), {
            "json": {"q": "x"}, "timeout": 120.0, "allow_redirects": False, "stream": True,
        }),
    ]
    assert response.read_sizes == [8192]
    assert response.closed == 1


def test_ollama_transport_uses_the_exact_launcher_origin_and_rejects_a_default_port_decoy():
    origin = "http://127.0.0.1:49155"
    target = origin + "/api/chat"
    response = FakeResponse([b'{"ok":true}'], content_length="11", url=target)

    assert ollama_json(
        "POST", target, expected_origin=origin, request=lambda *_args, **_kwargs: response
    ) == {"ok": True}

    with pytest.raises(ValueError, match="configured Ollama origin"):
        ollama_json(
            "POST",
            "http://127.0.0.1:11434/api/chat",
            expected_origin=origin,
            request=lambda *_args, **_kwargs: response,
        )


def test_ollama_transport_rejects_a_declared_gigabyte_without_reading_or_leaking_response():
    response = FakeResponse([b"unused"], content_length=str(1024**3))

    with pytest.raises(OllamaTransportError, match="Content-Length"):
        ollama_json("POST", "http://127.0.0.1:11434/api/chat", request=lambda *_args, **_kwargs: response)

    assert response.read_sizes == []
    assert response.closed == 1


def test_ollama_transport_rejects_chunked_body_that_exceeds_the_actual_cap():
    response = FakeResponse([b"x" * (8 * 1024 * 1024 + 1)], content_length="1")

    with pytest.raises(OllamaTransportError, match="too large"):
        ollama_json("POST", "http://127.0.0.1:11434/api/chat", request=lambda *_args, **_kwargs: response)

    assert response.closed == 1


def test_ollama_transport_rejects_a_nonconforming_huge_chunk_before_copying_it():
    class OversizedChunk(bytes):
        def __len__(self):
            return 8 * 1024 * 1024 + 1

    response = FakeResponse([OversizedChunk(b"x")], content_length="1")

    with pytest.raises(OllamaTransportError, match="too large"):
        ollama_json("POST", "http://127.0.0.1:11434/api/chat", request=lambda *_args, **_kwargs: response)

    assert response.closed == 1


@pytest.mark.parametrize("content_length", [None, "-1", "nope"])
def test_ollama_transport_rejects_missing_or_invalid_content_length(content_length):
    response = FakeResponse([json.dumps({"ok": True}).encode()], content_length=content_length)

    with pytest.raises(OllamaTransportError, match="Content-Length"):
        ollama_json("POST", "http://127.0.0.1:11434/api/chat", request=lambda *_args, **_kwargs: response)

    assert response.closed == 1
