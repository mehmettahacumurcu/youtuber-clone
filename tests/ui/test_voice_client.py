from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest
import soundfile as sf

import ui.voice_client as voice_client
from ui.voice_client import VoiceClient, VoiceClientError


def _wav() -> bytes:
    stream = io.BytesIO()
    sf.write(stream, np.zeros(24_000, dtype=np.float32), 24_000, format="WAV")
    return stream.getvalue()


def test_voice_client_requires_a_loopback_url():
    with pytest.raises(ValueError, match="loopback"):
        VoiceClient("http://example.test:7862", "s" * 64)


def test_voice_client_default_transport_bypasses_proxy_environment(monkeypatch):
    data = _wav()
    seen: dict[str, object] = {}
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:8080")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    class Response:
        status_code = 200
        headers = {
            "content-type": "audio/wav",
            "content-length": str(len(data)),
            "x-youtuber-sha256": hashlib.sha256(data).hexdigest(),
            "x-youtuber-sample-rate": "24000",
            "x-youtuber-duration-seconds": "1.000000",
            "x-youtuber-rvc-epoch": "200",
            "x-youtuber-index-rate": "0.75",
            "x-youtuber-request-id": "req-direct",
        }

        def iter_content(self, chunk_size):
            yield data

        def close(self):
            return None

    def direct_post(url, **kwargs):
        seen.update(url=url, authorization=kwargs["headers"]["Authorization"])
        return Response()

    monkeypatch.setattr(voice_client._SESSION, "post", direct_post)

    result = VoiceClient("http://127.0.0.1:7862", "s" * 64).synthesize(
        "Merhaba", "req-direct"
    )

    assert voice_client._SESSION.trust_env is False
    assert seen == {
        "url": "http://127.0.0.1:7862/v1/synthesize",
        "authorization": "Bearer " + "s" * 64,
    }
    assert result.wav_bytes == data


def test_voice_client_uses_the_proven_660_second_synthesis_bound_by_default():
    data = _wav()
    seen: dict[str, object] = {}

    class Response:
        status_code = 200
        headers = {
            "content-type": "audio/wav", "x-youtuber-sha256": hashlib.sha256(data).hexdigest(),
            "x-youtuber-sample-rate": "24000", "x-youtuber-duration-seconds": "1.000000",
            "x-youtuber-rvc-epoch": "200", "x-youtuber-index-rate": "0.75", "x-youtuber-request-id": "req-timeout", "content-length": str(len(data)),
        }

        def iter_content(self, chunk_size):
            yield data

        def close(self):
            return None

    def post(*args, **kwargs):
        seen["timeout"] = kwargs["timeout"]
        return Response()

    VoiceClient("http://127.0.0.1:7862", "s" * 64, post=post).synthesize(
        "Merhaba", "req-timeout"
    )

    assert seen["timeout"] == (3.0, 660.0)


def test_voice_client_accepts_660_seconds_but_rejects_a_larger_timeout():
    VoiceClient("http://127.0.0.1:7862", "s" * 64, timeout_seconds=660.0)

    with pytest.raises(ValueError, match="660"):
        VoiceClient("http://127.0.0.1:7862", "s" * 64, timeout_seconds=660.001)


def test_voice_client_validates_wav_and_integrity_headers():
    data = _wav()

    class Response:
        status_code = 200
        headers = {
            "content-type": "audio/wav", "x-youtuber-sha256": hashlib.sha256(data).hexdigest(),
            "x-youtuber-sample-rate": "24000", "x-youtuber-duration-seconds": "1.000000",
            "x-youtuber-rvc-epoch": "200", "x-youtuber-index-rate": "0.75", "x-youtuber-request-id": "req-1", "content-length": str(len(data)),
        }

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            yield data

        def close(self):
            return None

    audio = VoiceClient("http://127.0.0.1:7862", "s" * 64, post=lambda *args, **kwargs: Response()).synthesize("Merhaba", "req-1")
    assert audio.sample_rate == 24_000
    assert audio.wav_bytes == data


def test_voice_client_rejects_redirects_and_header_mismatches():
    class Redirect:
        status_code = 302
        content = b""
        headers = {}

    with pytest.raises(VoiceClientError, match="redirect"):
        VoiceClient("http://127.0.0.1:7862", "s" * 64, post=lambda *args, **kwargs: Redirect()).synthesize("Merhaba", "req")


def test_voice_client_rejects_non_wav_payloads_even_with_matching_headers():
    stream = io.BytesIO()
    sf.write(stream, np.zeros(24_000, dtype=np.float32), 24_000, format="FLAC")
    data = stream.getvalue()

    class Response:
        status_code = 200
        headers = {
            "content-type": "audio/wav", "x-youtuber-sha256": hashlib.sha256(data).hexdigest(),
            "x-youtuber-sample-rate": "24000", "x-youtuber-duration-seconds": "1.000000",
            "x-youtuber-rvc-epoch": "200", "x-youtuber-index-rate": "0.75", "x-youtuber-request-id": "req-1", "content-length": str(len(data)),
        }

        def iter_content(self, chunk_size):
            yield data

    with pytest.raises(VoiceClientError, match="invalid WAV"):
        VoiceClient("http://127.0.0.1:7862", "s" * 64, post=lambda *args, **kwargs: Response()).synthesize("Merhaba", "req-1")


def test_voice_client_streams_a_normal_response_without_reading_content_property():
    data = _wav()

    class Response:
        status_code = 200
        headers = {
            "content-type": "audio/wav", "x-youtuber-sha256": hashlib.sha256(data).hexdigest(),
            "x-youtuber-sample-rate": "24000", "x-youtuber-duration-seconds": "1.000000",
            "x-youtuber-rvc-epoch": "200", "x-youtuber-index-rate": "0.75", "x-youtuber-request-id": "req-1", "content-length": str(len(data)),
        }

        @property
        def content(self):
            raise AssertionError("client must not eagerly read response.content")

        def iter_content(self, chunk_size):
            yield data[:100]
            yield data[100:]

        def close(self):
            return None

    result = VoiceClient("http://127.0.0.1:7862", "s" * 64, post=lambda *args, **kwargs: Response()).synthesize("Merhaba", "req-1")
    assert result.wav_bytes == data


def test_voice_client_rejects_declared_oversized_body_before_reading(monkeypatch):
    class Response:
        status_code = 200
        headers = {"content-length": str(129 * 1024 * 1024)}

        @property
        def content(self):
            raise AssertionError("oversized content must not be read")

    with pytest.raises(VoiceClientError, match="content length"):
        VoiceClient("http://127.0.0.1:7862", "s" * 64, post=lambda *args, **kwargs: Response()).synthesize("Merhaba", "req-1")


def test_voice_client_rejects_stream_larger_than_the_declared_length():
    data = _wav()

    class Response:
        status_code = 200
        headers = {"content-type": "audio/wav", "content-length": str(len(data) - 1)}

        def iter_content(self, chunk_size):
            yield data

    with pytest.raises(VoiceClientError, match="content length"):
        VoiceClient("http://127.0.0.1:7862", "s" * 64, post=lambda *args, **kwargs: Response()).synthesize("Merhaba", "req-1")


def test_voice_client_rejects_truncated_stream_that_disagrees_with_content_length():
    data = _wav()

    class Response:
        status_code = 200
        headers = {"content-type": "audio/wav", "content-length": str(len(data) + 1)}

        def iter_content(self, chunk_size):
            yield data

    with pytest.raises(VoiceClientError, match="content length"):
        VoiceClient("http://127.0.0.1:7862", "s" * 64, post=lambda *args, **kwargs: Response()).synthesize("Merhaba", "req-1")


def test_voice_client_caps_actual_stream_bytes_when_length_lies(monkeypatch):
    data = _wav()
    monkeypatch.setattr(voice_client, "MAX_RESPONSE_BYTES", len(data) - 1, raising=False)

    class Response:
        status_code = 200
        headers = {"content-type": "audio/wav", "content-length": str(len(data) - 1)}

        def iter_content(self, chunk_size):
            yield data

    with pytest.raises(VoiceClientError, match="too large"):
        VoiceClient("http://127.0.0.1:7862", "s" * 64, post=lambda *args, **kwargs: Response()).synthesize("Merhaba", "req-1")


def test_voice_client_translates_nonfinite_duration_metadata_to_safe_error():
    data = _wav()

    class Response:
        status_code = 200
        headers = {
            "content-type": "audio/wav", "x-youtuber-sha256": hashlib.sha256(data).hexdigest(),
            "x-youtuber-sample-rate": "24000", "x-youtuber-duration-seconds": "nan",
            "x-youtuber-rvc-epoch": "200", "x-youtuber-index-rate": "0.75", "x-youtuber-request-id": "req-1", "content-length": str(len(data)),
        }

        def iter_content(self, chunk_size):
            yield data

    with pytest.raises(VoiceClientError, match="metadata"):
        VoiceClient("http://127.0.0.1:7862", "s" * 64, post=lambda *args, **kwargs: Response()).synthesize("Merhaba", "req-1")


def test_voice_client_closes_a_streamed_response_once_after_success():
    data = _wav()

    class Response:
        status_code = 200
        closed = 0
        headers = {
            "content-type": "audio/wav", "x-youtuber-sha256": hashlib.sha256(data).hexdigest(),
            "x-youtuber-sample-rate": "24000", "x-youtuber-duration-seconds": "1.000000",
            "x-youtuber-rvc-epoch": "200", "x-youtuber-index-rate": "0.75", "x-youtuber-request-id": "req-1", "content-length": str(len(data)),
        }

        def iter_content(self, chunk_size):
            yield data

        def close(self):
            self.closed += 1

    response = Response()
    assert VoiceClient("http://127.0.0.1:7862", "s" * 64, post=lambda *args, **kwargs: response).synthesize("Merhaba", "req-1").wav_bytes == data
    assert response.closed == 1


def test_voice_client_closes_once_and_preserves_primary_rejection_when_close_fails():
    class Response:
        status_code = 200
        closed = 0
        headers = {"content-length": "0"}

        def close(self):
            self.closed += 1
            raise RuntimeError("close failed")

    response = Response()
    with pytest.raises(VoiceClientError, match="content length"):
        VoiceClient("http://127.0.0.1:7862", "s" * 64, post=lambda *args, **kwargs: response).synthesize("Merhaba", "req-1")
    assert response.closed == 1


def test_voice_client_normalizes_close_failure_after_otherwise_valid_response():
    data = _wav()

    class Response:
        status_code = 200
        headers = {
            "content-type": "audio/wav", "x-youtuber-sha256": hashlib.sha256(data).hexdigest(),
            "x-youtuber-sample-rate": "24000", "x-youtuber-duration-seconds": "1.000000",
            "x-youtuber-rvc-epoch": "200", "x-youtuber-index-rate": "0.75", "x-youtuber-request-id": "req-1", "content-length": str(len(data)),
        }

        def iter_content(self, chunk_size):
            yield data

        def close(self):
            raise RuntimeError("close failed")

    with pytest.raises(VoiceClientError, match="close"):
        VoiceClient("http://127.0.0.1:7862", "s" * 64, post=lambda *args, **kwargs: Response()).synthesize("Merhaba", "req-1")


@pytest.mark.parametrize("headers", [None, object()])
def test_voice_client_normalizes_malformed_header_containers_and_closes(headers):
    class Response:
        status_code = 200
        closed = 0

        def __init__(self):
            self.headers = headers

        def close(self):
            self.closed += 1

    response = Response()
    with pytest.raises(VoiceClientError, match="headers"):
        VoiceClient("http://127.0.0.1:7862", "s" * 64, post=lambda *args, **kwargs: response).synthesize("Merhaba", "req-1")
    assert response.closed == 1
