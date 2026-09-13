"""Typed, loopback-only client for the isolated voice worker."""
from __future__ import annotations

import hashlib
import io
import ipaddress
import math
from collections.abc import Callable, Mapping
from urllib.parse import urlparse

import numpy as np
import requests
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from voice.api_models import (
    DEFAULT_VOICE_SYNTHESIS_TIMEOUT_SECONDS,
    MAX_VOICE_SYNTHESIS_TIMEOUT_SECONDS,
    SynthesisRequest,
)


# Bound streamed audio independently from the accepted text length.
MAX_RESPONSE_BYTES = 128 * 1024 * 1024
MAX_AUDIO_DURATION_SECONDS = 20 * 60
MAX_AUDIO_SAMPLES = 96_000 * MAX_AUDIO_DURATION_SECONDS
_STREAM_CHUNK_BYTES = 64 * 1024
_SESSION = requests.Session()
_SESSION.trust_env = False

class VoiceClientError(RuntimeError):
    """Safe error for the Studio UI; service internals never cross this boundary."""


class VoiceAudio(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    wav_bytes: bytes
    sample_rate: StrictInt = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_id: str


class VoiceClient:
    def __init__(
        self,
        base_url: str,
        session_secret: str,
        *,
        timeout_seconds: float = DEFAULT_VOICE_SYNTHESIS_TIMEOUT_SECONDS,
        post: Callable[..., object] | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        try:
            loopback = parsed.hostname is not None and ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            loopback = False
        if parsed.scheme != "http" or not parsed.netloc or parsed.username or parsed.password or not loopback:
            raise ValueError("voice URL must be an absolute loopback HTTP URL")
        if not session_secret:
            raise ValueError("voice client requires a session secret")
        if not 0 < timeout_seconds <= MAX_VOICE_SYNTHESIS_TIMEOUT_SECONDS:
            raise ValueError(
                "voice timeout must be between zero and "
                f"{MAX_VOICE_SYNTHESIS_TIMEOUT_SECONDS:g} seconds"
            )
        self._base_url = base_url.rstrip("/")
        self._session_secret = session_secret
        self._timeout_seconds = timeout_seconds
        self._post = post if post is not None else _SESSION.post

    def synthesize(self, text: str, request_id: str) -> VoiceAudio:
        try:
            request = SynthesisRequest(text=text, request_id=request_id)
        except Exception as error:
            raise VoiceClientError("voice request is invalid") from error
        response: object | None = None
        primary_error: BaseException | None = None
        try:
            response = self._post(
                self._base_url + "/v1/synthesize",
                json=request.model_dump(),
                headers={"Authorization": "Bearer " + self._session_secret},
                timeout=(3.0, self._timeout_seconds),
                allow_redirects=False,
                stream=True,
            )
            return self._parse_response(response, request_id)
        except requests.RequestException as error:
            primary_error = VoiceClientError("voice worker is unavailable")
            raise primary_error from error
        except VoiceClientError as error:
            primary_error = error
            raise
        except Exception as error:
            primary_error = VoiceClientError("voice worker returned invalid response")
            raise primary_error from error
        finally:
            if response is not None:
                try:
                    close = getattr(response, "close", None)
                    if not callable(close):
                        raise RuntimeError("response is not closable")
                    close()
                except Exception as error:
                    if primary_error is None:
                        raise VoiceClientError("voice worker response close failed") from error

    def _parse_response(self, response: object, request_id: str) -> VoiceAudio:
        status_code = getattr(response, "status_code", 0)
        if not isinstance(status_code, int):
            raise VoiceClientError("voice worker returned invalid status")
        if 300 <= status_code < 400:
            raise VoiceClientError("voice worker returned a redirect")
        if status_code != 200:
            raise VoiceClientError("voice worker rejected synthesis")
        headers = self._headers(response)
        declared_length = self._content_length(headers)
        if not str(headers.get("content-type", "")).lower().startswith("audio/wav"):
            raise VoiceClientError("voice worker returned an invalid content type")
        data = self._read_stream(response, declared_length)
        digest = hashlib.sha256(data).hexdigest()
        if headers.get("x-youtuber-sha256") != digest:
            raise VoiceClientError("voice worker audio integrity check failed")
        if headers.get("x-youtuber-request-id") != request_id:
            raise VoiceClientError("voice worker returned a mismatched request")
        try:
            sample_rate = int(headers["x-youtuber-sample-rate"])
            duration = float(headers["x-youtuber-duration-seconds"])
            if headers["x-youtuber-rvc-epoch"] != "200" or headers["x-youtuber-index-rate"] != "0.75":
                raise ValueError
            if not math.isfinite(duration) or not 0 < duration <= MAX_AUDIO_DURATION_SECONDS:
                raise ValueError
            stream = io.BytesIO(data)
            info = sf.info(stream)
            if info.format != "WAV" or info.frames <= 0 or info.frames > MAX_AUDIO_SAMPLES:
                raise VoiceClientError("voice worker returned invalid WAV audio")
            stream.seek(0)
            samples, actual_rate = sf.read(stream, dtype="float32", always_2d=True)
            waveform = np.asarray(samples, dtype=np.float32)
            if actual_rate != sample_rate or waveform.ndim != 2 or waveform.shape[1] != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
                raise VoiceClientError("voice worker returned invalid WAV audio")
            actual_duration = waveform.shape[0] / sample_rate
            if abs(actual_duration - duration) > 0.001:
                raise VoiceClientError("voice worker returned inconsistent duration")
            return VoiceAudio(
                wav_bytes=data, sample_rate=sample_rate, duration_seconds=duration,
                sha256=digest, request_id=request_id,
            )
        except VoiceClientError:
            raise
        except Exception as error:
            raise VoiceClientError("voice worker returned invalid audio metadata") from error

    @staticmethod
    def _headers(response: object) -> Mapping[str, object]:
        headers = getattr(response, "headers", None)
        if not isinstance(headers, Mapping) or not callable(getattr(headers, "get", None)):
            raise VoiceClientError("voice worker returned invalid headers")
        return headers

    @staticmethod
    def _content_length(headers: Mapping[str, object]) -> int:
        try:
            value = headers.get("content-length")
        except Exception as error:
            raise VoiceClientError("voice worker returned invalid headers") from error
        if not isinstance(value, str) or not value or not value.isascii() or not value.isdecimal():
            raise VoiceClientError("voice worker returned invalid content length")
        length = int(value)
        if length <= 0 or length > MAX_RESPONSE_BYTES:
            raise VoiceClientError("voice worker returned invalid content length")
        return length

    @staticmethod
    def _read_stream(response: object, declared_length: int) -> bytes:
        iterator = getattr(response, "iter_content", None)
        if not callable(iterator):
            raise VoiceClientError("voice worker did not stream audio")
        body = bytearray()
        try:
            for chunk in iterator(_STREAM_CHUNK_BYTES):
                if not isinstance(chunk, (bytes, bytearray)):
                    raise VoiceClientError("voice worker returned invalid audio bytes")
                if not chunk:
                    continue
                if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                    raise VoiceClientError("voice worker response is too large")
                if len(body) + len(chunk) > declared_length:
                    raise VoiceClientError("voice worker content length disagrees with stream")
                body.extend(chunk)
        except VoiceClientError:
            raise
        except Exception as error:
            raise VoiceClientError("voice worker stream failed") from error
        if len(body) != declared_length:
            raise VoiceClientError("voice worker content length disagrees with stream")
        return bytes(body)
