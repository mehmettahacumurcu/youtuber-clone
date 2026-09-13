"""Bounded, redirect-denying JSON transport for the local Ollama service."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlsplit

import requests


from runtime.settings import DEVELOPMENT_OLLAMA_ORIGIN, RuntimeSettings, validate_ollama_origin


OLLAMA_ORIGIN = DEVELOPMENT_OLLAMA_ORIGIN
MAX_OLLAMA_JSON_BYTES = 8 * 1024 * 1024
_READ_CHUNK_BYTES = 8 * 1024
_SESSION = requests.Session()
_SESSION.trust_env = False


class OllamaTransportError(RuntimeError):
    """The local Ollama response did not satisfy the bounded transport contract."""


def ollama_json(
    method: str,
    url: str,
    *,
    payload: Mapping[str, object] | None = None,
    timeout_s: float = 120.0,
    request: Callable[..., Any] | None = None,
    max_bytes: int = MAX_OLLAMA_JSON_BYTES,
    expected_origin: str | None = None,
) -> dict[str, object]:
    """Issue one local Ollama request and decode at most ``max_bytes`` of JSON."""
    expected_origin = expected_origin or RuntimeSettings.from_environment().ollama_origin
    _validate_ollama_url(url, expected_origin=expected_origin)
    if method not in {"GET", "POST"}:
        raise ValueError("Ollama method must be GET or POST")
    if not 0 < timeout_s <= 660:
        raise ValueError("Ollama timeout must be in (0, 660]")
    if max_bytes <= 0:
        raise ValueError("Ollama JSON cap must be positive")
    transport = request or _SESSION.request
    response = None
    try:
        response = transport(
            method, url, json=dict(payload) if payload is not None else None,
            timeout=timeout_s, allow_redirects=False, stream=True,
        )
        _validate_response(response, url, max_bytes)
        response.raise_for_status()
        body = _read_body(response, max_bytes)
        value = json.loads(body.decode("utf-8"))
        if type(value) is not dict:
            raise OllamaTransportError("Ollama response must be a JSON object")
        return value
    except OllamaTransportError:
        raise
    except (UnicodeError, ValueError, TypeError) as exc:
        raise OllamaTransportError("Ollama response is not valid JSON") from exc
    except requests.RequestException as exc:
        raise OllamaTransportError(f"Ollama request failed: {exc}") from exc
    finally:
        if response is not None:
            response.close()


def _validate_ollama_url(url: str, *, expected_origin: str = DEVELOPMENT_OLLAMA_ORIGIN) -> None:
    expected_port = validate_ollama_origin(expected_origin)
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or
        parsed.port != expected_port or parsed.username is not None or parsed.password is not None or
        parsed.fragment or f"{parsed.scheme}://{parsed.netloc}" != expected_origin
    ):
        raise ValueError("Ollama URL must use the exact configured Ollama origin")


def _validate_response(response: Any, expected_url: str, max_bytes: int) -> None:
    if 300 <= getattr(response, "status_code", 200) < 400 or getattr(response, "url", expected_url) != expected_url:
        raise OllamaTransportError("Ollama redirect or changed origin was rejected")
    length = getattr(response, "headers", {}).get("Content-Length")
    if type(length) is not str or not length.isascii() or not length.isdecimal() or int(length) > max_bytes:
        raise OllamaTransportError("Ollama Content-Length is missing, invalid, or too large")


def _read_body(response: Any, max_bytes: int) -> bytes:
    body = bytearray()
    for chunk in response.iter_content(chunk_size=_READ_CHUNK_BYTES):
        if not chunk:
            continue
        if len(chunk) > max_bytes - len(body):
            raise OllamaTransportError("Ollama response body is too large")
        body.extend(chunk)
    return bytes(body)
