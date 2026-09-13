"""Strict Ollama metadata and generation client for official stock Qwen3."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import requests


REQUIRED_MODEL = "qwen3:14b"
OFFICIAL_MODEL_DIGEST = (
    "bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8"
)
OFFICIAL_MODEL_METADATA = {
    "format": "gguf",
    "family": "qwen3",
    "parameter_size": "14.8B",
    "quantization_level": "Q4_K_M",
}
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
PREFLIGHT_TIMEOUT = (5, 30)
CHAT_TIMEOUT = (5, 900)
TRANSIENT_STATUSES = frozenset({502, 503, 504})
BACKOFF_SECONDS = (0.25, 0.5)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class OllamaError(RuntimeError):
    """Base class for typed harness failures."""


class ModelPolicyError(OllamaError):
    """The requested or installed model violates the frozen model policy."""


class PreflightRequiredError(OllamaError):
    """Generation was attempted before exact model identity was established."""


class OllamaTransportError(OllamaError):
    """A connection or timeout failure exhausted its retry budget."""


class OllamaHTTPError(OllamaError):
    """Ollama returned a non-success HTTP status."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class OllamaProtocolError(OllamaError):
    """Ollama returned malformed or identity-inconsistent data."""


class EmptyContentError(OllamaError):
    """A completed generation contained no answer text."""


class ThinkingLeakError(OllamaError):
    """A non-thinking request returned private reasoning content."""

    def __init__(self, message: str, *, content: str = "", thinking: str = ""):
        super().__init__(message)
        self.content = content
        self.thinking = thinking


@dataclass(frozen=True)
class ModelIdentity:
    name: str
    digest: str
    ollama_version: str
    format: str
    family: str
    parameter_size: str
    quantization_level: str
    file_type: int | None = None
    quantization_version: int | None = None

    def __post_init__(self) -> None:
        if self.name != REQUIRED_MODEL or "abliterat" in self.name.casefold():
            raise ModelPolicyError(f"model identity must be exactly {REQUIRED_MODEL}")
        if _SHA256_RE.fullmatch(self.digest) is None:
            raise OllamaProtocolError("model digest is not lowercase SHA-256")
        for field_name in (
            "ollama_version",
            "format",
            "family",
            "parameter_size",
            "quantization_level",
        ):
            value = getattr(self, field_name)
            if type(value) is not str or not value.strip():
                raise OllamaProtocolError(f"model {field_name} is missing")
        for field_name in ("file_type", "quantization_version"):
            value = getattr(self, field_name)
            if value is not None and (type(value) is not int or value < 0):
                raise OllamaProtocolError(f"model {field_name} must be a non-negative integer")

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "digest": self.digest,
            "ollama_version": self.ollama_version,
            "format": self.format,
            "family": self.family,
            "parameter_size": self.parameter_size,
            "quantization_level": self.quantization_level,
            "file_type": self.file_type,
            "quantization_version": self.quantization_version,
        }


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float
    seed: int
    top_p: float
    repeat_penalty: float
    num_ctx: int
    num_predict: int
    top_k: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("temperature", "top_p", "repeat_penalty"):
            value = getattr(self, field_name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repeat_penalty <= 0:
            raise ValueError("repeat_penalty must be positive")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        for field_name in ("num_ctx", "num_predict"):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.top_k is not None and (
            type(self.top_k) is not int or self.top_k <= 0
        ):
            raise ValueError("top_k must be None or a positive integer")

    def to_ollama_options(self) -> dict[str, int | float]:
        options: dict[str, int | float] = {
            "temperature": self.temperature,
            "seed": self.seed,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }
        if self.top_k is not None:
            options["top_k"] = self.top_k
        return options

    def to_json(self) -> dict[str, object]:
        return {
            "think": False,
            "stream": False,
            "keep_alive": 0,
            "options": self.to_ollama_options(),
        }


@dataclass(frozen=True)
class GenerationResult:
    content: str
    model: str
    done: bool
    done_reason: str | None = None
    created_at: str | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "content": self.content,
            "model": self.model,
            "done": self.done,
            "done_reason": self.done_reason,
            "created_at": self.created_at,
            "total_duration": self.total_duration,
            "load_duration": self.load_duration,
            "prompt_eval_count": self.prompt_eval_count,
            "prompt_eval_duration": self.prompt_eval_duration,
            "eval_count": self.eval_count,
            "eval_duration": self.eval_duration,
        }


class OllamaClient:
    """Serial, retry-bounded client tied to one preflighted stock model."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        session: Any | None = None,
        sleep: Callable[[float], object] = time.sleep,
    ) -> None:
        if type(base_url) is not str or not base_url.strip():
            raise ValueError("base_url must be non-empty")
        self.base_url = base_url.rstrip("/")
        if session is None:
            owned_session = requests.Session()
            owned_session.trust_env = False
            self.session = owned_session
        else:
            self.session = session
        self.sleep = sleep
        self.identity: ModelIdentity | None = None

    def preflight(self, model: str) -> ModelIdentity:
        """Resolve and freeze exact local model, digest, quantization, and version."""

        self.identity = None
        if model != REQUIRED_MODEL or "abliterat" in model.casefold():
            raise ModelPolicyError(
                f"requested model must be exactly official stock {REQUIRED_MODEL}"
            )

        version_payload = self._request_json(
            "GET", "/api/version", timeout=PREFLIGHT_TIMEOUT
        )
        version = version_payload.get("version")
        if type(version) is not str or not version.strip():
            raise OllamaProtocolError("Ollama version response is malformed")

        tags_payload = self._request_json(
            "GET", "/api/tags", timeout=PREFLIGHT_TIMEOUT
        )
        models = tags_payload.get("models")
        if type(models) is not list or any(type(row) is not dict for row in models):
            raise OllamaProtocolError("Ollama tags response has malformed models")
        exact_rows = [row for row in models if row.get("name") == model]
        if not exact_rows:
            raise ModelPolicyError(f"exact stock model {model} is not installed")
        if len(exact_rows) != 1:
            raise OllamaProtocolError(f"duplicate exact tag rows for model {model}")
        tag = exact_rows[0]
        if "model" in tag and tag["model"] != model:
            raise OllamaProtocolError("tag name/model identity mismatch")
        digest = tag.get("digest")
        if type(digest) is not str or _SHA256_RE.fullmatch(digest) is None:
            raise OllamaProtocolError("model tag digest is not lowercase SHA-256")
        if digest != OFFICIAL_MODEL_DIGEST:
            raise ModelPolicyError(
                f"{model} does not match the frozen official digest"
            )
        tag_details = tag.get("details")
        if type(tag_details) is not dict:
            raise OllamaProtocolError("model tag has malformed details")

        show_payload = self._request_json(
            "POST",
            "/api/show",
            json_body={"model": model, "verbose": False},
            timeout=PREFLIGHT_TIMEOUT,
        )
        details = show_payload.get("details")
        if type(details) is not dict:
            raise OllamaProtocolError("model show response has malformed details")
        format_name = _required_string(details, "format", "model format")
        family = _required_string(details, "family", "model family")
        parameter_size = _required_string(
            details, "parameter_size", "model parameter_size"
        )
        quantization_level = _required_string(
            details, "quantization_level", "model quantization_level"
        )
        observed_metadata = {
            "format": format_name,
            "family": family,
            "parameter_size": parameter_size,
            "quantization_level": quantization_level,
        }
        if observed_metadata != OFFICIAL_MODEL_METADATA:
            raise ModelPolicyError(
                "model show metadata does not match frozen official metadata"
            )
        tag_metadata = {
            key: _required_string(tag_details, key, f"tag {key}")
            for key in OFFICIAL_MODEL_METADATA
        }
        if tag_metadata != observed_metadata:
            raise OllamaProtocolError("tag and show model metadata do not match")
        model_info = show_payload.get("model_info", {})
        if type(model_info) is not dict:
            raise OllamaProtocolError("model show response has malformed model_info")
        file_type = _optional_nonnegative_int(model_info, "general.file_type")
        quantization_version = _optional_nonnegative_int(
            model_info, "general.quantization_version"
        )

        identity = ModelIdentity(
            name=model,
            digest=digest,
            ollama_version=version,
            format=format_name,
            family=family,
            parameter_size=parameter_size,
            quantization_level=quantization_level,
            file_type=file_type,
            quantization_version=quantization_version,
        )
        self.identity = identity
        return identity

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        settings: GenerationSettings,
    ) -> GenerationResult:
        """Generate once with native non-thinking chat and strict response parsing."""

        if self.identity is None:
            raise PreflightRequiredError("preflight must succeed before generation")
        if not isinstance(settings, GenerationSettings):
            raise TypeError("settings must be GenerationSettings")
        self._assert_identity_current()
        normalized_messages = _validate_messages(messages)
        body = {
            "model": self.identity.name,
            "messages": normalized_messages,
            "think": False,
            "stream": False,
            "keep_alive": 0,
            "options": settings.to_ollama_options(),
        }
        payload = self._request_json(
            "POST",
            "/api/chat",
            json_body=body,
            timeout=CHAT_TIMEOUT,
        )
        self._assert_identity_current()
        return self._parse_generation(payload)

    def _assert_identity_current(self) -> None:
        if self.identity is None:
            raise PreflightRequiredError("preflight must succeed before generation")
        version_payload = self._request_json(
            "GET", "/api/version", timeout=PREFLIGHT_TIMEOUT
        )
        if version_payload.get("version") != self.identity.ollama_version:
            raise ModelPolicyError("Ollama/model identity changed since preflight")
        tags_payload = self._request_json(
            "GET", "/api/tags", timeout=PREFLIGHT_TIMEOUT
        )
        models = tags_payload.get("models")
        if type(models) is not list or any(type(row) is not dict for row in models):
            raise OllamaProtocolError("Ollama tags response has malformed models")
        exact_rows = [row for row in models if row.get("name") == self.identity.name]
        if len(exact_rows) != 1:
            raise ModelPolicyError("Ollama/model identity changed since preflight")
        tag = exact_rows[0]
        details = tag.get("details")
        if type(details) is not dict:
            raise OllamaProtocolError("current model tag has malformed details")
        current_metadata = {
            key: details.get(key) for key in OFFICIAL_MODEL_METADATA
        }
        if (
            tag.get("digest") != self.identity.digest
            or tag.get("model", self.identity.name) != self.identity.name
            or current_metadata != OFFICIAL_MODEL_METADATA
        ):
            raise ModelPolicyError("Ollama/model identity changed since preflight")

    def _parse_generation(self, payload: dict[str, Any]) -> GenerationResult:
        assert self.identity is not None
        message = payload.get("message")
        if type(message) is not dict:
            raise OllamaProtocolError("chat response message is missing or malformed")
        if message.get("role") != "assistant":
            raise OllamaProtocolError("chat response role must be assistant")
        content = message.get("content")
        if type(content) is not str:
            raise OllamaProtocolError("chat response content is missing or malformed")
        thinking = message.get("thinking")
        if thinking is not None and type(thinking) is not str:
            raise OllamaProtocolError("chat response thinking field is malformed")
        if isinstance(thinking, str) and thinking:
            raise ThinkingLeakError(
                "native non-thinking request returned reasoning",
                content=content,
                thinking=thinking,
            )
        if not content.strip():
            raise EmptyContentError("chat response content is empty")

        done = payload.get("done")
        if done is not True:
            raise OllamaProtocolError("non-stream chat response must set done=true")
        response_model = payload.get("model")
        if type(response_model) is not str or response_model != self.identity.name:
            raise OllamaProtocolError("chat response model does not match preflight identity")
        done_reason = payload.get("done_reason")
        if done_reason is not None and type(done_reason) is not str:
            raise OllamaProtocolError("chat response done_reason is malformed")
        created_at = payload.get("created_at")
        if created_at is not None and type(created_at) is not str:
            raise OllamaProtocolError("chat response created_at is malformed")

        return GenerationResult(
            content=content,
            model=response_model,
            done=True,
            done_reason=done_reason,
            created_at=created_at,
            total_duration=_optional_nonnegative_int(payload, "total_duration"),
            load_duration=_optional_nonnegative_int(payload, "load_duration"),
            prompt_eval_count=_optional_nonnegative_int(payload, "prompt_eval_count"),
            prompt_eval_duration=_optional_nonnegative_int(
                payload, "prompt_eval_duration"
            ),
            eval_count=_optional_nonnegative_int(payload, "eval_count"),
            eval_duration=_optional_nonnegative_int(payload, "eval_duration"),
        )

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        timeout: tuple[int, int],
        json_body: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        kwargs: dict[str, object] = {"timeout": timeout}
        if json_body is not None:
            kwargs["json"] = json_body

        for attempt in range(3):
            try:
                response = self.session.request(method, url, **kwargs)
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as exc:
                if attempt < 2:
                    self.sleep(BACKOFF_SECONDS[attempt])
                    continue
                raise OllamaTransportError(
                    f"Ollama {method} {path} failed after 3 attempts"
                ) from exc
            except requests.RequestException as exc:
                raise OllamaTransportError(f"Ollama {method} {path} failed") from exc

            status = getattr(response, "status_code", None)
            if type(status) is not int:
                raise OllamaProtocolError("HTTP response has no integer status code")
            if status in TRANSIENT_STATUSES and attempt < 2:
                self.sleep(BACKOFF_SECONDS[attempt])
                continue
            if not 200 <= status < 300:
                body = getattr(response, "text", "")
                if type(body) is not str:
                    body = ""
                detail = body[:300].strip()
                suffix = f": {detail}" if detail else ""
                raise OllamaHTTPError(
                    status,
                    f"Ollama {method} {path} returned HTTP {status}{suffix}",
                )
            try:
                payload = response.json()
            except (TypeError, ValueError) as exc:
                raise OllamaProtocolError(
                    f"Ollama {method} {path} returned malformed JSON"
                ) from exc
            if type(payload) is not dict:
                raise OllamaProtocolError(
                    f"Ollama {method} {path} JSON root must be an object"
                )
            return payload

        raise AssertionError("unreachable retry state")


def _required_string(value: dict[str, Any], key: str, label: str) -> str:
    item = value.get(key)
    if type(item) is not str or not item.strip():
        raise OllamaProtocolError(f"{label} is missing")
    return item


def _optional_nonnegative_int(value: dict[str, Any], key: str) -> int | None:
    item = value.get(key)
    if item is None:
        return None
    if type(item) is not int or item < 0:
        raise OllamaProtocolError(f"{key} must be a non-negative integer")
    return item


def _validate_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise TypeError("messages must be a sequence of role/content mappings")
    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"message {index} must be a mapping")
        if set(message) != {"role", "content"}:
            raise ValueError(f"message {index} must contain exactly role and content")
        role = message["role"]
        content = message["content"]
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"message {index} has invalid role")
        if type(content) is not str or not content.strip():
            raise ValueError(f"message {index} content must be non-empty")
        normalized.append({"role": role, "content": content})
    if not normalized:
        raise ValueError("messages must not be empty")
    return normalized
