"""Launcher environment contract shared by the packaged workers."""
from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import urlsplit

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_ROOT_FIELDS = ("install_root", "data_root", "cache_root")
_PORT_FIELDS = ("studio_port", "rag_port", "voice_port")
DEVELOPMENT_OLLAMA_ORIGIN = "http://127.0.0.1:11434"
_MODEL_NAME = re.compile(r"^[A-Za-z0-9_.-]+(?::[A-Za-z0-9_.-]+)?$")
MIN_SESSION_SECRET_BYTES = 32
MAX_SESSION_SECRET_BYTES = 512


def session_secret_bytes(secret: str | None, *, allow_none: bool = False) -> bytes | None:
    """Validate the launcher secret once, using its UTF-8 wire representation."""
    if secret is None:
        if allow_none:
            return None
        raise ValueError("YOUTUBER_SESSION_SECRET must contain at least 32 bytes and at most 512 bytes")
    encoded = secret.encode("utf-8")
    if not MIN_SESSION_SECRET_BYTES <= len(encoded) <= MAX_SESSION_SECRET_BYTES:
        raise ValueError("YOUTUBER_SESSION_SECRET must contain at least 32 bytes and at most 512 bytes")
    return encoded


def validate_ollama_origin(origin: str) -> int:
    """Return the explicit port for one canonical launcher-owned loopback origin."""
    try:
        parsed = urlsplit(origin)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Ollama origin must be a canonical HTTP 127.0.0.1 origin") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or port is None
        or not 1024 <= port <= 65535
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.netloc != f"127.0.0.1:{port}"
    ):
        raise ValueError("Ollama origin must be a canonical HTTP 127.0.0.1 origin")
    return port


class RuntimeSettings(BaseSettings):
    """Immutable settings selected by the launcher, or developer defaults."""

    model_config = SettingsConfigDict(env_prefix="YOUTUBER_", extra="forbid", frozen=True)

    install_root: Path | None = None
    data_root: Path | None = None
    cache_root: Path | None = None
    studio_port: int = 7861
    rag_port: int = 7670
    voice_port: int = 7862
    ollama_origin: str = DEVELOPMENT_OLLAMA_ORIGIN
    answer_model: str = "speaker-v5-a636"
    verifier_model: str = "qwen3:4b"
    session_secret: str | None = None
    low_vram_voice: bool = False

    @property
    def packaged(self) -> bool:
        return self.install_root is not None

    @classmethod
    def from_environment(cls) -> "RuntimeSettings":
        return cls()

    @model_validator(mode="after")
    def _validate_launcher_contract(self) -> "RuntimeSettings":
        session_secret_bytes(self.session_secret, allow_none=True)
        roots = [getattr(self, name) for name in _ROOT_FIELDS]
        if any(root is not None for root in roots):
            missing = [name for name in _ROOT_FIELDS if getattr(self, name) is None]
            required = (
                *_PORT_FIELDS,
                "ollama_origin",
                "answer_model",
                "verifier_model",
                "session_secret",
            )
            missing.extend(name for name in required if name not in self.__pydantic_fields_set__)
            if missing:
                raise ValueError("packaged runtime requires launcher values: " + ", ".join(missing))
            for name in _ROOT_FIELDS:
                value = getattr(self, name)
                if value is None or not value.is_absolute():
                    raise ValueError(f"{name} must be an absolute path in packaged mode")
            session_secret_bytes(self.session_secret)

        for name in _PORT_FIELDS:
            port = getattr(self, name)
            if not 1024 <= port <= 65535:
                raise ValueError(f"{name} must be between 1024 and 65535")
        ports = [getattr(self, name) for name in _PORT_FIELDS]
        if len(set(ports)) != len(ports):
            raise ValueError("studio, RAG, and voice ports must be unique")
        ollama_port = validate_ollama_origin(self.ollama_origin)
        if self.packaged and ollama_port in ports:
            raise ValueError("Ollama, studio, RAG, and voice ports must be unique")
        for name in ("answer_model", "verifier_model"):
            value = getattr(self, name)
            if not _MODEL_NAME.fullmatch(value):
                raise ValueError(f"{name} must be a canonical Ollama model identity")
        return self
