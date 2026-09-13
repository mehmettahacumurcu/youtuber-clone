from __future__ import annotations

import pytest

from runtime.settings import RuntimeSettings


@pytest.fixture
def complete_runtime_env(tmp_path):
    return {
        "YOUTUBER_INSTALL_ROOT": str(tmp_path / "install"),
        "YOUTUBER_DATA_ROOT": str(tmp_path / "data"),
        "YOUTUBER_CACHE_ROOT": str(tmp_path / "cache"),
        "YOUTUBER_STUDIO_PORT": "49152",
        "YOUTUBER_RAG_PORT": "49153",
        "YOUTUBER_VOICE_PORT": "49154",
        "YOUTUBER_OLLAMA_ORIGIN": "http://127.0.0.1:49155",
        "YOUTUBER_ANSWER_MODEL": "speaker-v5-a636",
        "YOUTUBER_VERIFIER_MODEL": "qwen3:4b",
        "YOUTUBER_SESSION_SECRET": "a" * 64,
        "YOUTUBER_LOW_VRAM_VOICE": "1",
    }


def _set_environment(monkeypatch, values):
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_runtime_settings_load_complete_launcher_environment(monkeypatch, complete_runtime_env):
    _set_environment(monkeypatch, complete_runtime_env)

    settings = RuntimeSettings.from_environment()

    assert settings.packaged is True
    assert settings.studio_port == 49152
    assert settings.rag_port == 49153
    assert settings.ollama_origin == "http://127.0.0.1:49155"
    assert settings.answer_model == "speaker-v5-a636"
    assert settings.verifier_model == "qwen3:4b"
    assert settings.low_vram_voice is True
    with pytest.raises(Exception):
        settings.studio_port = 1


def test_runtime_settings_accepts_the_512_byte_secret_boundary(monkeypatch, complete_runtime_env):
    complete_runtime_env["YOUTUBER_SESSION_SECRET"] = "s" * 512
    _set_environment(monkeypatch, complete_runtime_env)

    assert RuntimeSettings.from_environment().session_secret == "s" * 512


def test_runtime_settings_reject_duplicate_ports(monkeypatch, complete_runtime_env):
    complete_runtime_env["YOUTUBER_RAG_PORT"] = complete_runtime_env["YOUTUBER_STUDIO_PORT"]
    _set_environment(monkeypatch, complete_runtime_env)

    with pytest.raises(ValueError, match="unique"):
        RuntimeSettings.from_environment()


@pytest.mark.parametrize("origin", [
    "http://localhost:49155",
    "http://127.0.0.1:11434/path",
    "https://127.0.0.1:49155",
    "http://user@127.0.0.1:49155",
    "http://127.0.0.1:49155?query=1",
    "http://127.0.0.1:49155#fragment",
])
def test_packaged_runtime_rejects_noncanonical_ollama_origins(
    monkeypatch, complete_runtime_env, origin
):
    complete_runtime_env["YOUTUBER_OLLAMA_ORIGIN"] = origin
    _set_environment(monkeypatch, complete_runtime_env)

    with pytest.raises(ValueError, match="Ollama origin"):
        RuntimeSettings.from_environment()


@pytest.mark.parametrize("name,value", [
    ("YOUTUBER_INSTALL_ROOT", "relative/install"),
    ("YOUTUBER_DATA_ROOT", "relative/data"),
    ("YOUTUBER_CACHE_ROOT", "relative/cache"),
])
def test_runtime_settings_rejects_nonabsolute_packaged_roots(
    monkeypatch, complete_runtime_env, name, value
):
    complete_runtime_env[name] = value
    _set_environment(monkeypatch, complete_runtime_env)

    with pytest.raises(ValueError, match="absolute"):
        RuntimeSettings.from_environment()


@pytest.mark.parametrize("name,value,match", [
    ("YOUTUBER_STUDIO_PORT", "1023", "1024"),
    ("YOUTUBER_RAG_PORT", "65536", "65535"),
    ("YOUTUBER_SESSION_SECRET", "short", "32 bytes"),
    ("YOUTUBER_SESSION_SECRET", "ı" * 15, "32 bytes"),
    ("YOUTUBER_SESSION_SECRET", "s" * 513, "512 bytes"),
])
def test_runtime_settings_rejects_invalid_launcher_values(
    monkeypatch, complete_runtime_env, name, value, match
):
    complete_runtime_env[name] = value
    _set_environment(monkeypatch, complete_runtime_env)

    with pytest.raises(ValueError, match=match):
        RuntimeSettings.from_environment()


def test_runtime_settings_keeps_existing_developer_defaults(monkeypatch):
    for name in (
        "YOUTUBER_INSTALL_ROOT", "YOUTUBER_DATA_ROOT", "YOUTUBER_CACHE_ROOT",
        "YOUTUBER_STUDIO_PORT", "YOUTUBER_RAG_PORT", "YOUTUBER_VOICE_PORT",
        "YOUTUBER_SESSION_SECRET", "YOUTUBER_LOW_VRAM_VOICE",
        "YOUTUBER_OLLAMA_ORIGIN", "YOUTUBER_ANSWER_MODEL", "YOUTUBER_VERIFIER_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = RuntimeSettings.from_environment()

    assert settings.packaged is False
    assert (settings.studio_port, settings.rag_port, settings.voice_port) == (7861, 7670, 7862)
    assert settings.session_secret is None
    assert settings.ollama_origin == "http://127.0.0.1:11434"
