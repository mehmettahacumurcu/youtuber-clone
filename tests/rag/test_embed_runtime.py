from __future__ import annotations

import sys
import types

from rag import embed


def _clear_runtime_environment(monkeypatch):
    for name in (
        "YOUTUBER_INSTALL_ROOT", "YOUTUBER_DATA_ROOT", "YOUTUBER_CACHE_ROOT",
        "YOUTUBER_STUDIO_PORT", "YOUTUBER_RAG_PORT", "YOUTUBER_VOICE_PORT",
        "YOUTUBER_SESSION_SECRET", "YOUTUBER_LOW_VRAM_VOICE",
        "YOUTUBER_OLLAMA_ORIGIN", "YOUTUBER_ANSWER_MODEL", "YOUTUBER_VERIFIER_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


def _set_packaged_environment(monkeypatch, tmp_path):
    values = {
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
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_embedder_cache_uses_effective_device_and_forces_packaged_cpu(monkeypatch, tmp_path):
    calls = []

    class FakeBge:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))

    monkeypatch.setitem(sys.modules, "datasets", types.ModuleType("datasets"))
    monkeypatch.setitem(sys.modules, "FlagEmbedding", types.SimpleNamespace(BGEM3FlagModel=FakeBge))
    embed.get_embedder.cache_clear()
    _clear_runtime_environment(monkeypatch)

    developer = embed.get_embedder("fake-model")
    assert embed.get_embedder("fake-model") is developer

    _set_packaged_environment(monkeypatch, tmp_path)
    packaged = embed.get_embedder("fake-model")

    assert packaged is embed.get_embedder("fake-model")
    assert packaged is not developer
    assert calls == [
        ("fake-model", {"use_fp16": True}),
        ("fake-model", {"use_fp16": False, "devices": "cpu"}),
    ]


def test_reranker_cache_uses_effective_device_and_forces_packaged_cpu(monkeypatch, tmp_path):
    calls = []

    class FakeReranker:
        def __init__(self, model_name, **kwargs):
            calls.append((model_name, kwargs))

    monkeypatch.setitem(sys.modules, "datasets", types.ModuleType("datasets"))
    monkeypatch.setitem(sys.modules, "FlagEmbedding", types.SimpleNamespace(FlagReranker=FakeReranker))
    embed.get_reranker.cache_clear()
    _clear_runtime_environment(monkeypatch)

    developer = embed.get_reranker("fake-model")
    assert embed.get_reranker("fake-model") is developer

    _set_packaged_environment(monkeypatch, tmp_path)
    packaged = embed.get_reranker("fake-model")

    assert packaged is embed.get_reranker("fake-model")
    assert packaged is not developer
    assert calls == [
        ("fake-model", {"use_fp16": True}),
        ("fake-model", {"use_fp16": False, "devices": "cpu"}),
    ]
