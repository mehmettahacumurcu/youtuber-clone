from unittest.mock import MagicMock

from rag.generate import generate_ollama


def _packaged_environment(monkeypatch, tmp_path, *, ollama_port: int = 49155):
    values = {
        "YOUTUBER_INSTALL_ROOT": str(tmp_path / "install"),
        "YOUTUBER_DATA_ROOT": str(tmp_path / "data"),
        "YOUTUBER_CACHE_ROOT": str(tmp_path / "cache"),
        "YOUTUBER_STUDIO_PORT": "49152",
        "YOUTUBER_RAG_PORT": "49153",
        "YOUTUBER_VOICE_PORT": "49154",
        "YOUTUBER_OLLAMA_ORIGIN": f"http://127.0.0.1:{ollama_port}",
        "YOUTUBER_ANSWER_MODEL": "speaker-v5-a636",
        "YOUTUBER_VERIFIER_MODEL": "qwen3:4b",
        "YOUTUBER_SESSION_SECRET": "s" * 64,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_generate_posts_raw_and_returns_response_through_injected_transport():
    fake = MagicMock()
    fake.json.return_value = {"response": "abi bak şöyle..."}
    fake.raise_for_status.return_value = None
    calls = []

    def post_json(method, url, *, payload, timeout_s):
        calls.append((method, url, payload, timeout_s))
        return {"response": fake.json.return_value["response"]}

    out = generate_ollama(
        "PROMPT", model="speaker-llama-ep2", temperature=0.78, top_p=0.85,
        repeat_penalty=1.3, num_predict=256, post_json=post_json,
    )
    assert out == "abi bak şöyle..."
    method, url, body, timeout = calls[0]
    assert method == "POST"
    assert url == "http://127.0.0.1:11434/api/generate"
    assert timeout == 180
    assert body["model"] == "speaker-llama-ep2"
    assert body["raw"] is True and body["stream"] is False
    assert body["prompt"] == "PROMPT"
    assert body["options"]["temperature"] == 0.78
    assert body["options"]["repeat_penalty"] == 1.3


def test_generate_never_needs_a_real_network_when_a_transport_is_injected():
    def no_network(*_args, **_kwargs):
        raise AssertionError("a test must not access the real Ollama service")

    assert generate_ollama(
        "PROMPT", model="speaker", temperature=0.1, top_p=0.2, repeat_penalty=1.0, num_predict=1,
        post_json=lambda *_args, **_kwargs: {"response": "safe"},
    ) == "safe"
    assert no_network


def test_generate_uses_the_exact_launcher_ollama_origin_not_default_11434(
    monkeypatch, tmp_path,
):
    _packaged_environment(monkeypatch, tmp_path)
    calls = []

    def post_json(method, url, *, payload, timeout_s):
        calls.append((method, url))
        return {"response": "safe"}

    assert generate_ollama(
        "PROMPT", model="speaker", temperature=0.1, top_p=0.2,
        repeat_penalty=1.0, num_predict=1, post_json=post_json,
    ) == "safe"
    assert calls == [("POST", "http://127.0.0.1:49155/api/generate")]
