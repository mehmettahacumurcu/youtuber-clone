from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from runtime.session_auth import SessionAuth
from runtime.settings import RuntimeSettings
from ui.server import _HttpOllamaHealthClient, _spawn_rag, create_app


SECRET = "s" * 64


def _settings(tmp_path: Path) -> RuntimeSettings:
    return RuntimeSettings.model_validate({
        "install_root": tmp_path / "install", "data_root": tmp_path / "data",
        "cache_root": tmp_path / "cache", "studio_port": 49152,
        "rag_port": 49153, "voice_port": 49154,
        "ollama_origin": "http://127.0.0.1:49155",
        "answer_model": "speaker-v5-a636", "verifier_model": "qwen3:4b",
        "session_secret": SECRET,
    })


@pytest.mark.parametrize("secret", [None, "", "s" * 31, "ı" * 15, "s" * 513])
def test_spawn_rag_rejects_every_invalid_secret_before_starting_child_or_logs(tmp_path: Path, monkeypatch, secret):
    settings = RuntimeSettings.model_construct(
        install_root=tmp_path / "install", data_root=tmp_path / "data", cache_root=tmp_path / "cache",
        studio_port=49152, rag_port=49153, voice_port=49154, session_secret=secret,
    )
    spawned = []
    monkeypatch.setattr("ui.server.managed_popen", lambda *_args, **_kwargs: spawned.append(True))

    with pytest.raises(ValueError, match="32 bytes|512 bytes"):
        _spawn_rag(settings)

    assert spawned == []
    assert not (tmp_path / "data" / "logs").exists()


@pytest.mark.parametrize("secret", ["s" * 32, "s" * 512])
def test_spawn_rag_passes_valid_secret_boundaries_to_the_process_once(tmp_path: Path, monkeypatch, secret):
    settings = RuntimeSettings.model_construct(
        install_root=tmp_path / "install", data_root=tmp_path / "data", cache_root=tmp_path / "cache",
        studio_port=49152, rag_port=49153, voice_port=49154, session_secret=secret,
    )
    calls = []
    monkeypatch.setattr("ui.server.managed_popen", lambda *_args, **kwargs: calls.append(kwargs) or object())

    _spawn_rag(settings)

    assert len(calls) == 1
    assert calls[0]["redactions"] == (secret.encode("utf-8"),)
    assert not (tmp_path / "data" / "logs").exists()


def test_spawn_rag_allows_absent_secret_only_for_developer_settings(monkeypatch):
    settings = RuntimeSettings.model_construct(
        install_root=None, data_root=None, cache_root=None, studio_port=49152,
        rag_port=49153, voice_port=49154, session_secret=None,
    )
    calls = []
    monkeypatch.setattr("ui.server.managed_popen", lambda *_args, **kwargs: calls.append(kwargs) or object())

    _spawn_rag(settings)

    assert len(calls) == 1
    assert calls[0]["redactions"] == ()


class FakeProcess:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.terminated = 0
        self.killed = 0

    def poll(self):
        return None

    def terminate(self) -> None:
        self.events.append("terminate")
        self.terminated += 1

    def wait(self, timeout: float) -> None:
        self.events.append("wait")

    def kill(self) -> None:
        self.events.append("kill")
        self.killed += 1


class FakeRag:
    def __init__(self, events: list[str], *, ready: bool = True, warming_once: bool = False) -> None:
        self.events = events
        self.is_ready = ready
        self.warming_once = warming_once

    def ready(self):
        self.events.append("rag-ready")
        if self.warming_once:
            self.warming_once = False
            return {"status": "unready", "runtime_api": 1, "schema_version": 2}
        if not self.is_ready:
            raise RuntimeError("RAG unavailable")
        return {
            "status": "ready", "runtime_api": 1, "schema_version": 2,
            "corpus_version": "corpus-v1", "index_version": "index-v1",
            "embedder": "BAAI/bge-m3", "reranker": "BAAI/bge-reranker-v2-m3",
        }


class FakeOllama:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def ready(self):
        self.events.append("ollama-ready")
        return {"answer_model": "speaker-v5-a636", "verifier_model": "qwen3:4b"}


def _ui() -> FastAPI:
    app = FastAPI()
    for path in ("/", "/assets/app.js", "/config", "/queue/join", "/api/predict", "/queue/data"):
        app.add_api_route(path, lambda: {"ui": "ok"}, methods=["GET", "POST"])

    @app.websocket("/queue/join")
    async def queue_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_text("mount-received")
        await websocket.close()

    return app


def _client(
    tmp_path: Path,
    *,
    events: list[str] | None = None,
    rag_ready: bool = True,
    auth: SessionAuth | None = None,
    expected_rag_identity=None,
    chat_handler=None,
    maintenance_lease_s: float = 30.0,
) -> tuple[TestClient, list[str], FakeProcess]:
    events = events if events is not None else []
    process = FakeProcess(events)
    app = create_app(
        _settings(tmp_path), auth=auth,
        process_factory=lambda _settings: events.append("rag-start") or process,
        rag_client=FakeRag(events, ready=rag_ready), ollama_client=FakeOllama(events),
        manifest_probe=lambda _ready: events.append("manifest"),
        expected_rag_identity=expected_rag_identity or (lambda _settings: None),
        ui_factory=lambda: events.append("studio-mount") or _ui(),
        mount_ui=lambda host, ui: host.mount("/", ui),
        chat_handler=chat_handler,
        sleep=lambda _seconds: None,
        startup_timeout_s=0.0 if not rag_ready else 30.0,
        maintenance_lease_s=maintenance_lease_s,
    )
    return TestClient(app, client=("127.0.0.1", 50000)), events, process


def _bearer() -> dict[str, str]:
    return {"Authorization": "Bearer " + SECRET}


def test_bootstrap_sets_a_distinct_strict_http_only_cookie_without_putting_secret_in_redirect(tmp_path: Path):
    client, _events, _process = _client(tmp_path)
    with client:
        response = client.post("/bootstrap", headers=_bearer(), follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert "httponly" in response.headers["set-cookie"].lower()
    assert "samesite=strict" in response.headers["set-cookie"].lower()
    assert "path=/" in response.headers["set-cookie"].lower()
    assert SECRET not in response.headers["location"]
    assert SECRET not in response.text


def test_invalid_bootstrap_bearer_is_forbidden_and_never_echoes_the_secret(tmp_path: Path):
    client, _events, _process = _client(tmp_path)
    with client:
        response = client.post("/bootstrap", headers={"Authorization": "Bearer bad"})

    assert response.status_code == 403
    assert SECRET not in response.text


def test_status_requires_session_and_reports_idle_when_no_generation_is_running(
    tmp_path: Path,
) -> None:
    client, _events, _process = _client(tmp_path)
    with client:
        unauthenticated = client.get("/v1/status")
        bearer_authenticated = client.get("/v1/status", headers=_bearer())
        client.post("/bootstrap", headers=_bearer(), follow_redirects=False)
        authenticated = client.get("/v1/status")

    assert unauthenticated.status_code == 403
    assert bearer_authenticated.status_code == 200
    assert bearer_authenticated.json() == {"status": "idle"}
    assert authenticated.status_code == 200
    assert authenticated.json() == {"status": "idle"}


def test_status_reports_busy_for_the_full_generation_critical_section(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocked_chat(*_args):
        entered.set()
        assert release.wait(timeout=5)
        return (
            "answer", "context", "meta",
            {"status": "answerable", "evidence_count": 1}, "generated",
        )

    client, _events, _process = _client(tmp_path, chat_handler=blocked_chat)
    payload = {
        "question": "question",
        "answer_mode": "grounded_fallback",
        "retrieval_mode": "clean",
        "answer_token_cap": 640,
    }
    responses = []
    with client:
        client.post("/bootstrap", headers=_bearer(), follow_redirects=False)
        worker = threading.Thread(target=lambda: responses.append(client.post("/v1/chat", json=payload)))
        worker.start()
        assert entered.wait(timeout=5)
        busy = client.get("/v1/status")
        release.set()
        worker.join(timeout=5)
        idle = client.get("/v1/status")

    assert not worker.is_alive()
    assert busy.json() == {"status": "busy"}
    assert idle.json() == {"status": "idle"}
    assert responses[0].status_code == 200


def test_launcher_maintenance_lease_atomically_blocks_chat_until_release(
    tmp_path: Path,
) -> None:
    client, _events, _process = _client(
        tmp_path,
        chat_handler=lambda *_args: (
            "answer", "context", "meta",
            {"status": "answerable", "evidence_count": 1}, "generated",
        ),
    )
    payload = {
        "question": "question",
        "answer_mode": "grounded_fallback",
        "retrieval_mode": "clean",
        "answer_token_cap": 640,
    }
    with client:
        client.post("/bootstrap", headers=_bearer(), follow_redirects=False)
        cookie_only = client.post("/v1/maintenance/acquire")
        acquired = client.post("/v1/maintenance/acquire", headers=_bearer())
        blocked = client.post("/v1/chat", json=payload)
        released = client.post(
            "/v1/maintenance/release",
            headers=_bearer(),
            json={"lease_token": acquired.json()["lease_token"]},
        )
        completed = client.post("/v1/chat", json=payload)

    assert cookie_only.status_code == 403
    assert acquired.status_code == 200
    assert len(acquired.json()["lease_token"]) >= 32
    assert blocked.status_code == 409
    assert released.json() == {"status": "released"}
    assert completed.status_code == 200


def test_lost_maintenance_acquire_response_self_heals_after_lease_expiry(
    tmp_path: Path,
) -> None:
    client, _events, _process = _client(
        tmp_path,
        maintenance_lease_s=0.05,
        chat_handler=lambda *_args: (
            "answer", "context", "meta",
            {"status": "answerable", "evidence_count": 1}, "generated",
        ),
    )
    payload = {
        "question": "question", "answer_mode": "grounded_fallback",
        "retrieval_mode": "clean", "answer_token_cap": 640,
    }
    with client:
        client.post("/bootstrap", headers=_bearer(), follow_redirects=False)
        acquired = client.post("/v1/maintenance/acquire", headers=_bearer())
        blocked = client.post("/v1/chat", json=payload)
        time.sleep(0.15)
        recovered = client.post("/v1/chat", json=payload)

    assert acquired.status_code == 200
    assert blocked.status_code == 409
    assert recovered.status_code == 200


def test_stale_expiry_callback_cannot_release_a_renewed_studio_lease(
    tmp_path: Path, monkeypatch,
) -> None:
    timers = []

    class ControlledTimer:
        def __init__(self, _interval, function, args=()):
            self.function = function
            self.args = args
            self.daemon = False
            timers.append(self)

        def start(self):
            return None

        def cancel(self):
            return None

        def fire(self):
            self.function(*self.args)

    monkeypatch.setattr("ui.server.threading.Timer", ControlledTimer)
    client, _events, _process = _client(
        tmp_path,
        chat_handler=lambda *_args: (
            "answer", "context", "meta",
            {"status": "answerable", "evidence_count": 1}, "generated",
        ),
    )
    payload = {
        "question": "question", "answer_mode": "grounded_fallback",
        "retrieval_mode": "clean", "answer_token_cap": 640,
    }
    with client:
        client.post("/bootstrap", headers=_bearer(), follow_redirects=False)
        acquired = client.post("/v1/maintenance/acquire", headers=_bearer())
        token = acquired.json()["lease_token"]
        renewed = client.post("/v1/maintenance/renew", headers=_bearer(), json={"lease_token": token})
        timers[0].fire()
        still_blocked = client.post("/v1/chat", json=payload)
        timers[1].fire()
        recovered = client.post("/v1/chat", json=payload)

    assert renewed.json()["status"] == "renewed"
    assert still_blocked.status_code == 409
    assert recovered.status_code == 200


def test_chat_endpoint_requires_cookie_and_rejects_extra_or_out_of_range_input(
    tmp_path: Path,
) -> None:
    calls = []
    client, _events, _process = _client(
        tmp_path,
        chat_handler=lambda *args: calls.append(args)
        or (
            "answer", "context", "meta",
            {"status": "answerable", "evidence_count": 1}, "generated",
        ),
    )
    payload = {
        "question": "Mutlak butlan hakkında ne düşünüyorsun?",
        "answer_mode": "grounded_fallback",
        "retrieval_mode": "clean_with_card_hints",
        "answer_token_cap": 640,
    }
    with client:
        unauthenticated = client.post("/v1/chat", json=payload)
        client.post("/bootstrap", headers=_bearer(), follow_redirects=False)
        extra = client.post("/v1/chat", json={**payload, "test_bypass": True})
        oversized = client.post("/v1/chat", json={**payload, "answer_token_cap": 4097})

    assert unauthenticated.status_code == 403
    assert extra.status_code == 422
    assert oversized.status_code == 422
    assert calls == []


def test_cookie_authenticated_chat_runs_inside_studio_and_returns_contract_metadata(
    tmp_path: Path,
) -> None:
    calls = []

    def chat_handler(question, model, answer_mode, retrieval_mode, answer_token_cap):
        calls.append((question, model, answer_mode, retrieval_mode, answer_token_cap))
        return (
            "model answer",
            "verified context",
            "generation metadata",
            {"status": "answerable", "evidence_count": 2},
            "generated",
        )

    client, _events, _process = _client(tmp_path, chat_handler=chat_handler)
    payload = {
        "question": "Mutlak butlan hakkında ne düşünüyorsun?",
        "answer_mode": "grounded_fallback",
        "retrieval_mode": "clean_with_card_hints",
        "answer_token_cap": 640,
    }
    with client:
        client.post("/bootstrap", headers=_bearer(), follow_redirects=False)
        response = client.post("/v1/chat", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "runtime_api": 1,
        "answer_mode_contract_version": 1,
        "answer_model": "speaker-v5-a636",
        "answer_mode": "grounded_fallback",
        "retrieval_mode": "clean_with_card_hints",
        "answer_token_cap": 640,
        "answer": "model answer",
        "context_markdown": "verified context",
        "metadata": "generation metadata",
        "grounding_status": "answerable",
        "evidence_count": 2,
        "generation_status": "generated",
    }
    assert calls == [
        (
            payload["question"],
            "speaker-v5-a636",
            "grounded_fallback",
            "clean_with_card_hints",
            640,
        )
    ]


def test_chat_response_reports_failed_retrieval_used_by_fallback_generation(
    tmp_path: Path,
) -> None:
    client, _events, _process = _client(
        tmp_path,
        chat_handler=lambda *_args: (
            "unguarded fallback answer",
            "RAG failed",
            "warning",
            {"status": "error", "evidence_count": 0},
            "generated",
        ),
    )
    with client:
        client.post("/bootstrap", headers=_bearer(), follow_redirects=False)
        response = client.post(
            "/v1/chat",
            json={
                "question": "question",
                "answer_mode": "grounded_fallback",
                "retrieval_mode": "clean",
                "answer_token_cap": 640,
            },
        )

    assert response.status_code == 200
    assert response.json()["grounding_status"] == "error"
    assert response.json()["evidence_count"] == 0
    assert response.json()["generation_status"] == "generated"


def test_chat_endpoint_rejects_empty_generation_despite_answerable_trace(
    tmp_path: Path, monkeypatch,
) -> None:
    from types import SimpleNamespace

    from ui.chat_runtime import _EMPTY_OUTPUT_TR

    monkeypatch.setattr(
        "ui.chat_runtime.chat_turn_with_trace",
        lambda *_args, **_kwargs: SimpleNamespace(
            history=[{"role": "assistant", "content": _EMPTY_OUTPUT_TR}],
            context="verified context",
            metadata="",
            trace={"status": "answerable", "evidence_count": 1},
            generation_status="empty_output",
        ),
    )
    client, _events, _process = _client(tmp_path)
    with client:
        client.post("/bootstrap", headers=_bearer(), follow_redirects=False)
        response = client.post(
            "/v1/chat",
            json={
                "question": "question",
                "answer_mode": "grounded_strict",
                "retrieval_mode": "clean",
                "answer_token_cap": 640,
            },
        )

    assert response.status_code == 502
    assert "answer" not in response.json()


@pytest.mark.parametrize(
    "stock_answer",
    [
        "Boş model çıktısı alındı; yanıt üretilemedi.",
        "RAG kanıt denetimi şu anda çalışmadığı için cevap üretmedim.",
        "Bu konuya pek değinmemişim, elimde bununla ilgili bir şey yok.",
    ],
)
def test_production_chat_rejects_stock_non_answers_even_if_marked_generated(
    monkeypatch, stock_answer: str,
) -> None:
    from types import SimpleNamespace

    from ui.server import _production_chat

    monkeypatch.setattr(
        "ui.chat_runtime.chat_turn_with_trace",
        lambda *_args, **_kwargs: SimpleNamespace(
            history=[{"role": "assistant", "content": stock_answer}],
            context="context",
            metadata="",
            trace={"status": "answerable", "evidence_count": 1},
            generation_status="generated",
        ),
    )

    with pytest.raises(RuntimeError, match="usable answer"):
        _production_chat("q", "model", "grounded_strict", "clean", 640)


def test_default_studio_rag_startup_bound_matches_packaged_controller_contract():
    import inspect

    assert inspect.signature(create_app).parameters["startup_timeout_s"].default == 120.0


def test_missing_cookie_blocks_studio_gradio_assets_api_and_queue_paths(tmp_path: Path):
    client, _events, _process = _client(tmp_path)
    with client:
        for path in ("/", "/assets/app.js", "/config", "/queue/join", "/api/predict", "/queue/data"):
            response = client.get(path)
            assert response.status_code == 403, path


def test_missing_cookie_closes_gradio_queue_websocket_before_mount_receives_it(tmp_path: Path):
    client, _events, _process = _client(tmp_path)
    with client:
        with pytest.raises(WebSocketDisconnect) as error:
            with client.websocket_connect("/queue/join") as socket:
                socket.receive_text()

    assert error.value.code == 1008


def test_browser_nonce_bootstrap_never_returns_bearer_and_is_single_use(tmp_path: Path):
    client, _events, _process = _client(tmp_path)
    with client:
        issued = client.post("/v1/browser-bootstrap", headers=_bearer())
        url = issued.json()["url"]
        exchanged = client.get(url, follow_redirects=False)
        replay = client.get(url, follow_redirects=False)

    assert issued.status_code == 200
    assert url.startswith("http://127.0.0.1:49152/v1/browser-exchange?nonce=")
    assert SECRET not in issued.text
    assert SECRET not in exchanged.headers["location"]
    assert exchanged.status_code == 303
    assert replay.status_code == 403


def test_startup_orders_rag_then_authenticated_ready_then_studio_mount_and_shutdown_stops_child(tmp_path: Path):
    events: list[str] = []
    client, events, process = _client(tmp_path, events=events)
    with client:
        assert events[:5] == ["rag-start", "rag-ready", "manifest", "ollama-ready", "studio-mount"]

    assert process.terminated == 1
    assert process.killed == 0


def test_startup_retries_a_structured_warming_response_before_validating_ready_identity(tmp_path: Path):
    """Break caught: the first HTTP 503/unready payload aborts normal RAG warmup."""
    events: list[str] = []
    process = FakeProcess(events)
    rag = FakeRag(events, warming_once=True)
    app = create_app(
        _settings(tmp_path),
        process_factory=lambda _settings: events.append("rag-start") or process,
        rag_client=rag,
        ollama_client=FakeOllama(events),
        expected_rag_identity=lambda _settings: None,
        ui_factory=lambda: events.append("studio-mount") or _ui(),
        mount_ui=lambda host, ui: host.mount("/", ui),
        sleep=lambda _seconds: None,
    )

    with TestClient(app, client=("127.0.0.1", 50000)):
        pass

    assert events[:5] == ["rag-start", "rag-ready", "rag-ready", "ollama-ready", "studio-mount"]


def test_failed_rag_readiness_terminates_child_before_studio_mount(tmp_path: Path):
    events: list[str] = []
    client, events, process = _client(tmp_path, events=events, rag_ready=False)

    with pytest.raises(RuntimeError, match="RAG unavailable"):
        with client:
            pass

    assert events[:3] == ["rag-start", "rag-ready", "terminate"]
    assert "studio-mount" not in events
    assert process.terminated == 1


def test_manifest_identity_mismatch_terminates_rag_before_studio_mount(tmp_path: Path):
    events: list[str] = []
    client, events, process = _client(
        tmp_path, events=events,
        expected_rag_identity=lambda _settings: {"corpus_version": "other-corpus"},
    )

    with pytest.raises(RuntimeError, match="identities differ"):
        with client:
            pass

    assert "studio-mount" not in events
    assert process.terminated == 1


def test_expected_rag_identity_is_loaded_once_across_repeated_health_polls(tmp_path: Path):
    calls: list[Path] = []

    def expected(settings: RuntimeSettings):
        calls.append(settings.data_root)
        return {"corpus_version": "corpus-v1", "index_version": "index-v1"}

    client, _events, _process = _client(
        tmp_path,
        expected_rag_identity=expected,
    )
    with client:
        for _ in range(5):
            response = client.get("/health/ready")
            assert response.status_code == 200

    assert calls == [tmp_path / "data"]


def test_local_expected_identity_uses_manifest_counts_without_opening_qdrant(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace
    from ui.server import _local_expected_rag_identity

    rag = SimpleNamespace(
        store_path=tmp_path / "qdrant",
        evidence_collection="speaker_clean",
        card_collection="speaker_cards",
        card_catalog_path=tmp_path / "cards.json",
    )
    configured = SimpleNamespace(rag=rag)
    monkeypatch.setattr("pipeline.config.load_settings", lambda _path: configured)
    monkeypatch.setattr(
        "rag.readiness.load_clean_manifest",
        lambda _path: {"clean_count": 3551},
    )
    monkeypatch.setattr(
        "rag.card_manifest.load_card_catalog",
        lambda _path: SimpleNamespace(cards=tuple(range(2371))),
    )
    seen = []

    def approved(_settings, *, store_probe):
        seen.append(store_probe(rag.store_path, rag.evidence_collection, rag.card_collection))
        return {"corpus_version": "manifest-only"}

    monkeypatch.setattr("rag.readiness.approved_identity", approved)

    identity = _local_expected_rag_identity(_settings(tmp_path))

    assert identity == {"corpus_version": "manifest-only"}
    assert seen == [{"speaker_clean": 3551, "speaker_cards": 2371}]


def test_ollama_readiness_uses_launcher_origin_and_model_roles_not_a_11434_decoy(tmp_path: Path, monkeypatch):
    """Break caught: another installed speaker-* tag silently becomes the packaged answer model."""
    paths = tmp_path / "install" / "runtime"
    paths.mkdir(parents=True)
    (paths / "config.yaml").write_text(
        "paths:\n"
        "  data_dir: data\n  audio_dir: data/audio\n  meta_dir: data/meta\n"
        "  vad_dir: data/vad\n  diarize_dir: data/diarize\n  identify_dir: data/identify\n"
        "  transcribe_dir: data/transcribe\n  filter_dir: data/filter\n  dataset_dir: data/dataset\n"
        "  reference_clip: data/reference.wav\n  review_dir: data/review\n"
        "  decisions_db: data/decisions.sqlite\n  logs_dir: logs\n"
        "rag:\n  ollama_model: speaker-v5-a636\n  verifier_model: qwen3:4b\n",
        encoding="utf-8",
    )
    calls = []

    def tags(url, **_kwargs):
        calls.append(url)
        return {
            "models": [
                {"name": "speaker-llama-ep2:latest"},
                {"name": "qwen3:4b"},
                {"name": "speaker-v5-a636:latest"},
            ]
        }

    monkeypatch.setattr("ui.server.loopback_json", tags)

    ready = _HttpOllamaHealthClient(_settings(tmp_path)).ready()

    assert ready == {"answer_model": "speaker-v5-a636:latest", "verifier_model": "qwen3:4b"}
    assert calls == ["http://127.0.0.1:49155/api/tags"]
    assert all(":11434" not in url for url in calls)
