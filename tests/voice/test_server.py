from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from runtime.settings import RuntimeSettings
from voice.pipeline import PipelineBusyError, VoiceResult
import voice.__main__ as voice_main
import voice.server as voice_server
from voice.server import create_app


def _settings(tmp_path: Path) -> RuntimeSettings:
    return RuntimeSettings.model_validate({
        "install_root": tmp_path / "install", "data_root": tmp_path / "data",
        "cache_root": tmp_path / "cache", "studio_port": 49152,
        "rag_port": 49153, "voice_port": 49154,
        "ollama_origin": "http://127.0.0.1:49155",
        "answer_model": "speaker-v5-a636", "verifier_model": "qwen3:4b",
        "session_secret": "s" * 64,
    })


class FakePipeline:
    status = "idle"

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.calls = 0
        self.texts: list[str] = []
        self.unloads = 0

    def synthesize(self, text: str, request_id: str) -> VoiceResult:
        self.calls += 1
        self.texts.append(text)
        path = self.tmp_path / f"{request_id}.wav"
        sf.write(path, np.zeros(24_000, dtype=np.float32), 24_000, format="WAV")
        data = path.read_bytes()
        return VoiceResult(
            final_path=path, sample_rate=24_000, duration_seconds=1.0,
            sha256=hashlib.sha256(data).hexdigest(), rvc_epoch=200, index_rate=0.75,
        )

    def unload(self) -> None:
        self.unloads += 1


def _auth() -> dict[str, str]:
    return {"Authorization": "Bearer " + "s" * 64}


def _client(
    tmp_path: Path,
    pipeline: FakePipeline | None = None,
    *,
    host: str = "127.0.0.1",
    maintenance_lease_s: float = 30.0,
) -> TestClient:
    return TestClient(
        create_app(
            _settings(tmp_path), pipeline=pipeline or FakePipeline(tmp_path),
            readiness_probe=lambda: None, maintenance_lease_s=maintenance_lease_s,
        ),
        client=(host, 50000),
    )


def test_synthesis_requires_exact_bearer_token_without_leaking_secret(tmp_path: Path):
    client = _client(tmp_path)
    for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic value"}):
        response = client.post("/v1/synthesize", headers=headers, json={"text": "Merhaba", "request_id": "req"})
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
        assert "s" * 64 not in response.text


def test_service_accepts_loopback_authorized_synthesis_and_returns_verified_headers(tmp_path: Path):
    client = _client(tmp_path)
    response = client.post("/v1/synthesize", headers=_auth(), json={"text": "Merhaba", "request_id": "req-1"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.headers["x-youtuber-rvc-epoch"] == "200"
    assert response.headers["x-youtuber-index-rate"] == "0.75"
    assert response.headers["x-youtuber-request-id"] == "req-1"
    assert response.headers["x-youtuber-sha256"] == hashlib.sha256(response.content).hexdigest()


def test_service_forwards_a_4096_token_sized_answer_without_truncation(tmp_path: Path):
    pipeline = FakePipeline(tmp_path)
    text = "a" * 32_768

    response = _client(tmp_path, pipeline).post(
        "/v1/synthesize",
        headers=_auth(),
        json={"text": text, "request_id": "long-answer"},
    )

    assert response.status_code == 200
    assert pipeline.texts == [text]


def test_service_keeps_the_long_answer_contract_bounded(tmp_path: Path):
    pipeline = FakePipeline(tmp_path)

    response = _client(tmp_path, pipeline).post(
        "/v1/synthesize",
        headers=_auth(),
        json={"text": "a" * 32_769, "request_id": "too-long"},
    )

    assert response.status_code == 422
    assert pipeline.calls == 0


def test_loopback_routes_reject_non_loopback_clients_even_live_health(tmp_path: Path):
    response = _client(tmp_path, host="192.0.2.1").get("/health/live")
    assert response.status_code == 403


def test_invalid_request_ids_are_rejected_before_pipeline_execution(tmp_path: Path):
    pipeline = FakePipeline(tmp_path)
    response = _client(tmp_path, pipeline).post(
        "/v1/synthesize", headers=_auth(), json={"text": "Merhaba", "request_id": "../bad"}
    )
    assert response.status_code == 422
    assert pipeline.calls == 0


def test_concurrent_request_gets_busy_without_second_pipeline_call(tmp_path: Path):
    started, release = threading.Event(), threading.Event()

    class BlockingPipeline(FakePipeline):
        def synthesize(self, text: str, request_id: str) -> VoiceResult:
            self.calls += 1
            started.set()
            assert release.wait(2)
            path = self.tmp_path / f"{request_id}.wav"
            sf.write(path, np.zeros(24_000, dtype=np.float32), 24_000, format="WAV")
            data = path.read_bytes()
            return VoiceResult(
                final_path=path, sample_rate=24_000, duration_seconds=1.0,
                sha256=hashlib.sha256(data).hexdigest(), rvc_epoch=200, index_rate=0.75,
            )

    pipeline = BlockingPipeline(tmp_path)
    client = _client(tmp_path, pipeline)
    first: list[object] = []
    thread = threading.Thread(target=lambda: first.append(client.post("/v1/synthesize", headers=_auth(), json={"text": "Bir", "request_id": "one"})))
    thread.start()
    assert started.wait(2)
    busy = client.post("/v1/synthesize", headers=_auth(), json={"text": "İki", "request_id": "two"})
    release.set()
    thread.join(2)
    assert busy.status_code == 409
    assert pipeline.calls == 1


def test_maintenance_lease_blocks_synthesis_until_the_matching_release(tmp_path: Path):
    pipeline = FakePipeline(tmp_path)
    client = _client(tmp_path, pipeline)

    acquired = client.post("/v1/maintenance/acquire", headers=_auth())
    blocked = client.post(
        "/v1/synthesize", headers=_auth(),
        json={"text": "Merhaba", "request_id": "blocked"},
    )
    wrong_release = client.post(
        "/v1/maintenance/release", headers=_auth(),
        json={"lease_token": "x" * 43},
    )
    released = client.post(
        "/v1/maintenance/release", headers=_auth(),
        json={"lease_token": acquired.json()["lease_token"]},
    )
    completed = client.post(
        "/v1/synthesize", headers=_auth(),
        json={"text": "Merhaba", "request_id": "completed"},
    )

    assert acquired.status_code == 200
    assert blocked.status_code == 409
    assert wrong_release.status_code == 409
    assert released.json() == {"status": "released"}
    assert completed.status_code == 200
    assert pipeline.calls == 1


def test_failed_maintenance_release_self_heals_after_lease_expiry(tmp_path: Path):
    pipeline = FakePipeline(tmp_path)
    client = _client(tmp_path, pipeline, maintenance_lease_s=0.05)

    acquired = client.post("/v1/maintenance/acquire", headers=_auth())
    failed_release = client.post(
        "/v1/maintenance/release", headers=_auth(), json={"lease_token": "x" * 43},
    )
    blocked = client.post(
        "/v1/synthesize", headers=_auth(),
        json={"text": "Merhaba", "request_id": "blocked-expiry"},
    )
    time.sleep(0.15)
    recovered = client.post(
        "/v1/synthesize", headers=_auth(),
        json={"text": "Merhaba", "request_id": "recovered-expiry"},
    )

    assert acquired.status_code == 200
    assert failed_release.status_code == 409
    assert blocked.status_code == 409
    assert recovered.status_code == 200


def test_stale_expiry_callback_cannot_release_a_renewed_voice_lease(tmp_path: Path, monkeypatch):
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

    monkeypatch.setattr("voice.server.threading.Timer", ControlledTimer)
    pipeline = FakePipeline(tmp_path)
    client = _client(tmp_path, pipeline)
    acquired = client.post("/v1/maintenance/acquire", headers=_auth())
    token = acquired.json()["lease_token"]
    renewed = client.post("/v1/maintenance/renew", headers=_auth(), json={"lease_token": token})

    timers[0].fire()
    still_blocked = client.post(
        "/v1/synthesize", headers=_auth(),
        json={"text": "Merhaba", "request_id": "blocked-stale"},
    )
    timers[1].fire()
    recovered = client.post(
        "/v1/synthesize", headers=_auth(),
        json={"text": "Merhaba", "request_id": "recovered-stale"},
    )

    assert renewed.json()["status"] == "renewed"
    assert still_blocked.status_code == 409
    assert recovered.status_code == 200


def test_readiness_runs_injected_runtime_import_probe_after_manifest_validation(monkeypatch, tmp_path: Path):
    calls: list[Path] = []
    monkeypatch.setattr(voice_server, "_validate_manifest", lambda paths: None, raising=False)

    voice_server.run_readiness_check(_settings(tmp_path), runtime_probe=lambda paths: calls.append(paths.code_root))

    assert calls == [tmp_path / "install"]


def test_readiness_propagates_injected_runtime_import_failure(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(voice_server, "_validate_manifest", lambda paths: None, raising=False)

    with __import__("pytest").raises(RuntimeError, match="Coqui unavailable"):
        voice_server.run_readiness_check(_settings(tmp_path), runtime_probe=lambda paths: (_ for _ in ()).throw(RuntimeError("Coqui unavailable")))


def test_package_healthcheck_probes_imports_without_requiring_model_artifacts(monkeypatch, tmp_path: Path):
    calls: list[Path] = []
    monkeypatch.setattr(
        voice_server,
        "_validate_manifest",
        lambda paths: (_ for _ in ()).throw(AssertionError("must not read model artifacts")),
    )

    voice_server.run_package_healthcheck(
        _settings(tmp_path), runtime_probe=lambda paths: calls.append(paths.code_root)
    )

    assert calls == [tmp_path / "install"]


def test_healthcheck_returns_nonzero_without_binding_when_readiness_fails(
    monkeypatch, tmp_path: Path, capsys,
):
    monkeypatch.setattr(voice_main.RuntimeSettings, "from_environment", lambda: _settings(tmp_path))
    monkeypatch.setattr(voice_main, "run_package_healthcheck", lambda settings: (_ for _ in ()).throw(RuntimeError("bad runtime")))
    monkeypatch.setattr(voice_main.uvicorn, "run", lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not bind")))
    monkeypatch.setattr(sys, "argv", ["voice", "--healthcheck"])

    assert voice_main.main() == 1
    assert capsys.readouterr().err == "voice package healthcheck failed: RuntimeError: bad runtime\n"


def test_healthcheck_success_returns_zero_without_binding(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(voice_main.RuntimeSettings, "from_environment", lambda: _settings(tmp_path))
    monkeypatch.setattr(voice_main, "run_package_healthcheck", lambda settings: None)
    monkeypatch.setattr(voice_main.uvicorn, "run", lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not bind")))
    monkeypatch.setattr(sys, "argv", ["voice", "--healthcheck"])

    assert voice_main.main() == 0
