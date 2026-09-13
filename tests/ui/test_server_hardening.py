from __future__ import annotations

import sys
from pathlib import Path

import ui.__main__ as studio_main
from runtime.settings import RuntimeSettings
from ui.__main__ import build_rag_command


def _settings(tmp_path: Path) -> RuntimeSettings:
    return RuntimeSettings.model_validate({
        "install_root": tmp_path / "install", "data_root": tmp_path / "data",
        "cache_root": tmp_path / "cache", "studio_port": 49152,
        "rag_port": 49153, "voice_port": 49154,
        "ollama_origin": "http://127.0.0.1:49155",
        "answer_model": "speaker-v5-a636", "verifier_model": "qwen3:4b",
        "session_secret": "s" * 64,
    })


def test_frozen_rag_command_dispatches_through_worker_executable_without_python_m(tmp_path: Path):
    command = build_rag_command(_settings(tmp_path), executable="studio-worker.exe", frozen=True)

    assert command == [
        "studio-worker.exe", "--rag-worker", "--port", "49153",
        "--config", str(tmp_path / "install" / "runtime" / "config.yaml"),
        "--host", "127.0.0.1",
    ]


def test_developer_rag_command_uses_module_dispatch(tmp_path: Path):
    command = build_rag_command(_settings(tmp_path), executable=sys.executable, frozen=False)

    assert command[:3] == [sys.executable, "-m", "rag.retrieve_server"]


def test_studio_package_healthcheck_neither_loads_launcher_settings_nor_constructs_a_server(
    monkeypatch,
):
    calls: list[str] = []
    monkeypatch.setattr(
        studio_main.RuntimeSettings,
        "from_environment",
        lambda: (_ for _ in ()).throw(AssertionError("must not load authenticated settings")),
    )
    monkeypatch.setattr(
        studio_main,
        "run_package_healthcheck",
        lambda: calls.append("imports"),
        raising=False,
    )
    monkeypatch.setattr(sys, "argv", ["studio-worker", "--healthcheck"])

    assert studio_main.main() == 0
    assert calls == ["imports"]
