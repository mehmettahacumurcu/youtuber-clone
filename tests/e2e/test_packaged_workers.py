from __future__ import annotations

import json
import io
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

from scripts.smoke_packaged_workers import (
    MAX_TAIL_LINE_CHARS,
    ProcessLog,
    _start,
    _stage_file,
    _terminate_tree,
    _verify_hardlinked_source_integrity,
    _prepare_output_root,
    _wait_pids_gone,
    build_voice_manifest,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "smoke_packaged_workers.py"
STUDIO_EXE = ROOT / "dist" / "workers" / "studio-worker" / "studio-worker.exe"
VOICE_EXE = ROOT / "dist" / "workers" / "voice-worker" / "voice-worker.exe"
STUDIO_ZIP = ROOT / "dist" / "workers" / "studio-worker.zip"


@pytest.mark.skipif(os.name != "nt", reason="packaged worker gate requires Windows")
def test_studio_archive_healthcheck_needs_no_launcher_secret_or_runtime_roots(
    tmp_path: Path,
) -> None:
    """Activate the shipped archive and exercise its import-only healthcheck."""
    if not STUDIO_ZIP.is_file():
        pytest.skip("packaged Studio archive unavailable")
    extraction = (tmp_path / "activation").resolve()
    extraction.mkdir()
    with zipfile.ZipFile(STUDIO_ZIP) as archive:
        for entry in archive.infolist():
            target = (extraction / entry.filename).resolve()
            assert target.is_relative_to(extraction), entry.filename
        archive.extractall(extraction)
    bundle = extraction / "studio-worker"
    executable = bundle / "studio-worker.exe"
    assert executable.is_file()
    assert (bundle / "runtime" / "config.yaml").is_file()
    environment = {
        name: value for name, value in os.environ.items()
        if not name.startswith("YOUTUBER_")
    }
    environment["PYTHONNOUSERSITE"] = "1"

    completed = subprocess.run(
        [str(executable), "--healthcheck"],
        cwd=bundle,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr


@pytest.mark.skipif(os.name != "nt", reason="packaged worker gate requires Windows")
def test_fixture_gate_executes_both_frozen_workers_and_releases_every_child(tmp_path: Path) -> None:
    """Break caught: a smoke that silently substitutes imported Python workers for the EXEs."""
    missing = [str(path) for path in (STUDIO_EXE, VOICE_EXE) if not path.is_file()]
    if missing:
        pytest.skip("packaged worker bundle unavailable: " + ", ".join(missing))

    output_root = tmp_path / "packaged smoke"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--fixture",
            "--output-root",
            str(output_root),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    report_path = output_root / "result.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "youtuber.packaged-smoke.v1"
    assert report["mode"] == "fixture"
    assert Path(report["studio"]["executable"]).samefile(STUDIO_EXE)
    assert Path(report["voice"]["executable"]).samefile(VOICE_EXE)
    assert report["studio"]["frozen_rag_dispatch"] is True
    assert report["studio"]["live_status"] == 200
    assert report["studio"]["unauthenticated_ready_status"] == 403
    assert report["voice"]["live_status"] == 200
    assert report["voice"]["unauthenticated_status_status"] == 401
    assert report["cleanup"]["launched_pids_alive"] == []
    assert report["cleanup"]["ports_released"] is True
    assert report["cleanup"]["job_objects_closed"] is True
    assert "session_secret" not in completed.stdout.casefold()
    assert "authorization" not in completed.stdout.casefold()


def test_output_cleanup_refuses_an_unowned_existing_directory(tmp_path: Path) -> None:
    """Break caught: a typo in --output-root recursively deletes unrelated user files."""
    output = tmp_path / "user-folder"
    output.mkdir()
    victim = output / "keep-me.txt"
    victim.write_text("owned by user", encoding="utf-8")

    with pytest.raises(ValueError, match="not owned"):
        _prepare_output_root(output)

    assert victim.read_text(encoding="utf-8") == "owned by user"


def test_cleanup_waits_for_terminated_child_pids_to_disappear(monkeypatch) -> None:
    states = iter((True, True, False))
    monkeypatch.setattr(
        "scripts.smoke_packaged_workers._pid_alive",
        lambda _pid: next(states),
    )
    monkeypatch.setattr("scripts.smoke_packaged_workers.time.sleep", lambda _seconds: None)

    assert _wait_pids_gone({1234}, timeout=1.0) == []


def test_voice_manifest_builder_locks_epoch_200_and_index_rate_point_75() -> None:
    """Break caught: the staged manifest selects an audition RVC model or rate."""
    selected = {
        "xtts/best_model.pth": (10, "a" * 64),
        "xtts/config.json": (11, "b" * 64),
        "xtts/vocab.json": (12, "c" * 64),
        "references/ref.wav": (13, "d" * 64),
        "rvc/speaker-e200.pth": (14, "e" * 64),
        "rvc/speaker-e200.index": (15, "f" * 64),
        "contentvec/config.json": (16, "1" * 64),
        "contentvec/pytorch_model.bin": (17, "2" * 64),
        "rmvpe/rmvpe.pt": (18, "3" * 64),
    }

    manifest = build_voice_manifest(selected, reference_paths=("references/ref.wav",))

    assert manifest["schema"] == "youtuber.voice.v1"
    assert manifest["rvc"] == {
        "weight": "rvc/speaker-e200.pth",
        "index": "rvc/speaker-e200.index",
        "epoch": 200,
        "index_rate": 0.75,
        "pitch": 0,
        "f0_method": "rmvpe",
        "protect": 0.5,
    }


def test_direct_script_bootstraps_repo_imports_under_isolated_python() -> None:
    """Break caught: direct CLI execution cannot import the repo's rag/runtime packages."""
    code = (
        "import runpy; "
        f"runpy.run_path({str(SCRIPT)!r}, run_name='smoke_import_probe'); "
        "import rag.card_manifest; print('ROOT_IMPORT_OK')"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=ROOT.parent,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ROOT_IMPORT_OK"


class _ChunkSource:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = iter(chunks)
        self.closed = False

    def read(self, _size: int) -> bytes:
        return next(self.chunks, b"")

    def close(self) -> None:
        self.closed = True


def test_process_log_redacts_secret_across_every_chunk_boundary_and_partial_eof(
    tmp_path: Path,
) -> None:
    secret = "boundary-secret-" + "x" * 48
    encoded = secret.encode("utf-8")
    source = _ChunkSource(
        [b"before ", *[bytes([byte]) for byte in encoded], b" after\npartial ", encoded[:24]]
    )
    log = ProcessLog(tmp_path / "worker.log", secret)

    log.pump(source)

    persisted = (tmp_path / "worker.log").read_bytes()
    assert b"[REDACTED]" in persisted
    assert encoded not in persisted
    assert encoded[:24] not in persisted
    assert source.closed is True


def test_process_log_tail_is_bounded_and_defensively_redacted(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.smoke_packaged_workers as smoke

    secret = "tail-secret-" + "z" * 48
    monkeypatch.setattr(smoke, "MAX_LOG_BYTES", 2048)
    source = _ChunkSource([b"a" * 4096 for _ in range(16)])
    log = ProcessLog(tmp_path / "worker.log", secret)
    log.pump(source)
    # Defense in depth: even an accidentally unredacted in-memory entry cannot escape tail().
    log._lines.append("prefix " + secret + " suffix")

    tail = log.tail()

    assert (tmp_path / "worker.log").stat().st_size <= 2048
    assert len(tail) <= 200
    assert all(len(line) <= MAX_TAIL_LINE_CHARS for line in tail)
    assert secret not in "\n".join(tail)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object regression")
def test_dead_parent_job_cleanup_reaps_orphan_descendant(tmp_path: Path) -> None:
    pid_file = tmp_path / "descendant.pid"
    parent_code = (
        "import pathlib, subprocess, sys; "
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid), encoding='ascii')"
    )
    logs = tmp_path / "logs"
    logs.mkdir()
    child = _start(
        "orphan-owner",
        [sys.executable, "-c", parent_code, str(pid_file)],
        cwd=tmp_path,
        env=os.environ.copy(),
        logs=logs,
        secret="s" * 64,
    )
    descendant_pid = 0
    try:
        child.process.wait(timeout=10)
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        descendant_pid = int(pid_file.read_text(encoding="ascii"))
        assert _wait_pids_gone({descendant_pid}, timeout=0.2) == [descendant_pid]

        _terminate_tree(child)

        assert _wait_pids_gone({descendant_pid}, timeout=10) == []
    finally:
        if descendant_pid and _wait_pids_gone({descendant_pid}, timeout=0.1):
            subprocess.run(
                ["taskkill.exe", "/PID", str(descendant_pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )


def test_ollama_residency_restoration_is_computed_from_snapshots() -> None:
    from scripts.smoke_packaged_workers import _restore_ollama_residency

    state = {"answer:latest", "unrelated:latest"}
    actions: list[tuple[str, str]] = []

    def present(model: str) -> None:
        actions.append(("present", model))
        state.add(model)

    def absent(model: str) -> None:
        actions.append(("absent", model))
        state.discard(model)

    state.add("verifier:latest")
    result = _restore_ollama_residency(
        initial=["answer:latest", "unrelated:latest"],
        required={"answer:latest", "verifier:latest"},
        loaded=lambda: sorted(state),
        ensure_present=present,
        ensure_absent=absent,
    )

    assert actions == [("present", "answer:latest"), ("absent", "verifier:latest")]
    assert result["after"] == ["answer:latest", "unrelated:latest"]
    assert result["required_restored"] is True
    assert result["unrelated_unchanged"] is True
    assert result["restored"] is True


def test_controller_source_routes_generation_through_frozen_studio_http() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "from ui.chat_runtime import chat_turn" not in source
    assert 'f"http://127.0.0.1:{ports[0]}/v1/chat"' in source
    assert '"answer_mode": "grounded_strict"' in source
    assert 'chat_payload.get("grounding_status")' in source
    assert 'chat_payload.get("evidence_count", 0) <= 0' in source
    assert 'chat_payload.get("generation_status")' in source


@pytest.mark.parametrize(
    ("answer", "generation_status"),
    [
        ("Boş model çıktısı alındı; yanıt üretilemedi.", "empty_output"),
        ("Boş model çıktısı alındı; yanıt üretilemedi.", "generated"),
        ("RAG kanıt denetimi şu anda çalışmadığı için cevap üretmedim.", "generated"),
        ("Bu konuya pek değinmemişim, elimde bununla ilgili bir şey yok.", "generated"),
    ],
)
def test_controller_rejects_non_answer_before_an_artifact_can_be_written(
    tmp_path: Path, answer: str, generation_status: str,
) -> None:
    import scripts.smoke_packaged_workers as smoke

    artifact = tmp_path / "answer.json"
    payload = {
        "answer": answer,
        "generation_status": generation_status,
        "grounding_status": "answerable",
        "evidence_count": 1,
    }

    validator = getattr(smoke, "_validated_chat_answer", None)
    assert validator is not None, "controller generation validator is missing"
    with pytest.raises(RuntimeError, match="model-generated answer"):
        validator(payload)

    assert not artifact.exists()


def test_hardlinks_are_declared_read_only_and_post_run_hash_detects_any_write(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "stage" / "asset.bin"
    source.write_bytes(b"immutable source")
    selected: list[dict[str, object]] = []
    _stage_file(
        source,
        destination,
        mutable=False,
        selected=selected,
        logical_path="models/asset.bin",
        approved_root=tmp_path,
    )
    if selected[0]["method"] != "hardlink":
        pytest.skip("same-volume hardlinks unavailable")
    selection = {"files": selected}

    assert selected[0]["consumer_access"] == "read_only"
    assert _verify_hardlinked_source_integrity(selection) == 1

    destination.write_bytes(b"consumer wrote")
    with pytest.raises(RuntimeError, match="hardlinked source changed"):
        _verify_hardlinked_source_integrity(selection)


def test_cleanup_writes_report_and_removes_private_staging_when_ollama_disappears(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts.smoke_packaged_workers import _finalize_real_run

    output = tmp_path / "output"
    staged = {
        "output_root": output,
        "install_root": output / "install",
        "data_root": output / "data",
        "cache_root": output / "cache",
        "logs": output / "logs",
        "selection": {"files": []},
    }
    for key in ("install_root", "data_root", "cache_root", "logs"):
        staged[key].mkdir(parents=True)
    report = {"ollama": {}, "assets": {}, "cleanup": {}}
    monkeypatch.setattr(
        "scripts.smoke_packaged_workers._restore_ollama_residency",
        lambda **_kwargs: (_ for _ in ()).throw(ConnectionError("Ollama vanished")),
    )
    monkeypatch.setattr(
        "scripts.smoke_packaged_workers._ollama_loaded_models",
        lambda: (_ for _ in ()).throw(ConnectionError("still gone")),
    )

    final_error = _finalize_real_run(
        report=report,
        children=[],
        launched_tree_pids=set(),
        staged=staged,
        secret="s" * 64,
        initial_ollama=[],
        required_ollama={"answer:latest", "verifier:latest"},
        ports=[49161, 49162, 49163],
        keep_output=False,
        primary_error=RuntimeError("primary generation failure"),
    )

    persisted = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert isinstance(final_error, RuntimeError)
    assert str(final_error) == "primary generation failure"
    assert persisted["failure"]["type"] == "RuntimeError"
    assert {row["step"] for row in persisted["cleanup"]["errors"]} >= {
        "ollama_restore", "ollama_snapshot",
    }
    assert persisted["staging_removed"] is True
    assert not staged["install_root"].exists()
    assert not staged["data_root"].exists()
    assert not staged["cache_root"].exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows reparse regression")
def test_staging_rejects_file_symlink_before_resolve_and_preserves_external_victim(
    tmp_path: Path,
) -> None:
    approved = tmp_path / "approved"
    external = tmp_path / "external"
    approved.mkdir()
    external.mkdir()
    victim = external / "victim.bin"
    victim.write_bytes(b"must survive")
    link = approved / "linked.bin"
    try:
        os.symlink(victim, link)
    except OSError as exc:
        pytest.skip(f"file symlink unavailable: {exc}")

    with pytest.raises(ValueError, match="reparse|symlink"):
        _stage_file(
            link,
            tmp_path / "stage" / "linked.bin",
            mutable=False,
            selected=[],
            logical_path="linked.bin",
            approved_root=approved,
        )

    assert victim.read_bytes() == b"must survive"
    assert not (tmp_path / "stage" / "linked.bin").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_staging_walk_rejects_directory_junction_without_traversing_external_tree(
    tmp_path: Path,
) -> None:
    from scripts.smoke_packaged_workers import _stage_tree

    approved = tmp_path / "approved"
    external = tmp_path / "external"
    approved.mkdir()
    external.mkdir()
    victim = external / "victim.bin"
    victim.write_bytes(b"must survive")
    junction = approved / "escape"
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(external)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip("junction creation unavailable")
    try:
        with pytest.raises(ValueError, match="reparse|junction"):
            _stage_tree(
                approved,
                tmp_path / "stage",
                mutable=False,
                selected=[],
                logical_prefix="data",
                approved_root=approved,
            )
        assert victim.read_bytes() == b"must survive"
        assert not (tmp_path / "stage" / "escape" / "victim.bin").exists()
    finally:
        if junction.exists():
            os.rmdir(junction)


def test_controller_has_explicit_outer_studio_bootstrap_margin() -> None:
    import scripts.smoke_packaged_workers as smoke

    assert smoke.READINESS_TIMEOUT_SECONDS == 120.0
    assert smoke.STUDIO_BOOTSTRAP_TIMEOUT_SECONDS == 180.0
    source = SCRIPT.read_text(encoding="utf-8")
    assert "timeout=STUDIO_BOOTSTRAP_TIMEOUT_SECONDS" in source


def test_smoke_start_assigns_job_before_resume_and_log_pump(tmp_path: Path, monkeypatch) -> None:
    import scripts.smoke_packaged_workers as smoke

    events: list[str] = []

    class Pipe(io.BytesIO):
        def read(self, size=-1):
            events.append("pump")
            return super().read(size)

    class Process:
        pid = 42
        _handle = 43
        stdout = Pipe(b"")

        def kill(self): events.append("kill")
        def wait(self, timeout=None): return 0
        def poll(self): return None

    def popen(*_args, **kwargs):
        events.append("popen-suspended" if kwargs["creationflags"] & smoke.CREATE_SUSPENDED else "popen-running")
        return Process()

    class Job:
        def __init__(self, _process): events.append("job")
        def close(self): events.append("job-close")

    monkeypatch.setattr(smoke.subprocess, "Popen", popen)
    monkeypatch.setattr(smoke, "_WindowsJob", Job)
    monkeypatch.setattr(smoke, "_resume_windows_process", lambda _process: events.append("resume"))
    logs = tmp_path / "logs"
    logs.mkdir()

    child = smoke._start(
        "ordered", ["worker"], cwd=tmp_path, env={}, logs=logs, secret="s" * 64
    )
    child.pump.join(timeout=2)

    assert events[:3] == ["popen-suspended", "job", "resume"]
    assert events.index("resume") < events.index("pump")


def test_smoke_start_reaps_child_if_post_resume_thread_start_fails(
    tmp_path: Path, monkeypatch
) -> None:
    import scripts.smoke_packaged_workers as smoke

    events: list[str] = []

    class Pipe(io.BytesIO):
        def close(self):
            events.append("pipe-close")
            super().close()

    class Process:
        pid = 42
        _handle = 43

        def __init__(self):
            self.stdout = Pipe(b"")

        def kill(self): events.append("kill")
        def wait(self, timeout=None): events.append("wait"); return 0
        def poll(self): return None

    class Job:
        def __init__(self, _process): events.append("job")
        def close(self): events.append("job-close")

    class Thread:
        def __init__(self, **_kwargs): pass
        def start(self): raise RuntimeError("thread start failed")

    monkeypatch.setattr(smoke.subprocess, "Popen", lambda *_args, **_kwargs: Process())
    monkeypatch.setattr(smoke, "_WindowsJob", Job)
    monkeypatch.setattr(smoke, "_resume_windows_process", lambda _process: events.append("resume"))
    monkeypatch.setattr(smoke.threading, "Thread", Thread)
    logs = tmp_path / "logs"
    logs.mkdir()

    with pytest.raises(RuntimeError, match="thread start failed"):
        smoke._start(
            "ordered", ["worker"], cwd=tmp_path, env={}, logs=logs, secret="s" * 64
        )

    assert events == ["job", "resume", "job-close", "kill", "wait", "pipe-close"]
