from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import pytest

import runtime.managed_rag_process as managed_module
from runtime.managed_rag_process import BoundedLog, ManagedRagProcess, _LogPump, _StreamingRedactor, managed_popen


class FakeChild:
    def __init__(self, *, exited: bool = False, timeout: bool = False) -> None:
        self.exited, self.timeout = exited, timeout
        self.calls: list[str] = []

    def poll(self):
        return 1 if self.exited else None

    def terminate(self):
        self.calls.append("terminate")

    def wait(self, timeout):
        self.calls.append("wait")
        if self.timeout and self.calls.count("wait") == 1:
            raise subprocess.TimeoutExpired("rag", timeout)

    def kill(self):
        self.calls.append("kill")


def test_managed_process_kills_tree_even_when_direct_child_has_already_exited():
    child = FakeChild(exited=True)
    managed = ManagedRagProcess(child)

    managed.close()
    managed.close()

    assert child.calls == ["terminate", "wait"]


def test_managed_process_kills_after_graceful_timeout_and_closes_handles():
    child = FakeChild(timeout=True)
    closed = []
    managed = ManagedRagProcess(child, closers=[lambda: closed.append(True)])

    managed.close()

    assert child.calls == ["terminate", "wait", "kill", "wait"]
    assert closed == [True]


def test_bounded_log_rotates_inside_the_runtime_log_directory(tmp_path: Path):
    log = BoundedLog(tmp_path / "logs", "rag", max_bytes=3, backups=2)
    log.directory.mkdir()
    log.path.write_bytes(b"abcd")

    stream = log.open()
    stream.close()

    assert log.path.parent == tmp_path / "logs"
    assert log.path.with_suffix(".log.1").read_bytes() == b"abcd"


def test_close_attempts_tree_kill_and_closes_handles_when_terminate_errors():
    child = FakeChild()
    closed = []

    def broken_terminate():
        raise RuntimeError("terminate failed")

    managed = ManagedRagProcess(
        child,
        tree_terminate=broken_terminate,
        tree_kill=lambda: child.calls.append("tree-kill"),
        closers=[lambda: closed.append(True)],
    )

    with pytest.raises(RuntimeError, match="terminate failed"):
        managed.close()

    assert child.calls == ["tree-kill", "wait"]
    assert closed == [True]


def test_managed_popen_pumps_redacted_child_output_into_bounded_logs(tmp_path: Path, monkeypatch):
    class Pipe:
        def __init__(self, lines):
            self.lines = iter(lines)
            self.closed = False

        def read(self, _size):
            return next(self.lines)

        def close(self):
            self.closed = True

    class Process(FakeChild):
        def __init__(self):
            super().__init__()
            self.pid = 42
            self.stdout = Pipe([b"session=secret\\n", b""])
            self.stderr = Pipe([b"error secret\\n", b""])

    process = Process()

    class Job:
        def __init__(self, _process):
            pass

        def close(self):
            pass

    monkeypatch.setattr("runtime.managed_rag_process.subprocess.Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr("runtime.managed_rag_process._WindowsJob", Job)
    monkeypatch.setattr("runtime.managed_rag_process._resume_windows_process", lambda _process: None)
    stdout = BoundedLog(tmp_path / "logs", "rag", max_bytes=64)
    stderr = BoundedLog(tmp_path / "logs", "rag-error", max_bytes=64)

    managed = managed_popen(
        ["worker"], cwd=str(tmp_path), env={}, stdout=stdout, stderr=stderr, redactions=(b"secret",),
    )
    managed.close()

    assert b"secret" not in stdout.path.read_bytes() + stderr.path.read_bytes()
    assert b"[REDACTED]" in stdout.path.read_bytes()
    assert process.stdout.closed and process.stderr.closed


def test_windows_managed_popen_assigns_job_before_resume_and_before_log_pumps(
    tmp_path: Path, monkeypatch
):
    events: list[str] = []

    class Pipe:
        def read(self, _size):
            events.append("pump")
            return b""

        def close(self):
            pass

    class Process(FakeChild):
        def __init__(self):
            super().__init__()
            self.pid = 42
            self._handle = 43
            self.stdout = Pipe()
            self.stderr = Pipe()

    process = Process()

    def popen(*_args, **kwargs):
        assert kwargs["creationflags"] & managed_module.CREATE_SUSPENDED
        events.append("popen-suspended")
        return process

    class Job:
        def __init__(self, _process):
            events.append("job")

        def close(self):
            events.append("job-close")

    monkeypatch.setattr("runtime.managed_rag_process._is_windows", lambda: True)
    monkeypatch.setattr("runtime.managed_rag_process.subprocess.Popen", popen)
    monkeypatch.setattr("runtime.managed_rag_process._WindowsJob", Job)
    monkeypatch.setattr(
        "runtime.managed_rag_process._resume_windows_process",
        lambda _process: events.append("resume"),
    )

    managed = managed_popen(
        ["worker"],
        cwd=str(tmp_path),
        env={},
        stdout=BoundedLog(tmp_path / "logs", "out"),
        stderr=BoundedLog(tmp_path / "logs", "err"),
    )
    for pump in managed._closers[:2]:
        # Pumps can finish concurrently; closing makes the observation deterministic.
        pump()

    assert events[:3] == ["popen-suspended", "job", "resume"]
    assert events.index("resume") < events.index("pump")


def test_windows_managed_popen_reaps_child_if_post_resume_log_setup_fails(
    tmp_path: Path, monkeypatch
):
    events: list[str] = []

    class Pipe:
        closed = False

        def close(self):
            self.closed = True

    class Process(FakeChild):
        def __init__(self):
            super().__init__()
            self.pid = 42
            self._handle = 43
            self.stdout = Pipe()
            self.stderr = Pipe()

    class Sink(BoundedLog):
        def close(self):
            events.append("sink-close")
            super().close()

    class Job:
        def __init__(self, _process):
            events.append("job")

        def close(self):
            events.append("job-close")

    process = Process()
    monkeypatch.setattr(managed_module, "_is_windows", lambda: True)
    monkeypatch.setattr(managed_module.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(managed_module, "_WindowsJob", Job)
    monkeypatch.setattr(managed_module, "_resume_windows_process", lambda _process: events.append("resume"))
    monkeypatch.setattr(
        managed_module,
        "_LogPump",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("pump setup failed")),
    )

    with pytest.raises(RuntimeError, match="pump setup failed"):
        managed_popen(
            ["worker"],
            cwd=str(tmp_path),
            env={},
            stdout=Sink(tmp_path / "logs", "out"),
            stderr=Sink(tmp_path / "logs", "err"),
        )

    assert events[:3] == ["job", "resume", "job-close"]
    assert process.calls == ["kill", "wait"]
    assert process.stdout.closed and process.stderr.closed
    assert events.count("sink-close") == 2


def test_log_pump_reads_bounded_chunks_and_redacts_split_case_insensitive_secret(tmp_path: Path):
    class Pipe:
        def __init__(self):
            self.chunks = iter([b"prefix Se", b"CrEt suffix", b""])
            self.read_sizes = []
            self.closed = False

        def read(self, size):
            self.read_sizes.append(size)
            return next(self.chunks)

        def close(self):
            self.closed = True

    pipe = Pipe()
    log = BoundedLog(tmp_path / "logs", "rag", max_bytes=64)
    pump = _LogPump(pipe, log, (b"secret",))
    pump.close()

    assert pipe.read_sizes and max(pipe.read_sizes) <= 8192
    assert b"secret" not in log.path.read_bytes().lower()
    assert b"[REDACTED]" in log.path.read_bytes()
    assert not pump.thread.is_alive()


@pytest.mark.parametrize("secret", [b"", b"s" * 513])
def test_streaming_redactor_rejects_an_unredactable_secret(secret):
    with pytest.raises(ValueError, match="512"):
        _StreamingRedactor((secret,))


def test_streaming_redactor_keeps_the_512_byte_secret_boundary_redacted_across_chunks():
    redactor = _StreamingRedactor((b"s" * 512,))
    result = redactor.feed(b"S" * 256) + redactor.feed(b"s" * 256) + redactor.finish()

    assert result == b"[REDACTED]"


def test_log_pump_never_requests_an_unbounded_read_for_huge_output_without_newlines(tmp_path: Path):
    class Pipe:
        def __init__(self):
            self.remaining = b"x" * (3 * 8192 + 17)
            self.read_sizes = []

        def read(self, size):
            self.read_sizes.append(size)
            chunk, self.remaining = self.remaining[:size], self.remaining[size:]
            return chunk

        def close(self):
            pass

    pipe = Pipe()
    log = BoundedLog(tmp_path / "logs", "rag", max_bytes=8192, backups=2)
    pump = _LogPump(pipe, log, ())
    pump.close()

    assert pipe.read_sizes and max(pipe.read_sizes) <= 8192
    assert all(path.stat().st_size <= 8192 for path in log.directory.glob("rag.log*"))


def test_log_pump_redacts_split_bearer_and_cookie_tokens_in_rotated_binary_logs(tmp_path: Path):
    class Pipe:
        def __init__(self):
            self.chunks = iter([
                b"\xffAuthorization: Bea", b"rer very-long-token; Cookie: sess",
                b"ion=another-long-token\r\n", b"filler" * 40, b"",
            ])

        def read(self, _size):
            return next(self.chunks)

        def close(self):
            pass

    log = BoundedLog(tmp_path / "logs", "rag", max_bytes=64, backups=2)
    pump = _LogPump(Pipe(), log, ())
    pump.close()
    files = [path for path in log.directory.glob("rag.log*") if path.is_file()]
    persisted = b"".join(path.read_bytes() for path in files).lower()

    assert files and all(path.stat().st_size <= 64 for path in files)
    assert b"very-long-token" not in persisted
    assert b"another-long-token" not in persisted
    assert b"[redacted]" in persisted


def test_log_pump_close_unblocks_a_blocked_pipe_and_joins_non_daemon_thread(tmp_path: Path):
    entered = threading.Event()
    released = threading.Event()

    class Pipe:
        def read(self, _size):
            entered.set()
            released.wait(1)
            return b""

        def close(self):
            released.set()

    pump = _LogPump(Pipe(), BoundedLog(tmp_path / "logs", "rag"), ())
    assert entered.wait(1)
    pump.close()

    assert not pump.thread.daemon
    assert not pump.thread.is_alive()


def test_log_pump_captures_reader_errors_and_does_not_leave_a_thread_alive(tmp_path: Path):
    class Pipe:
        def read(self, _size):
            raise OSError("broken pipe")

        def close(self):
            pass

    pump = _LogPump(Pipe(), BoundedLog(tmp_path / "logs", "rag"), ())
    pump.close()

    assert isinstance(pump.error, OSError)
    assert not pump.thread.is_alive()


def test_managed_popen_closes_supplied_handles_if_process_creation_fails(tmp_path: Path, monkeypatch):
    class Handle:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    stdout, stderr = Handle(), Handle()
    monkeypatch.setattr(
        "runtime.managed_rag_process.subprocess.Popen", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("nope")),
    )

    with pytest.raises(OSError, match="nope"):
        managed_popen(["worker"], cwd=str(tmp_path), env={}, stdout=stdout, stderr=stderr)

    assert stdout.closed and stderr.closed


def test_posix_process_group_ignores_esrch_after_child_exit(tmp_path: Path, monkeypatch):
    child = FakeChild(exited=True)
    child.pid = 7
    stdout = type("Handle", (), {"close": lambda self: None})()
    stderr = type("Handle", (), {"close": lambda self: None})()
    monkeypatch.setattr(managed_module, "_is_windows", lambda: False)
    monkeypatch.setattr(
        managed_module.os, "killpg", lambda *_args: (_ for _ in ()).throw(ProcessLookupError()), raising=False,
    )
    monkeypatch.setattr("runtime.managed_rag_process.subprocess.Popen", lambda *_args, **_kwargs: child)

    managed = managed_popen(["worker"], cwd=str(tmp_path), env={}, stdout=stdout, stderr=stderr)
    managed.close()

    assert child.calls == ["wait"]
