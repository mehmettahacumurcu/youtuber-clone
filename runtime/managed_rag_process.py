"""Contained RAG child lifecycle and bounded local log ownership."""
from __future__ import annotations

import os
import signal
import subprocess
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


CREATE_SUSPENDED = getattr(subprocess, "CREATE_SUSPENDED", 0x00000004)


class _WindowsJob:
    """Minimal kill-on-close Job Object wrapper; created only on Windows."""

    def __init__(self, process: Any) -> None:
        import ctypes
        from ctypes import wintypes

        class _Basic(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD),
            ]
        class _IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]
        class _Extended(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _Basic), ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        self._close = kernel32.CloseHandle
        self.handle = kernel32.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        limits = _Extended()
        limits.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
        if not kernel32.AssignProcessToJobObject(self.handle, wintypes.HANDLE(process._handle)):
            self.close()
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")

    def close(self) -> None:
        if self.handle:
            self._close(self.handle)
            self.handle = None


def _is_windows() -> bool:
    return os.name == "nt"


def _resume_windows_process(process: Any) -> None:
    """Resume a process only after its entire future tree is owned by a Job."""
    import ctypes
    from ctypes import wintypes

    ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.restype = ctypes.c_long
    status = int(ntdll.NtResumeProcess(wintypes.HANDLE(process._handle)))
    if status != 0:
        raise OSError(status, "NtResumeProcess failed")


class BoundedLog:
    def __init__(self, directory: Path, name: str, *, max_bytes: int = 10 * 1024 * 1024, backups: int = 5) -> None:
        self.directory = Path(directory).resolve(strict=False)
        self.path = self.directory / f"{name}.log"
        self.max_bytes = max_bytes
        self.backups = backups
        self._lock = threading.Lock()
        self._stream = None

    def open(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size >= self.max_bytes:
            for index in range(self.backups - 1, 0, -1):
                source = self.path.with_suffix(f".log.{index}")
                target = self.path.with_suffix(f".log.{index + 1}")
                if source.exists():
                    source.replace(target)
            self.path.replace(self.path.with_suffix(".log.1"))
        return self.path.open("ab", buffering=0)

    def write(self, data: bytes) -> None:
        with self._lock:
            if self._stream is None:
                self._stream = self.open()
            if self.path.stat().st_size + len(data) > self.max_bytes:
                self._stream.close()
                self._stream = None
                if self.path.exists():
                    for index in range(self.backups - 1, 0, -1):
                        source = self.path.with_suffix(f".log.{index}")
                        target = self.path.with_suffix(f".log.{index + 1}")
                        if source.exists(): source.replace(target)
                    self.path.replace(self.path.with_suffix(".log.1"))
                self._stream = self.open()
            self._stream.write(data[:self.max_bytes])

    def close(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None


class _LogPump:
    _READ_CHUNK_BYTES = 8 * 1024

    def __init__(self, source: Any, sink: BoundedLog, redactions: tuple[bytes, ...]) -> None:
        self.source, self.sink = source, sink
        self.redactor = _StreamingRedactor(redactions)
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def _run(self) -> None:
        try:
            while data := self.source.read(self._READ_CHUNK_BYTES):
                self.sink.write(self.redactor.feed(data))
            self.sink.write(self.redactor.finish())
        except BaseException as exc:
            self.error = exc

    def close(self) -> None:
        try:
            self.source.close()
        except Exception:
            pass
        self.thread.join()


class _StreamingRedactor:
    """Bounded binary-safe redactor for child output that may split tokens."""

    _REPLACEMENT = b"[REDACTED]"
    _HEADER_PATTERNS = (b"authorization: bearer ", b"bearer ", b"cookie: ", b"session=", b"cookie=")
    _TOKEN_END = b" \t\r\n;"
    _MAX_CARRY = 512

    def __init__(self, secrets: tuple[bytes, ...]) -> None:
        if any(not secret or len(secret) > self._MAX_CARRY for secret in secrets):
            raise ValueError("log redaction secrets must contain between 1 and 512 bytes")
        self._secrets = secrets
        self._patterns = (*self._secrets, *self._HEADER_PATTERNS)
        self._carry_limit = min(
            self._MAX_CARRY,
            max((len(pattern) for pattern in self._patterns), default=1),
        )
        self._pending = b""
        self._discard_until = False

    def feed(self, data: bytes) -> bytes:
        if not isinstance(data, bytes):
            data = bytes(data)
        self._pending += data
        return self._drain(final=False)

    def finish(self) -> bytes:
        output = bytearray(self._drain(final=True))
        # A child can exit after writing only the beginning of a secret. Treat a
        # terminal secret prefix as sensitive rather than leaking that fragment.
        lowered = bytes(output).lower()
        prefix_length = max(
            (
                length
                for secret in self._secrets
                for length in range(1, len(secret))
                if lowered.endswith(secret[:length].lower())
            ),
            default=0,
        )
        if prefix_length:
            del output[-prefix_length:]
            output.extend(self._REPLACEMENT)
        return bytes(output)

    def _drain(self, *, final: bool) -> bytes:
        out = bytearray()
        while self._pending:
            if self._discard_until:
                boundary = next((index for index, char in enumerate(self._pending) if char in self._TOKEN_END), None)
                if boundary is None:
                    # The token body is intentionally never buffered; it may
                    # be arbitrarily long and is already represented by one
                    # replacement marker in the output.
                    self._pending = b""
                    return bytes(out)
                self._pending = self._pending[boundary:]
                self._discard_until = False
                continue
            lowered = self._pending.lower()
            matches = [(lowered.find(pattern.lower()), pattern) for pattern in self._patterns]
            matches = [(index, pattern) for index, pattern in matches if index >= 0]
            if matches:
                index, pattern = min(matches, key=lambda item: item[0])
                out.extend(self._pending[:index])
                if pattern in self._secrets:
                    out.extend(self._REPLACEMENT)
                    self._pending = self._pending[index + len(pattern):]
                else:
                    out.extend(self._pending[index:index + len(pattern)])
                    self._pending = self._pending[index + len(pattern):]
                    self._discard_until = True
                    out.extend(self._REPLACEMENT)
                continue
            keep = 0 if final else self._carry_limit - 1
            if len(self._pending) <= keep:
                return bytes(out)
            emit = len(self._pending) - keep
            out.extend(self._pending[:emit])
            self._pending = self._pending[emit:]
        return bytes(out)


class ManagedRagProcess:
    """Own child-tree termination and log handles; ``close`` is idempotent."""

    def __init__(
        self,
        process: Any,
        *,
        closers: Iterable[Callable[[], None]] = (),
        tree_terminate: Callable[[], None] | None = None,
        tree_kill: Callable[[], None] | None = None,
    ) -> None:
        self.process = process
        self._closers = list(closers)
        self._tree_terminate = tree_terminate or process.terminate
        self._tree_kill = tree_kill or process.kill
        self._closed = False

    def poll(self):
        return self.process.poll()

    def close(self, timeout: float = 3.0) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            # A descendant may survive a crashed/exited parent, so always address the tree.
            try:
                self._tree_terminate()
                self.process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, TimeoutError):
                self._tree_kill()
                self.process.wait(timeout=timeout)
            except BaseException:
                # Preserve the termination error, but still make one best-effort
                # tree kill before surfacing it to the caller.
                try:
                    self._tree_kill()
                    self.process.wait(timeout=timeout)
                except BaseException:
                    pass
                raise
        finally:
            for closer in self._closers:
                try:
                    closer()
                except Exception:
                    pass


def managed_popen(command: list[str], *, cwd: str, env: dict[str, str], stdout: Any, stderr: Any, redactions: tuple[bytes, ...] = ()) -> ManagedRagProcess:
    """Start in a new POSIX process group; Windows uses the launcher Job Object in production."""
    sinks = isinstance(stdout, BoundedLog) and isinstance(stderr, BoundedLog)
    kwargs: dict[str, Any] = {"cwd": cwd, "env": env, "stdout": subprocess.PIPE if sinks else stdout, "stderr": subprocess.PIPE if sinks else stderr, "shell": False}
    if not _is_windows():
        kwargs["start_new_session"] = True
    else:
        kwargs["creationflags"] = CREATE_SUSPENDED
    try:
        process = subprocess.Popen(command, **kwargs)
    except Exception:
        for handle in (stdout, stderr):
            getattr(handle, "close", lambda: None)()
        raise
    sink_closers = [getattr(stdout, "close", lambda: None), getattr(stderr, "close", lambda: None)]
    closers = list(sink_closers)
    job = None
    if _is_windows():
        try:
            job = _WindowsJob(process)
            _resume_windows_process(process)
        except Exception:
            if job is not None:
                try:
                    job.close()
                except Exception:
                    pass
            try:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=3)
            except Exception:
                pass
            for pipe in (getattr(process, "stdout", None), getattr(process, "stderr", None)):
                try:
                    if pipe is not None:
                        pipe.close()
                except Exception:
                    pass
            for closer in sink_closers:
                try:
                    closer()
                except Exception:
                    pass
            raise

    pumps = []
    try:
        if sinks:
            for source, sink in ((process.stdout, stdout), (process.stderr, stderr)):
                pumps.append(_LogPump(source, sink, redactions))
            closers = [pump.close for pump in pumps] + closers
    except Exception:
        if job is not None:
            try:
                job.close()
            except Exception:
                pass
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)
        except Exception:
            pass
        for pump in pumps:
            try:
                pump.close()
            except Exception:
                pass
        for pipe in (getattr(process, "stdout", None), getattr(process, "stderr", None)):
            try:
                if pipe is not None:
                    pipe.close()
            except Exception:
                pass
        for closer in sink_closers:
            try:
                closer()
            except Exception:
                pass
        raise
    if not _is_windows():
        def terminate_group() -> None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        def kill_group() -> None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return ManagedRagProcess(
            process,
            tree_terminate=terminate_group, tree_kill=kill_group,
            closers=closers,
        )
    assert job is not None
    return ManagedRagProcess(
        process, tree_kill=job.close,
        closers=[*closers, job.close],
    )
