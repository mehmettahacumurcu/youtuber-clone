"""Exercise the frozen Windows workers through their loopback HTTP boundaries.

Fixture mode deliberately avoids model loads.  It runs the real Voice service and
the real Studio executable's ``--rag-worker`` dispatch with a structurally valid,
empty runtime layout.  Real-assets mode extends this gate with the complete Studio,
retrieval, answer, and voice paths.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime.managed_rag_process import (
    _StreamingRedactor,
    _WindowsJob,
    _resume_windows_process,
)


STUDIO_EXE = ROOT / "dist" / "workers" / "studio-worker" / "studio-worker.exe"
VOICE_EXE = ROOT / "dist" / "workers" / "voice-worker" / "voice-worker.exe"
MAX_HTTP_BYTES = 2 * 1024 * 1024
MAX_LOG_BYTES = 10 * 1024 * 1024
MAX_TAIL_LINE_CHARS = 8192
READINESS_TIMEOUT_SECONDS = 120.0
STUDIO_BOOTSTRAP_TIMEOUT_SECONDS = 180.0
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
CREATE_SUSPENDED = getattr(subprocess, "CREATE_SUSPENDED", 0x00000004)
_OWNERSHIP_MARKER = ".youtuber-packaged-smoke-output"
_STOCK_NON_ANSWERS = frozenset(
    {
        "Boş model çıktısı alındı; yanıt üretilemedi.",
        "RAG kanıt denetimi şu anda çalışmadığı için cevap üretmedim.",
        "Bu konuya pek değinmemişim, elimde bununla ilgili bir şey yok.",
    }
)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


@dataclass(frozen=True)
class HttpResult:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> dict[str, Any]:
        value = json.loads(self.body.decode("utf-8"))
        if type(value) is not dict:
            raise RuntimeError("loopback response is not a JSON object")
        return value


def _http(
    method: str,
    url: str,
    *,
    bearer: str | None = None,
    payload: object | None = None,
    timeout: float = 5.0,
    extra_headers: dict[str, str] | None = None,
    max_bytes: int = MAX_HTTP_BYTES,
) -> HttpResult:
    headers = {"Accept": "application/json"}
    headers.update(extra_headers or {})
    body = None
    if bearer is not None:
        headers["Authorization"] = "Bearer " + bearer
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise RuntimeError("loopback response exceeded the safety cap")
        return HttpResult(
            status=int(response.status),
            headers={str(key).casefold(): str(value) for key, value in response.headers.items()},
            body=data,
        )


def _validated_chat_answer(chat_payload: dict[str, Any]) -> str:
    answer = chat_payload.get("answer")
    if (
        chat_payload.get("generation_status") != "generated"
        or type(answer) is not str
        or not answer.strip()
        or answer.strip() in _STOCK_NON_ANSWERS
        or answer.startswith("[LLM error:")
        or "cevap üretmedim" in answer.casefold()
    ):
        raise RuntimeError("frozen Studio returned no genuine model-generated answer")
    return answer


class ProcessLog:
    """Bounded, secret-redacted process log with a 200-line diagnostic tail."""

    def __init__(self, path: Path, secret: str) -> None:
        self.path = path
        self._secret = secret.encode("utf-8")
        self._lines: collections.deque[str] = collections.deque(maxlen=200)
        self._written = 0
        self._lock = threading.Lock()

    def pump(self, source: Any) -> None:
        redactor = _StreamingRedactor((self._secret,))
        with self.path.open("wb") as target:
            while True:
                chunk = source.read(8192)
                if not chunk:
                    break
                self._write_redacted(target, redactor.feed(bytes(chunk)))
            self._write_redacted(target, redactor.finish())
        source.close()

    def _write_redacted(self, target: Any, data: bytes) -> None:
        if not data:
            return
        # The streaming matcher is authoritative. This exact replacement is a
        # second barrier if a future redactor regression ever reaches this sink.
        clean = data.replace(self._secret, b"[REDACTED]")
        remaining = max(0, MAX_LOG_BYTES - self._written)
        persisted = clean[:remaining]
        if persisted:
            target.write(persisted)
            self._written += len(persisted)
            text = persisted.decode("utf-8", errors="replace")
            with self._lock:
                self._lines.extend(
                    line[:MAX_TAIL_LINE_CHARS] for line in text.splitlines()
                )

    def tail(self) -> list[str]:
        with self._lock:
            secret = self._secret.decode("utf-8", errors="strict")
            return [line.replace(secret, "[REDACTED]") for line in self._lines]


@dataclass
class Child:
    name: str
    process: subprocess.Popen[bytes]
    log: ProcessLog
    pump: threading.Thread
    job: Any | None = None


def _free_ports(count: int = 3) -> list[int]:
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            sockets.append(listener)
        ports = [int(listener.getsockname()[1]) for listener in sockets]
        if len(set(ports)) != count:
            raise RuntimeError("failed to allocate distinct worker ports")
        return ports
    finally:
        for listener in sockets:
            listener.close()


def _base_environment(
    install_root: Path,
    data_root: Path,
    cache_root: Path,
    ports: list[int],
    secret: str,
    *,
    ollama_origin: str = "http://127.0.0.1:11434",
    answer_model: str = "speaker-v5-a636",
    verifier_model: str = "qwen3:4b",
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "YOUTUBER_INSTALL_ROOT": str(install_root.resolve()),
            "YOUTUBER_DATA_ROOT": str(data_root.resolve()),
            "YOUTUBER_CACHE_ROOT": str(cache_root.resolve()),
            "YOUTUBER_STUDIO_PORT": str(ports[0]),
            "YOUTUBER_RAG_PORT": str(ports[1]),
            "YOUTUBER_VOICE_PORT": str(ports[2]),
            "YOUTUBER_OLLAMA_ORIGIN": ollama_origin,
            "YOUTUBER_ANSWER_MODEL": answer_model,
            "YOUTUBER_VERIFIER_MODEL": verifier_model,
            "YOUTUBER_SESSION_SECRET": secret,
            "YOUTUBER_LOW_VRAM_VOICE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _start(name: str, command: list[str], *, cwd: Path, env: dict[str, str], logs: Path, secret: str) -> Child:
    log = ProcessLog(logs / f"{name}.log", secret)
    creationflags = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    if os.name == "nt":
        creationflags |= CREATE_SUSPENDED
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        shell=False,
    )
    job = None
    if os.name == "nt":
        try:
            job = _WindowsJob(process)
            _resume_windows_process(process)
        except BaseException:
            if job is not None:
                try:
                    job.close()
                except Exception:
                    pass
            try:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
            finally:
                if process.stdout is not None:
                    process.stdout.close()
            raise
    assert process.stdout is not None
    try:
        pump = threading.Thread(target=log.pump, args=(process.stdout,), name=f"{name}-log", daemon=False)
        pump.start()
    except BaseException:
        if job is not None:
            try:
                job.close()
            except Exception:
                pass
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
        finally:
            process.stdout.close()
        raise
    return Child(name=name, process=process, log=log, pump=pump, job=job)


def _await_http(child: Child, url: str, *, expected: set[int], timeout: float = READINESS_TIMEOUT_SECONDS) -> HttpResult:
    deadline = time.monotonic() + timeout
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if child.process.poll() is not None:
            raise RuntimeError(f"{child.name} exited before readiness (exit={child.process.returncode})")
        try:
            result = _http("GET", url, timeout=2.0)
            if result.status in expected:
                return result
            last_error = RuntimeError(f"unexpected HTTP {result.status}")
        except Exception as error:
            last_error = error
        time.sleep(0.1)
    raise RuntimeError(f"{child.name} readiness timed out: {type(last_error).__name__ if last_error else 'unknown'}")


def _terminate_tree(child: Child) -> int | None:
    process = child.process
    if child.job is not None:
        # Closing KILL_ON_JOB_CLOSE owns descendants even after a dead parent.
        child.job.close()
    if process.poll() is None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW,
                check=False,
                timeout=15,
            )
        else:
            process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)
    child.pump.join(timeout=5)
    return process.returncode


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        completed = subprocess.run(
            ["tasklist.exe", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=CREATE_NO_WINDOW,
            timeout=5,
            check=False,
        )
        return f'"{pid}"' in completed.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _wait_pids_gone(pids: set[int], *, timeout: float = 10.0) -> list[int]:
    """Allow Windows a bounded interval to reap processes after taskkill returns."""
    pending = sorted(pids)
    deadline = time.monotonic() + timeout
    while pending:
        pending = [pid for pid in pending if _pid_alive(pid)]
        if not pending or time.monotonic() >= deadline:
            return pending
        time.sleep(0.1)
    return []


def _ports_released(ports: list[int]) -> bool:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        bound: list[socket.socket] = []
        try:
            for port in ports:
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.bind(("127.0.0.1", port))
                bound.append(listener)
            return True
        except OSError:
            time.sleep(0.1)
        finally:
            for listener in bound:
                listener.close()
    return False


def _prepare_output_root(path: Path) -> Path:
    """Create or safely reset a directory owned by this smoke tool."""
    output = Path(path).resolve()
    if output.exists():
        if not (output / _OWNERSHIP_MARKER).is_file():
            raise ValueError(f"output directory is not owned by the packaged smoke: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    (output / _OWNERSHIP_MARKER).write_text("youtuber.packaged-smoke.v1\n", encoding="ascii")
    return output


def build_voice_manifest(
    selected: dict[str, tuple[int, str]], *, reference_paths: tuple[str, ...]
) -> dict[str, Any]:
    """Build the fixed public voice selection from already-hashed staged files."""
    return {
        "schema": "youtuber.voice.v1",
        "runtime_api": 1,
        "sample_rate": 24_000,
        "audio_format": "WAV",
        "files": [
            {"path": path, "size": size, "sha256": digest}
            for path, (size, digest) in sorted(selected.items())
        ],
        "xtts": {
            "checkpoint": "xtts/best_model.pth",
            "config": "xtts/config.json",
            "vocab": "xtts/vocab.json",
            "reference_wavs": list(reference_paths),
        },
        "rvc": {
            "weight": "rvc/speaker-e200.pth",
            "index": "rvc/speaker-e200.index",
            "epoch": 200,
            "index_rate": 0.75,
            "pitch": 0,
            "f0_method": "rmvpe",
            "protect": 0.5,
        },
        "contentvec": {
            "config": "contentvec/config.json",
            "model": "contentvec/pytorch_model.bin",
        },
        "rmvpe": "rmvpe/rmvpe.pt",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_reparse(path: Path) -> bool:
    try:
        return bool(path.lstat().st_file_attributes & 0x400)
    except (AttributeError, OSError):
        return path.is_symlink()


def _absolute_without_resolve(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _validate_approved_source_root(path: Path) -> tuple[Path, Path]:
    """Validate every raw component before any operation can follow it."""
    raw = _absolute_without_resolve(path)
    current = Path(raw.anchor)
    for part in raw.parts[1:]:
        current /= part
        try:
            current.lstat()
        except OSError as exc:
            raise ValueError(f"approved source root component is unavailable: {current}") from exc
        if current.is_symlink() or _is_reparse(current):
            raise ValueError(f"approved source root contains a symlink or reparse point: {current}")
        if not current.is_dir():
            raise ValueError(f"approved source root component is not a directory: {current}")
    return raw, raw.resolve(strict=True)


def _checked_source_path(source: Path, approved_root: Path, *, directory: bool) -> Path:
    raw_root, resolved_root = _validate_approved_source_root(approved_root)
    raw_source = _absolute_without_resolve(source)
    try:
        relative = raw_source.relative_to(raw_root)
    except ValueError as exc:
        raise ValueError(f"selected asset is outside the approved source root: {raw_source}") from exc
    current = raw_root
    for index, part in enumerate(relative.parts):
        current /= part
        try:
            current.lstat()
        except OSError as exc:
            raise ValueError(f"selected asset component is unavailable: {current}") from exc
        if current.is_symlink() or _is_reparse(current):
            raise ValueError(f"selected asset contains a symlink or reparse point: {current}")
        if index < len(relative.parts) - 1 and not current.is_dir():
            raise ValueError(f"selected asset parent is not a directory: {current}")
    resolved = raw_source.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"selected asset resolves outside the approved source root: {raw_source}") from exc
    if directory and not resolved.is_dir():
        raise ValueError(f"selected asset root is not a directory: {raw_source}")
    if not directory and not resolved.is_file():
        raise ValueError(f"selected asset is not a regular file: {raw_source}")
    return resolved


def _stage_file(
    source: Path,
    destination: Path,
    *,
    mutable: bool,
    selected: list[dict[str, Any]],
    logical_path: str,
    approved_root: Path,
) -> None:
    source = _checked_source_path(source, approved_root, directory=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    method = "copy"
    if not mutable:
        try:
            os.link(source, destination)
            method = "hardlink"
        except OSError:
            shutil.copy2(source, destination)
    else:
        shutil.copy2(source, destination)
    source_hash = _sha256_file(source)
    staged_hash = _sha256_file(destination)
    if source_hash != staged_hash or source.stat().st_size != destination.stat().st_size:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"staged asset integrity mismatch: {logical_path}")
    selected.append(
        {
            "logical_path": logical_path,
            "source": str(source),
            "source_size": source.stat().st_size,
            "source_sha256": source_hash,
            "staged_size": destination.stat().st_size,
            "staged_sha256": staged_hash,
            "method": method,
            "consumer_access": "read_write" if mutable else "read_only",
        }
    )


def _verify_hardlinked_source_integrity(selection: dict[str, Any]) -> int:
    """Prove every read-only hardlink consumer left its source inode unchanged."""
    verified = 0
    for row in selection["files"]:
        if row["method"] != "hardlink":
            continue
        if row.get("consumer_access") != "read_only":
            raise RuntimeError("a writable consumer was staged through a hardlink")
        source = Path(row["source"])
        if (
            not source.is_file()
            or source.stat().st_size != row["source_size"]
            or _sha256_file(source) != row["source_sha256"]
        ):
            raise RuntimeError(
                f"hardlinked source changed during packaged gate: {row['logical_path']}"
            )
        verified += 1
    return verified


def _stage_tree(
    source_root: Path,
    destination_root: Path,
    *,
    mutable: bool,
    selected: list[dict[str, Any]],
    logical_prefix: str,
    approved_root: Path,
    exclude_names: set[str] | None = None,
) -> None:
    source_root = _checked_source_path(source_root, approved_root, directory=True)
    excluded = exclude_names or set()

    def walk(directory: Path):
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name.casefold())
        for entry in ordered:
            source = Path(entry.path)
            source.lstat()
            if entry.is_symlink() or _is_reparse(source):
                raise ValueError(f"selected asset tree contains a symlink, junction, or reparse point: {source}")
            if entry.is_dir(follow_symlinks=False):
                yield from walk(source)
            elif entry.is_file(follow_symlinks=False):
                yield source
            else:
                raise ValueError(f"selected asset tree contains a non-regular entry: {source}")

    for source in walk(source_root):
        if source.name in excluded:
            continue
        relative = source.relative_to(source_root)
        _stage_file(
            source,
            destination_root / relative,
            mutable=mutable,
            selected=selected,
            logical_path=f"{logical_prefix}/{relative.as_posix()}",
            approved_root=approved_root,
        )


def _canonical_tree_fingerprint(rows: list[dict[str, Any]], prefix: str) -> str:
    values = [
        (row["logical_path"], row["staged_size"], row["staged_sha256"])
        for row in rows
        if str(row["logical_path"]).startswith(prefix)
    ]
    payload = json.dumps(sorted(values), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _validate_ollama_models() -> dict[str, Any]:
    result = _http("GET", "http://127.0.0.1:11434/api/tags", timeout=5.0)
    if result.status != 200:
        raise RuntimeError("the existing fixed-origin Ollama service is unavailable")
    models = result.json().get("models")
    if type(models) is not list:
        raise RuntimeError("Ollama tag response is malformed")
    names = {item.get("name") for item in models if type(item) is dict}
    answer = "speaker-v5-a636:latest"
    verifier = "qwen3:4b"
    if answer not in names or verifier not in names:
        raise RuntimeError("required real-smoke Ollama identities are unavailable")
    return {"answer_model": answer, "verifier_model": verifier, "available": sorted(name for name in names if isinstance(name, str))}


def _qdrant_counts(path: Path) -> dict[str, int]:
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(path))
    try:
        result = {}
        for name in ("speaker_clean", "speaker_cards"):
            if not client.collection_exists(name):
                raise RuntimeError(f"staged Qdrant collection is missing: {name}")
            result[name] = int(client.count(name, exact=True).count)
        return result
    finally:
        client.close()


def stage_real_assets(source_root: Path, output_root: Path) -> dict[str, Any]:
    """Create a temporary immutable selection without altering any source artifact."""
    _, source_root = _validate_approved_source_root(source_root)
    output_root = _prepare_output_root(output_root)
    install_root = output_root / "install"
    data_root = output_root / "data"
    cache_root = output_root / "cache"
    selected: list[dict[str, Any]] = []
    for directory in (install_root / "runtime", data_root, cache_root, output_root / "logs"):
        directory.mkdir(parents=True, exist_ok=True)

    _stage_file(
        source_root / "config.yaml",
        install_root / "runtime" / "config.yaml",
        mutable=False,
        selected=selected,
        logical_path="install/runtime/config.yaml",
        approved_root=source_root,
    )
    _stage_tree(
        source_root / "data" / "clean",
        data_root / "clean",
        mutable=False,
        selected=selected,
        logical_prefix="data/clean",
        approved_root=source_root,
    )
    _stage_tree(
        source_root / "data" / "rag" / "qdrant",
        data_root / "rag" / "qdrant",
        mutable=True,
        selected=selected,
        logical_prefix="data/rag/qdrant",
        approved_root=source_root,
        exclude_names={".lock"},
    )
    for name in ("speaker_cards.json", "speaker_cards.index_manifest.json"):
        _stage_file(
            source_root / "data" / "cards" / name,
            data_root / "cards" / name,
            mutable=name.endswith("manifest.json"),
            selected=selected,
            logical_path=f"data/cards/{name}",
            approved_root=source_root,
        )

    from rag.card_manifest import relocate_card_manifest

    card_manifest = data_root / "cards" / "speaker_cards.index_manifest.json"
    relocate_card_manifest(
        catalog_path=data_root / "cards" / "speaker_cards.json",
        manifest_path=card_manifest,
        destination_store_path=data_root / "rag" / "qdrant",
        collection="speaker_cards",
        embedder="BAAI/bge-m3",
    )
    for row in selected:
        if row["logical_path"] == "data/cards/speaker_cards.index_manifest.json":
            row["staged_size"] = card_manifest.stat().st_size
            row["staged_sha256"] = _sha256_file(card_manifest)
            row["transformed"] = "relocated staged store_path"

    counts = _qdrant_counts(data_root / "rag" / "qdrant")
    if counts["speaker_clean"] <= 0 or counts["speaker_cards"] != 2371:
        raise RuntimeError(f"unexpected staged RAG collection counts: {counts}")
    rebuild = {
        "schema": "youtuber.rag.v1",
        "corpus_version": "local-" + _canonical_tree_fingerprint(selected, "data/clean/")[:16],
        "corpus_fingerprint": _canonical_tree_fingerprint(selected, "data/clean/"),
        "index_fingerprint": _canonical_tree_fingerprint(selected, "data/rag/qdrant/"),
        "clean_collection": "speaker_clean",
        "clean_count": counts["speaker_clean"],
    }
    _write_json(data_root / "rag" / "rebuild_manifest.json", rebuild)

    voice_root = data_root / "models" / "voice"
    voice_sources = {
        "xtts/best_model.pth": source_root / "models" / "xtts_speaker_v2" / "best_model.pth",
        "xtts/config.json": source_root / "models" / "xtts_speaker_v2" / "config.json",
        "xtts/vocab.json": source_root / "models" / "xtts_speaker_v2" / "vocab.json",
        "references/speaker_reference.wav": source_root / "data" / "reference" / "speaker_reference.wav",
        "rvc/speaker-e200.pth": source_root / "results" / "demo_showcase_v1" / "rvc_model" / "artifacts" / "speaker_rvc_full_v1_200e_15800s.pth",
        "rvc/speaker-e200.index": source_root / "results" / "demo_showcase_v1" / "rvc_model" / "artifacts" / "speaker_rvc_full_v1.index",
        "contentvec/config.json": source_root / "results" / "demo_showcase_v1" / "runtime" / "Applio" / "rvc" / "models" / "embedders" / "contentvec" / "config.json",
        "contentvec/pytorch_model.bin": source_root / "results" / "demo_showcase_v1" / "runtime" / "Applio" / "rvc" / "models" / "embedders" / "contentvec" / "pytorch_model.bin",
        "rmvpe/rmvpe.pt": source_root / "results" / "demo_showcase_v1" / "runtime" / "Applio" / "rvc" / "models" / "predictors" / "rmvpe.pt",
    }
    voice_selected: dict[str, tuple[int, str]] = {}
    for logical, source in voice_sources.items():
        _stage_file(
            source,
            voice_root / logical,
            mutable=False,
            selected=selected,
            logical_path=f"data/models/voice/{logical}",
            approved_root=source_root,
        )
        destination = voice_root / logical
        voice_selected[logical] = (destination.stat().st_size, _sha256_file(destination))
    if voice_selected["rvc/speaker-e200.pth"][1] != "a68b9190622f10eccab3432a1e18ed8a20dd1361509bc1fbc75a6c64dcda1a61":
        raise RuntimeError("selected RVC weight is not the locked epoch-200 artifact")
    if voice_selected["rvc/speaker-e200.index"][1] != "ef9e0c8628c6457bbe26cb3df0a02c11b79306dd6dbee26978558ad10c424141":
        raise RuntimeError("selected RVC index is not the locked epoch-200 index")
    _write_json(
        voice_root / "manifest.json",
        build_voice_manifest(voice_selected, reference_paths=("references/speaker_reference.wav",)),
    )
    selection = {
        "schema": "youtuber.smoke-asset-selection.v1",
        "source_root": str(source_root),
        "counts": counts,
        "generated_rebuild_manifest": True,
        "files": selected,
    }
    _write_json(output_root / "asset-selection.json", selection)
    return {
        "output_root": output_root,
        "install_root": install_root,
        "data_root": data_root,
        "cache_root": cache_root,
        "logs": output_root / "logs",
        "selection": selection,
        "rag_identity": rebuild | {"card_count": counts["speaker_cards"]},
    }


def _ollama_loaded_models() -> list[str]:
    result = _http("GET", "http://127.0.0.1:11434/api/ps", timeout=5.0)
    models = result.json().get("models") if result.status == 200 else None
    if type(models) is not list:
        raise RuntimeError("Ollama residency response is malformed")
    names = [item.get("name") for item in models if type(item) is dict]
    if any(type(name) is not str for name in names):
        raise RuntimeError("Ollama residency identity is malformed")
    return sorted(names)


def _ensure_ollama_model_resident(model: str) -> None:
    """Load one required model without replacing or stopping the Ollama service."""
    response = _http(
        "POST",
        "http://127.0.0.1:11434/api/generate",
        payload={"model": model, "prompt": "", "stream": False, "keep_alive": -1},
        timeout=180.0,
    )
    if response.status != 200:
        raise RuntimeError(f"Ollama could not restore required model {model}")
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if model in _ollama_loaded_models():
            return
        time.sleep(0.25)
    raise RuntimeError(f"Ollama did not confirm restored model {model}")


def _ensure_ollama_model_absent(model: str) -> None:
    from rag.ollama_residency import ensure_model_absent

    ensure_model_absent(
        model,
        base_url="http://127.0.0.1:11434",
        timeout_s=60.0,
        poll_interval_s=0.25,
    )


def _restore_ollama_residency(
    *,
    initial: list[str],
    required: set[str],
    loaded: Any = _ollama_loaded_models,
    ensure_present: Any = _ensure_ollama_model_resident,
    ensure_absent: Any = _ensure_ollama_model_absent,
) -> dict[str, Any]:
    """Restore required identities and compute unrelated-model noninterference."""
    initial_set = set(initial)
    current = set(loaded())
    for model in sorted(required):
        if model in initial_set:
            ensure_present(model)
        elif model in current:
            ensure_absent(model)
        current = set(loaded())
    after = sorted(loaded())
    after_set = set(after)
    required_restored = {
        model for model in required if model in after_set
    } == {model for model in required if model in initial_set}
    unrelated_unchanged = (after_set - required) == (initial_set - required)
    return {
        "before": sorted(initial_set),
        "after": after,
        "required_restored": required_restored,
        "unrelated_unchanged": unrelated_unchanged,
        "restored": required_restored and unrelated_unchanged,
    }


def _cookie_from_bootstrap(response: HttpResult, secret: str) -> str:
    raw = response.headers.get("set-cookie", "")
    lowered = raw.casefold()
    if response.status != 303 or "httponly" not in lowered or "samesite=strict" not in lowered:
        raise RuntimeError("Studio bootstrap cookie is not strict HttpOnly/SameSite")
    if secret in raw or secret.encode("utf-8") in response.body:
        raise RuntimeError("Studio bootstrap exposed the launcher secret")
    parsed = SimpleCookie()
    parsed.load(raw)
    if len(parsed) != 1:
        raise RuntimeError("Studio bootstrap returned an invalid cookie")
    morsel = next(iter(parsed.values()))
    return f"{morsel.key}={morsel.value}"


def _process_tree_pids(root_pid: int) -> list[int]:
    if os.name != "nt":
        return [root_pid]
    command = (
        "$ErrorActionPreference='Stop';"
        "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId | ConvertTo-Json -Compress"
    )
    completed = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=CREATE_NO_WINDOW,
        timeout=15,
        check=True,
    )
    raw = json.loads(completed.stdout)
    rows = raw if isinstance(raw, list) else [raw]
    by_parent: dict[int, list[int]] = {}
    for row in rows:
        if type(row) is not dict:
            continue
        try:
            pid, parent = int(row["ProcessId"]), int(row["ParentProcessId"])
        except (KeyError, TypeError, ValueError):
            continue
        by_parent.setdefault(parent, []).append(pid)
    result, pending = [], [root_pid]
    while pending:
        pid = pending.pop()
        if pid in result:
            continue
        result.append(pid)
        pending.extend(by_parent.get(pid, ()))
    return sorted(result)


def _preserve_internal_logs(cache_root: Path, logs_root: Path, secret: str) -> None:
    source_root = cache_root / "logs"
    if not source_root.is_dir():
        return
    token = secret.encode("utf-8")
    for source in source_root.glob("*.log*"):
        if not source.is_file() or source.is_symlink() or _is_reparse(source):
            continue
        with source.open("rb") as handle:
            if source.stat().st_size > MAX_LOG_BYTES:
                handle.seek(-MAX_LOG_BYTES, os.SEEK_END)
            data = handle.read(MAX_LOG_BYTES).replace(token, b"[REDACTED]")
        (logs_root / f"internal-{source.name}").write_bytes(data)


def _validate_voice_response(response: HttpResult, *, request_id: str, output: Path) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    if response.status != 200 or not response.headers.get("content-type", "").casefold().startswith("audio/wav"):
        raise RuntimeError("Voice synthesis did not return WAV audio")
    digest = hashlib.sha256(response.body).hexdigest()
    expected = response.headers.get("x-youtuber-sha256")
    if expected != digest or response.headers.get("x-youtuber-request-id") != request_id:
        raise RuntimeError("Voice response integrity metadata disagrees with the body")
    if response.headers.get("x-youtuber-rvc-epoch") != "200" or response.headers.get("x-youtuber-index-rate") != "0.75":
        raise RuntimeError("Voice response did not use the selected epoch/index rate")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(response.body)
    info = sf.info(output)
    samples, sample_rate = sf.read(output, dtype="float32", always_2d=True)
    waveform = np.asarray(samples, dtype=np.float32)
    if (
        info.format != "WAV"
        or info.channels != 1
        or info.frames <= 0
        or waveform.shape != (info.frames, 1)
        or not np.isfinite(waveform).all()
    ):
        raise RuntimeError("Voice response is not finite nonempty mono WAV audio")
    clipped = float(np.count_nonzero(np.abs(waveform) >= 1.0)) / waveform.size
    if clipped >= 0.001:
        raise RuntimeError("Voice response exceeds the clipping guard")
    header_rate = int(response.headers["x-youtuber-sample-rate"])
    header_duration = float(response.headers["x-youtuber-duration-seconds"])
    duration = info.frames / info.samplerate
    if header_rate != sample_rate or abs(header_duration - duration) > 0.001:
        raise RuntimeError("Voice response audio metadata is inconsistent")
    return {
        "path": str(output.resolve()),
        "sha256": digest,
        "sample_rate": sample_rate,
        "duration_seconds": duration,
        "clipped_fraction": clipped,
        "channels": info.channels,
        "rvc_epoch": 200,
        "index_rate": 0.75,
        "request_id": request_id,
    }


def _finalize_real_run(
    *,
    report: dict[str, Any],
    children: list[Child],
    launched_tree_pids: set[int],
    staged: dict[str, Any],
    secret: str,
    initial_ollama: list[str],
    required_ollama: set[str],
    ports: list[int],
    keep_output: bool,
    primary_error: BaseException | None,
) -> BaseException | None:
    """Finish every independent cleanup step and persist sanitized diagnostics."""
    cleanup_errors: list[dict[str, str]] = []

    def failed(step: str, exc: BaseException) -> None:
        cleanup_errors.append({"step": step, "type": type(exc).__name__})

    for child in reversed(children):
        try:
            launched_tree_pids.update(_process_tree_pids(child.process.pid))
        except BaseException as exc:
            failed(f"process_snapshot:{child.name}", exc)

    exit_codes: dict[str, int | None] = {}
    for child in reversed(children):
        try:
            exit_codes[child.name] = _terminate_tree(child)
        except BaseException as exc:
            failed(f"process_terminate:{child.name}", exc)
            try:
                exit_codes[child.name] = child.process.poll()
            except BaseException as poll_exc:
                failed(f"process_poll:{child.name}", poll_exc)
                exit_codes[child.name] = None

    try:
        alive = _wait_pids_gone(launched_tree_pids)
    except BaseException as exc:
        failed("process_reap_probe", exc)
        alive = sorted(launched_tree_pids)

    try:
        _preserve_internal_logs(staged["cache_root"], staged["logs"], secret)
    except BaseException as exc:
        failed("preserve_internal_logs", exc)

    try:
        restoration = _restore_ollama_residency(
            initial=initial_ollama,
            required=required_ollama,
        )
    except BaseException as exc:
        failed("ollama_restore", exc)
        try:
            after_restore = _ollama_loaded_models()
        except BaseException as snapshot_exc:
            failed("ollama_snapshot", snapshot_exc)
            after_restore = []
        restoration = {
            "before": sorted(initial_ollama),
            "after": after_restore,
            "required_restored": False,
            "unrelated_unchanged": False,
            "restored": False,
            "error": type(exc).__name__,
        }
    report.setdefault("ollama", {})["restoration"] = restoration
    report["ollama"]["after_cleanup"] = restoration.get("after", [])

    try:
        verified_hardlinks = _verify_hardlinked_source_integrity(staged["selection"])
        source_integrity_verified = True
    except BaseException as exc:
        failed("source_integrity", exc)
        verified_hardlinks = 0
        source_integrity_verified = False
        report.setdefault("assets", {})["source_integrity_error"] = type(exc).__name__
    report.setdefault("assets", {})["verified_read_only_hardlinks"] = verified_hardlinks

    try:
        ports_released = _ports_released(ports)
    except BaseException as exc:
        failed("port_release_probe", exc)
        ports_released = False

    try:
        job_objects_closed = all(
            child.job is None or getattr(child.job, "handle", None) is None
            for child in children
        )
    except BaseException as exc:
        failed("job_close_probe", exc)
        job_objects_closed = False

    staging_removed = False
    if not keep_output:
        for name in ("install_root", "data_root", "cache_root"):
            directory = staged[name]
            try:
                if directory.exists():
                    shutil.rmtree(directory)
            except BaseException as exc:
                failed(f"staging_remove:{name}", exc)
        try:
            staging_removed = all(
                not staged[name].exists()
                for name in ("install_root", "data_root", "cache_root")
            )
        except BaseException as exc:
            failed("staging_remove_probe", exc)
            staging_removed = False
    report["staging_removed"] = staging_removed

    report["cleanup"] = {
        "exit_codes": exit_codes,
        "launched_tree_pids": sorted(launched_tree_pids),
        "launched_pids_alive": alive,
        "ports_released": ports_released,
        "job_objects_closed": job_objects_closed,
        "ollama_restored": bool(restoration.get("restored")),
        "source_integrity_verified": source_integrity_verified,
        "errors": cleanup_errors,
    }

    cleanup_failed = bool(
        cleanup_errors
        or alive
        or not ports_released
        or not job_objects_closed
        or not restoration.get("restored")
        or not source_integrity_verified
        or (not keep_output and not staging_removed)
    )
    final_error = primary_error
    if final_error is None and cleanup_failed:
        final_error = RuntimeError("packaged cleanup failed")

    if final_error is not None:
        processes: dict[str, dict[str, Any]] = {}
        for child in children:
            try:
                exit_code = child.process.poll()
            except BaseException as exc:
                failed(f"failure_poll:{child.name}", exc)
                exit_code = None
            try:
                tail = child.log.tail()
            except BaseException as exc:
                failed(f"failure_tail:{child.name}", exc)
                tail = []
            processes[child.name] = {"exit_code": exit_code, "tail": tail}
        report["failure"] = {
            "type": type(final_error).__name__,
            "processes": processes,
        }
        report["cleanup"]["errors"] = cleanup_errors

    _write_json(Path(staged["output_root"]) / "result.json", report)
    return final_error


def run_real(
    source_root: Path,
    output_root: Path,
    *,
    question: str,
    voice_enabled: bool,
    keep_output: bool,
) -> dict[str, Any]:
    if question != "Mutlak butlan hakkında ne düşünüyorsun?":
        raise ValueError("real gate requires the approved exact Turkish question")
    for executable in (STUDIO_EXE, VOICE_EXE):
        if not executable.is_file():
            raise FileNotFoundError(executable)
    ollama = _validate_ollama_models()
    initial_ollama = _ollama_loaded_models()
    required_ollama = {ollama["answer_model"], ollama["verifier_model"]}
    staged = stage_real_assets(source_root, output_root)
    output_root = Path(staged["output_root"])
    ports = _free_ports()
    secret = secrets.token_urlsafe(48)
    studio_env = _base_environment(
        staged["install_root"], staged["data_root"], staged["cache_root"], ports, secret
    )
    studio_env.update(
        {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    )
    voice_env = _base_environment(
        VOICE_EXE.parent.resolve(), staged["data_root"], staged["cache_root"], ports, secret
    )
    children: list[Child] = []
    launched_tree_pids: set[int] = set()
    report: dict[str, Any] = {
        "schema": "youtuber.packaged-smoke.v1",
        "mode": "real",
        "question": question,
        "ports": {"studio": ports[0], "rag": ports[1], "voice": ports[2]},
        "assets": {
            "selected_file_count": len(staged["selection"]["files"]),
            "rag_identity": staged["rag_identity"],
            "selection_manifest": str((output_root / "asset-selection.json").resolve()),
        },
        "ollama": {"identities": ollama, "before": initial_ollama},
    }
    error: BaseException | None = None
    try:
        studio = _start("studio", [str(STUDIO_EXE)], cwd=STUDIO_EXE.parent, env=studio_env, logs=staged["logs"], secret=secret)
        children.append(studio)
        ready = _await_http(
            studio,
            f"http://127.0.0.1:{ports[0]}/health/ready",
            expected={200},
            timeout=STUDIO_BOOTSTRAP_TIMEOUT_SECONDS,
        )
        ready_payload = ready.json()
        if (
            ready_payload.get("status") != "ready"
            or ready_payload.get("runtime_api") != 1
            or ready_payload.get("answer_model") != ollama["answer_model"]
            or ready_payload.get("verifier_model") != ollama["verifier_model"]
            or ready_payload.get("answer_token_cap", {}).get("maximum") != 4096
        ):
            raise RuntimeError(f"Studio readiness identity is incompatible: {ready_payload}")
        unauth_root = _http("GET", f"http://127.0.0.1:{ports[0]}/")
        unauth_bootstrap = _http("POST", f"http://127.0.0.1:{ports[0]}/bootstrap")
        bootstrap = _http(
            "POST", f"http://127.0.0.1:{ports[0]}/bootstrap", bearer=secret
        )
        cookie = _cookie_from_bootstrap(bootstrap, secret)
        authenticated_root = _http(
            "GET",
            f"http://127.0.0.1:{ports[0]}/",
            extra_headers={"Cookie": cookie},
            max_bytes=8 * 1024 * 1024,
        )
        if (unauth_root.status, unauth_bootstrap.status, authenticated_root.status) != (403, 403, 200):
            raise RuntimeError("Studio authentication boundary failed")
        rag_ready = _http(
            "GET", f"http://127.0.0.1:{ports[1]}/health/ready", bearer=secret
        )
        rag_identity = rag_ready.json()
        expected_identity = staged["rag_identity"]
        if (
            rag_ready.status != 200
            or rag_identity.get("status") != "ready"
            or rag_identity.get("clean_count") != expected_identity["clean_count"]
            or rag_identity.get("card_count") != expected_identity["card_count"]
            or rag_identity.get("corpus_fingerprint") != expected_identity["corpus_fingerprint"]
        ):
            raise RuntimeError("frozen RAG readiness identity differs from staged manifests")
        query = urllib.parse.urlencode({"q": question, "mode": "clean_with_card_hints"})
        retrieval = _http(
            "GET",
            f"http://127.0.0.1:{ports[1]}/retrieve?{query}",
            bearer=secret,
            timeout=660.0,
            max_bytes=8 * 1024 * 1024,
        )
        decision = retrieval.json()
        if retrieval.status != 200 or decision.get("status") not in {"answerable", "partial"} or not decision.get("evidence_hits"):
            raise RuntimeError(f"approved question was not grounded: {decision.get('status')}")
        report["ollama"]["after_retrieval"] = _ollama_loaded_models()

        chat = _http(
            "POST",
            f"http://127.0.0.1:{ports[0]}/v1/chat",
            extra_headers={"Cookie": cookie},
            payload={
                "question": question,
                "answer_mode": "grounded_strict",
                "retrieval_mode": "clean_with_card_hints",
                "answer_token_cap": 640,
            },
            timeout=660.0,
            max_bytes=8 * 1024 * 1024,
        )
        chat_payload = chat.json()
        if (
            chat.status != 200
            or chat_payload.get("runtime_api") != 1
            or chat_payload.get("answer_mode_contract_version") != 1
            or chat_payload.get("answer_model") != ollama["answer_model"]
            or chat_payload.get("answer_mode") != "grounded_strict"
            or chat_payload.get("retrieval_mode") != "clean_with_card_hints"
            or chat_payload.get("answer_token_cap") != 640
        ):
            raise RuntimeError("frozen Studio chat response contract is incompatible")
        grounding_status = chat_payload.get("grounding_status")
        if (
            grounding_status not in {"answerable", "partial"}
            or type(chat_payload.get("evidence_count")) is not int
            or chat_payload.get("evidence_count", 0) <= 0
        ):
            raise RuntimeError(
                f"frozen Studio generation was not grounded: {grounding_status}"
            )
        answer = _validated_chat_answer(chat_payload)
        answer_path = output_root / "results" / "answer.json"
        _write_json(
            answer_path,
            {
                "question": question,
                "answer": answer,
                "answer_mode": "grounded_strict",
                "answer_model": ollama["answer_model"],
                "generation_status": chat_payload["generation_status"],
                "retrieval_status": grounding_status,
                "evidence_count": chat_payload["evidence_count"],
                "context_markdown": chat_payload["context_markdown"],
                "metadata": chat_payload["metadata"],
            },
        )
        report["studio"] = {
            "executable": str(STUDIO_EXE.resolve()),
            "pid": studio.process.pid,
            "ready": ready_payload,
            "bootstrap_status": bootstrap.status,
            "cookie_http_only": True,
            "cookie_same_site": "strict",
            "authenticated_root_status": authenticated_root.status,
            "chat_status": chat.status,
        }
        report["rag"] = {
            "ready": rag_identity,
            "preflight_status": decision["status"],
            "preflight_evidence_count": len(decision["evidence_hits"]),
            "generation_grounding_status": grounding_status,
            "retrieval_status": grounding_status,
            "evidence_count": chat_payload["evidence_count"],
            "preflight_claim_count": len(decision.get("claims", [])),
        }
        report["answer"] = {
            "artifact": str(answer_path.resolve()),
            "sha256": _sha256_file(answer_path),
            "characters": len(answer),
            "model": ollama["answer_model"],
            "mode": "grounded_strict",
            "generation_status": chat_payload["generation_status"],
        }
        report["ollama"]["after_answer"] = _ollama_loaded_models()

        if voice_enabled:
            _ensure_ollama_model_absent(ollama["answer_model"])
            report["ollama"]["before_voice"] = _ollama_loaded_models()
            if ollama["answer_model"] in report["ollama"]["before_voice"]:
                raise RuntimeError("answer model remained resident before Voice startup")
            voice = _start("voice", [str(VOICE_EXE)], cwd=VOICE_EXE.parent, env=voice_env, logs=staged["logs"], secret=secret)
            children.append(voice)
            voice_live = _await_http(voice, f"http://127.0.0.1:{ports[2]}/health/live", expected={200})
            voice_ready = _http("GET", f"http://127.0.0.1:{ports[2]}/health/ready", timeout=120.0)
            voice_unauth = _http("GET", f"http://127.0.0.1:{ports[2]}/v1/status")
            voice_status = _http("GET", f"http://127.0.0.1:{ports[2]}/v1/status", bearer=secret)
            if voice_live.status != 200 or voice_ready.status != 200 or voice_unauth.status != 401 or voice_status.status != 200:
                raise RuntimeError("Voice readiness/authentication boundary failed")
            request_id = "real-" + secrets.token_hex(8)
            voice_response = _http(
                "POST",
                f"http://127.0.0.1:{ports[2]}/v1/synthesize",
                bearer=secret,
                payload={"text": answer, "request_id": request_id},
                timeout=660.0,
                max_bytes=128 * 1024 * 1024,
            )
            voice_metrics = _validate_voice_response(
                voice_response,
                request_id=request_id,
                output=output_root / "results" / "answer-voice.wav",
            )
            unload = _http("POST", f"http://127.0.0.1:{ports[2]}/v1/unload", bearer=secret, timeout=120.0)
            if unload.status != 200 or unload.json().get("status") != "idle":
                raise RuntimeError("Voice unload failed")
            report["voice"] = {
                "executable": str(VOICE_EXE.resolve()),
                "pid": voice.process.pid,
                "ready": voice_ready.json(),
                "unauthenticated_status": voice_unauth.status,
                "initial_status": voice_status.json(),
                "unload_status": unload.json(),
                **voice_metrics,
            }
        for child in children:
            launched_tree_pids.update(_process_tree_pids(child.process.pid))
    except BaseException as caught:
        error = caught
    finally:
        error = _finalize_real_run(
            report=report,
            children=children,
            launched_tree_pids=launched_tree_pids,
            staged=staged,
            secret=secret,
            initial_ollama=initial_ollama,
            required_ollama=required_ollama,
            ports=ports,
            keep_output=keep_output,
            primary_error=error,
        )
    if error is not None:
        raise RuntimeError(
            f"real packaged gate failed ({type(error).__name__}); see {output_root / 'result.json'}"
        ) from error
    if report["cleanup"]["launched_pids_alive"] or not report["cleanup"]["ports_released"]:
        raise RuntimeError("real packaged gate cleanup failed")
    return report


def _write_fixture_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "paths:\n"
        "  data_dir: data\n  audio_dir: data/audio\n  meta_dir: data/meta\n"
        "  vad_dir: data/vad\n  diarize_dir: data/diarize\n  identify_dir: data/identify\n"
        "  transcribe_dir: data/transcribe\n  filter_dir: data/filter\n  clean_dir: data/clean\n"
        "  dataset_dir: data/dataset\n  reference_clip: data/reference/speaker_reference.wav\n"
        "  review_dir: data/review\n  decisions_db: data/decisions.sqlite\n  logs_dir: logs\n"
        "rag:\n  embedder: BAAI/bge-m3\n  reranker: BAAI/bge-reranker-v2-m3\n"
        "  store_path: data/rag/qdrant\n  collection: speaker_clean\n"
        "  evidence_collection: speaker_clean\n  card_collection: speaker_cards\n"
        "  card_catalog_path: data/cards/speaker_cards.json\n"
        "  card_manifest_path: data/cards/speaker_cards.index_manifest.json\n"
        "  verifier_model: qwen3:4b\n",
        encoding="utf-8",
    )


def run_fixture(output_root: Path) -> dict[str, Any]:
    for executable in (STUDIO_EXE, VOICE_EXE):
        if not executable.is_file():
            raise FileNotFoundError(executable)
    output_root = _prepare_output_root(output_root)
    install_root = output_root / "install"
    data_root = output_root / "data"
    cache_root = output_root / "cache"
    logs = output_root / "logs"
    for directory in (data_root, cache_root, logs):
        directory.mkdir(parents=True, exist_ok=True)
    config = install_root / "runtime" / "config.yaml"
    _write_fixture_config(config)
    ports = _free_ports()
    secret = secrets.token_urlsafe(48)
    environment = _base_environment(install_root, data_root, cache_root, ports, secret)
    children: list[Child] = []
    launched_tree_pids: set[int] = set()
    report: dict[str, Any] = {
        "schema": "youtuber.packaged-smoke.v1",
        "mode": "fixture",
        "ports": {"studio": ports[0], "rag": ports[1], "voice": ports[2]},
    }
    try:
        voice = _start("voice", [str(VOICE_EXE)], cwd=VOICE_EXE.parent, env=environment, logs=logs, secret=secret)
        children.append(voice)
        voice_live = _await_http(voice, f"http://127.0.0.1:{ports[2]}/health/live", expected={200})
        voice_unauth = _http("GET", f"http://127.0.0.1:{ports[2]}/v1/status")
        if voice_live.json() != {"status": "live"} or voice_unauth.status != 401:
            raise RuntimeError("Voice fixture boundary contract failed")
        report["voice"] = {
            "executable": str(VOICE_EXE.resolve()),
            "pid": voice.process.pid,
            "live_status": voice_live.status,
            "unauthenticated_status_status": voice_unauth.status,
        }

        studio = _start(
            "studio-rag",
            [str(STUDIO_EXE), "--rag-worker", "--port", str(ports[1]), "--config", str(config), "--host", "127.0.0.1"],
            cwd=STUDIO_EXE.parent,
            env=environment,
            logs=logs,
            secret=secret,
        )
        children.append(studio)
        studio_live = _await_http(studio, f"http://127.0.0.1:{ports[1]}/health/live", expected={200})
        studio_unauth = _http("GET", f"http://127.0.0.1:{ports[1]}/health/ready")
        if studio_live.json().get("runtime_api") != 1 or studio_unauth.status != 403:
            raise RuntimeError("Studio frozen RAG fixture boundary contract failed")
        report["studio"] = {
            "executable": str(STUDIO_EXE.resolve()),
            "pid": studio.process.pid,
            "frozen_rag_dispatch": True,
            "live_status": studio_live.status,
            "unauthenticated_ready_status": studio_unauth.status,
        }
    except BaseException as error:
        diagnostics = {
            child.name: {"exit_code": child.process.poll(), "tail": child.log.tail()}
            for child in children
        }
        raise RuntimeError(json.dumps({"error": type(error).__name__, "processes": diagnostics}, ensure_ascii=False)) from error
    finally:
        for child in children:
            launched_tree_pids.update(_process_tree_pids(child.process.pid))
        exit_codes = {child.name: _terminate_tree(child) for child in reversed(children)}
        launched_tree_pids.update(child.process.pid for child in children)
        alive = _wait_pids_gone(launched_tree_pids)
        report["cleanup"] = {
            "exit_codes": exit_codes,
            "launched_tree_pids": sorted(launched_tree_pids),
            "launched_pids_alive": alive,
            "ports_released": _ports_released(ports),
            "job_objects_closed": all(
                child.job is None or child.job.handle is None for child in children
            ),
        }
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if report["cleanup"]["launched_pids_alive"] or not report["cleanup"]["ports_released"]:
        raise RuntimeError("packaged fixture cleanup failed")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--real-assets-root", type=Path)
    parser.add_argument("--question", default="Mutlak butlan hakkında ne düşünüyorsun?")
    parser.add_argument("--voice", action="store_true")
    parser.add_argument("--output-root", type=Path, default=ROOT / "build" / "packaged-smoke")
    parser.add_argument("--keep-output", action="store_true")
    args = parser.parse_args(argv)
    if args.real_assets_root is not None:
        if args.fixture:
            parser.error("--fixture and --real-assets-root are mutually exclusive")
        report = run_real(
            args.real_assets_root,
            args.output_root,
            question=args.question,
            voice_enabled=args.voice,
            keep_output=args.keep_output,
        )
    elif args.fixture:
        report = run_fixture(args.output_root)
    else:
        parser.error("select --fixture or --real-assets-root")
    print(json.dumps({"status": "pass", "mode": report["mode"], "result": str((args.output_root / 'result.json').resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
