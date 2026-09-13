"""Transactional XTTS-to-RVC orchestration with a process-wide synthesis gate."""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from voice.api_models import VoiceStatus


# XTTS/RVC GPU residency is process-global even if a caller constructs two wrappers.
_SYNTHESIS_GATE = threading.Lock()


class PipelineError(RuntimeError):
    """A controlled voice pipeline failure safe to translate at an HTTP boundary."""


class PipelineBusyError(PipelineError):
    """Another synthesis owns the process-wide mutable model stack."""


class XttsStage(Protocol):
    def synthesize(self, text: str, output: Path) -> object: ...

    def unload(self) -> None: ...


class RvcStage(Protocol):
    def convert(self, source: Path, output: Path) -> object: ...

    def unload(self) -> None: ...


class VoiceResult(BaseModel):
    """Internal immutable metadata for one atomically promoted converted WAV."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    final_path: Path
    sample_rate: StrictInt = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rvc_epoch: Literal[200] = 200
    index_rate: Literal[0.75] = 0.75


class VoicePipeline:
    """Owns one XTTS/RVC pair; low-VRAM operation serializes their residency."""

    def __init__(
        self,
        xtts: XttsStage,
        rvc: RvcStage,
        cache_root: Path,
        *,
        low_vram_voice: bool = False,
        keep_intermediate_audio: bool = False,
    ) -> None:
        self._xtts = xtts
        self._rvc = rvc
        self._cache_root = Path(cache_root).absolute()
        self._low_vram_voice = low_vram_voice
        self._keep_intermediate_audio = keep_intermediate_audio
        self._status_lock = threading.RLock()
        self._status: VoiceStatus = "idle"

    @property
    def status(self) -> VoiceStatus:
        with self._status_lock:
            return self._status

    def _set_status(self, status: VoiceStatus) -> None:
        with self._status_lock:
            self._status = status

    def synthesize(self, text: str, request_id: str) -> VoiceResult:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must not be blank")
        if not _SYNTHESIS_GATE.acquire(blocking=False):
            raise PipelineBusyError("voice synthesis is busy")
        try:
            return self._synthesize_owned(text, request_id)
        except BaseException:
            self._set_status("error")
            raise
        finally:
            _SYNTHESIS_GATE.release()

    def _synthesize_owned(self, text: str, request_id: str) -> VoiceResult:
        job_dir = self._job_directory(request_id)
        raw = job_dir / "raw_xtts.wav"
        converted = job_dir / "converted.wav"
        raw_partial = job_dir / "raw_xtts.wav.partial"
        converted_partial = job_dir / "converted.wav.partial"
        raw_backup = job_dir / "raw_xtts.wav.backup"
        for path in (raw, converted, raw_partial, converted_partial, raw_backup):
            self._reject_link_or_collision(path)
        self._remove_regular_file(raw_partial)
        self._remove_regular_file(converted_partial)
        self._remove_regular_file(raw_backup)
        try:
            self._set_status("loading_xtts")
            self._set_status("synthesizing")
            self._xtts.synthesize(text, raw_partial)
            raw_rate, raw_duration = self._validate_wav(raw_partial, "XTTS output")
            if raw.exists():
                raw.replace(raw_backup)
            self._promote(raw_partial, raw)
            if self._low_vram_voice:
                self._xtts.unload()
            self._set_status("loading_rvc")
            self._set_status("converting")
            self._rvc.convert(raw, converted_partial)
            sample_rate, duration = self._validate_wav(converted_partial, "RVC output")
            if abs(duration - raw_duration) / raw_duration > 0.10:
                raise PipelineError("converted audio duration is outside the permitted range")
            self._promote(converted_partial, converted)
            if not self._keep_intermediate_audio:
                self._remove_regular_file(raw)
            self._remove_regular_file(raw_backup)
            if self._low_vram_voice:
                self._rvc.unload()
            self._set_status("idle")
            return VoiceResult(
                final_path=converted,
                sample_rate=sample_rate,
                duration_seconds=duration,
                sha256=self._sha256(converted),
            )
        except BaseException:
            self._remove_regular_file(raw_partial)
            self._remove_regular_file(converted_partial)
            if raw_backup.exists():
                self._remove_regular_file(raw)
                raw_backup.replace(raw)
            elif not self._keep_intermediate_audio:
                self._remove_regular_file(raw)
            self._remove_regular_file(raw_backup)
            if self._low_vram_voice:
                self._unload_safely(self._xtts)
                self._unload_safely(self._rvc)
            raise

    def unload(self) -> None:
        """Release both engines; calling repeatedly is deliberately harmless."""
        with _SYNTHESIS_GATE:
            errors: list[BaseException] = []
            for engine in (self._xtts, self._rvc):
                try:
                    engine.unload()
                except BaseException as error:
                    errors.append(error)
            if errors:
                self._set_status("error")
                raise PipelineError("voice engine unload failed") from errors[0]
            if self.status != "error":
                self._set_status("idle")

    def _job_directory(self, request_id: str) -> Path:
        # API validation enforces the format too; this defense keeps direct callers safe.
        if not request_id or len(request_id) > 64 or not all(char.isascii() and (char.isalnum() or char in "_-") for char in request_id):
            raise PipelineError("invalid request ID")
        self._reject_link_or_collision(self._cache_root)
        audio_root = self._cache_root / "audio"
        self._reject_link_or_collision(audio_root)
        if audio_root.exists() and not audio_root.is_dir():
            raise PipelineError("audio cache collision")
        audio_root.mkdir(parents=True, exist_ok=True)
        job_dir = audio_root / request_id
        self._reject_link_or_collision(job_dir)
        if job_dir.exists() and not job_dir.is_dir():
            raise PipelineError("request cache collision")
        job_dir.mkdir(exist_ok=True)
        return job_dir

    @staticmethod
    def _is_reparse(path: Path) -> bool:
        try:
            attributes = path.lstat().st_file_attributes
        except (AttributeError, FileNotFoundError, OSError):
            return False
        return bool(attributes & 0x400)  # FILE_ATTRIBUTE_REPARSE_POINT on Windows

    def _reject_link_or_collision(self, path: Path) -> None:
        if path.is_symlink() or self._is_reparse(path):
            raise PipelineError("unsafe cache path")
        if path.exists() and path.is_dir() and path.name.endswith(".wav"):
            raise PipelineError("audio output collision")

    @staticmethod
    def _remove_regular_file(path: Path) -> None:
        if path.exists():
            if path.is_symlink() or path.is_dir():
                raise PipelineError("unsafe cache path")
            path.unlink()

    def _promote(self, partial: Path, final: Path) -> None:
        self._reject_link_or_collision(partial)
        self._reject_link_or_collision(final)
        if not partial.is_file():
            raise PipelineError("audio stage produced no output")
        partial.replace(final)

    @staticmethod
    def _validate_wav(path: Path, label: str) -> tuple[int, float]:
        try:
            info = sf.info(path)
            samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        except RuntimeError as error:
            raise PipelineError(f"{label} is not a readable WAV") from error
        waveform = np.asarray(samples, dtype=np.float32)
        if info.format != "WAV" or info.channels != 1 or info.frames <= 0 or sample_rate <= 0:
            raise PipelineError(f"{label} must be a non-empty mono WAV")
        if waveform.shape != (info.frames, 1) or not np.isfinite(waveform).all():
            raise PipelineError(f"{label} must contain finite mono audio")
        clipped = float(np.count_nonzero(np.abs(waveform) >= 1.0)) / waveform.size
        if clipped >= 0.001:
            raise PipelineError(f"{label} exceeds clipping guard")
        return sample_rate, info.frames / sample_rate

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _unload_safely(engine: object) -> None:
        try:
            unload = getattr(engine, "unload")
            unload()
        except BaseException:
            pass
