"""Immutable contracts for the packaged voice artifact and XTTS engine."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

if TYPE_CHECKING:
    from runtime.paths import RuntimePaths


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or ":" in value
        or ".." in path.parts
    ):
        raise ValueError("artifact paths must be safe relative POSIX paths")
    return path.as_posix()


class ArtifactFile(BaseModel):
    """One immutable, hash-pinned voice artifact file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    size: StrictInt = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_path(self) -> "ArtifactFile":
        _safe_relative_path(self.path)
        return self


class XttsArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint: str
    config: str
    vocab: str
    reference_wavs: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_paths(self) -> "XttsArtifact":
        for value in (self.checkpoint, self.config, self.vocab, *self.reference_wavs):
            _safe_relative_path(value)
        if (self.checkpoint, self.config, self.vocab) != (
            "xtts/best_model.pth",
            "xtts/config.json",
            "xtts/vocab.json",
        ):
            raise ValueError("XTTS artifact must select best_model.pth, config.json, and vocab.json")
        if not self.reference_wavs:
            raise ValueError("XTTS artifact requires at least one approved reference WAV")
        if any(Path(path).suffix.lower() != ".wav" for path in self.reference_wavs):
            raise ValueError("XTTS references must be WAV files")
        return self


class RvcArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    weight: str
    index: str
    epoch: StrictInt
    index_rate: Literal[0.75]
    pitch: StrictInt
    f0_method: Literal["rmvpe"]
    protect: Literal[0.5]

    @model_validator(mode="after")
    def _validate_paths(self) -> "RvcArtifact":
        _safe_relative_path(self.weight)
        _safe_relative_path(self.index)
        if self.epoch != 200:
            raise ValueError("RVC artifact must select epoch 200")
        if self.pitch != 0:
            raise ValueError("RVC artifact must use pitch 0")
        return self


class ContentVecArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config: str
    model: str

    @model_validator(mode="after")
    def _validate_paths(self) -> "ContentVecArtifact":
        _safe_relative_path(self.config)
        _safe_relative_path(self.model)
        return self


class VoiceArtifactManifest(BaseModel):
    """Strict v1 voice-artifact selection and identity contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, serialize_by_alias=True)

    artifact_schema: Literal["youtuber.voice.v1"] = Field(alias="schema")
    runtime_api: StrictInt
    sample_rate: StrictInt = Field(gt=0)
    audio_format: Literal["WAV"]
    files: tuple[ArtifactFile, ...]
    xtts: XttsArtifact
    rvc: RvcArtifact
    contentvec: ContentVecArtifact
    rmvpe: str

    @property
    def schema(self) -> Literal["youtuber.voice.v1"]:
        """Compatibility accessor for the serialized manifest field."""
        return self.artifact_schema

    @model_validator(mode="after")
    def _validate_references(self) -> "VoiceArtifactManifest":
        if self.runtime_api != 1:
            raise ValueError("voice manifest must use runtime API 1")
        _safe_relative_path(self.rmvpe)
        paths = {entry.path.casefold() for entry in self.files}
        if len(paths) != len(self.files):
            raise ValueError("voice manifest contains duplicate file paths")
        required = {
            self.xtts.checkpoint,
            self.xtts.config,
            self.xtts.vocab,
            *self.xtts.reference_wavs,
            self.rvc.weight,
            self.rvc.index,
            self.contentvec.config,
            self.contentvec.model,
            self.rmvpe,
        }
        missing = sorted(path for path in required if path.casefold() not in paths)
        if missing:
            raise ValueError("voice manifest path is not an approved file: " + ", ".join(missing))
        return self

    def file_entry(self, relative_path: str) -> ArtifactFile:
        safe = _safe_relative_path(relative_path)
        for entry in self.files:
            if entry.path == safe:
                return entry
        raise ValueError(f"voice manifest path is not an approved file: {safe}")

    def validate_file(self, voice_root: Path, relative_path: str) -> Path:
        entry = self.file_entry(relative_path)
        root = Path(voice_root).resolve(strict=False)
        candidate = (root / PurePosixPath(entry.path)).resolve(strict=False)
        if not candidate.is_relative_to(root):
            raise ValueError(f"voice artifact path escapes voice root: {entry.path}")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if candidate.stat().st_size != entry.size:
            raise ValueError(f"voice artifact size mismatch: {entry.path}")
        digest = _sha256_file(candidate)
        if digest != entry.sha256:
            raise ValueError(f"voice artifact hash mismatch: {entry.path}")
        return candidate


class XttsSettings(BaseModel):
    """Validated XTTS selection; paths remain immutable after construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_dir: Path
    reference_wavs: tuple[Path, ...]
    language: str = "tr"
    sample_rate: StrictInt = Field(default=24_000, gt=0)
    audio_format: Literal["WAV"] = "WAV"
    clipping_guard: float = Field(default=0.001, ge=0.0, le=1.0)
    checkpoint_path: Path | None = None
    config_path: Path | None = None
    vocab_path: Path | None = None

    @model_validator(mode="after")
    def _validate_references(self) -> "XttsSettings":
        if not self.reference_wavs:
            raise ValueError("at least one approved reference WAV is required")
        for reference in self.reference_wavs:
            if reference.suffix.lower() != ".wav":
                raise ValueError("reference paths must be WAV files")
        return self

    @property
    def checkpoint_file(self) -> Path:
        return self.checkpoint_path or self.checkpoint_dir / "best_model.pth"

    @property
    def config_file(self) -> Path:
        return self.config_path or self.checkpoint_dir / "config.json"

    @property
    def vocab_file(self) -> Path:
        return self.vocab_path or self.checkpoint_dir / "vocab.json"

    @classmethod
    def from_runtime_paths(
        cls, paths: RuntimePaths, manifest: VoiceArtifactManifest
    ) -> "XttsSettings":
        root = paths.voice_root.resolve(strict=False)
        checkpoint_paths = (
            manifest.validate_file(root, manifest.xtts.checkpoint),
            manifest.validate_file(root, manifest.xtts.config),
            manifest.validate_file(root, manifest.xtts.vocab),
        )
        checkpoint_dir = checkpoint_paths[0].parent
        if any(path.parent != checkpoint_dir for path in checkpoint_paths):
            raise ValueError("XTTS checkpoint files must share one checkpoint directory")
        references = tuple(
            manifest.validate_file(root, relative) for relative in manifest.xtts.reference_wavs
        )
        return cls(
            checkpoint_dir=checkpoint_dir,
            reference_wavs=references,
            sample_rate=manifest.sample_rate,
            audio_format=manifest.audio_format,
        )


class AudioArtifact(BaseModel):
    """A validated generated WAV, suitable to hand to the next pipeline stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path
    sample_rate: StrictInt = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
