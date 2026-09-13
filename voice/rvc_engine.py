"""Manifest-selected, epoch-200 RVC conversion with a lazy Applio boundary."""
from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import sys
import tempfile
import threading
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from voice.models import ArtifactFile, AudioArtifact, VoiceArtifactManifest

if TYPE_CHECKING:
    from runtime.paths import RuntimePaths


EPOCH_200_WEIGHT_SHA256 = "a68b9190622f10eccab3432a1e18ed8a20dd1361509bc1fbc75a6c64dcda1a61"
EPOCH_200_INDEX_SHA256 = "ef9e0c8628c6457bbe26cb3df0a02c11b79306dd6dbee26978558ad10c424141"
_BACKEND_LOCK = threading.RLock()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _isolated_applio_state(vendor_root: Path, working_directory: Path):
    """Temporarily isolate upstream import mutations and restore process state exactly."""
    original_cwd = Path.cwd()
    original_sys_path = list(sys.path)
    original_dont_write_bytecode = sys.dont_write_bytecode
    os.chdir(working_directory)
    sys.path.insert(0, str(vendor_root))
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = original_dont_write_bytecode
        sys.path[:] = original_sys_path
        os.chdir(original_cwd)


def probe_applio_runtime(vendor_root: Path) -> None:
    """Verify the pinned inference API without constructing or loading a voice model."""
    root = Path(vendor_root).resolve(strict=False)
    infer_source = root / "rvc" / "infer" / "infer.py"
    if not infer_source.is_file():
        raise RuntimeError("pinned Applio inference runtime is missing")
    with _BACKEND_LOCK:
        with _isolated_applio_state(root, root):
            module = importlib.import_module("rvc.infer.infer")
            module_file = Path(str(getattr(module, "__file__", ""))).resolve(strict=False)
            if not module_file.is_relative_to(root):
                raise RuntimeError("pinned Applio inference runtime resolved outside packaged vendor")
            if not callable(getattr(module, "VoiceConverter", None)):
                raise RuntimeError("pinned Applio inference API is unavailable")


class VoiceConverterBackend(Protocol):
    def convert_audio(self, **kwargs: object) -> object: ...

    def unload(self) -> None: ...


class RvcSettings(BaseModel):
    """Immutable RVC selection. Production instances originate from the voice manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    weight: Path
    index: Path
    contentvec_root: Path
    rmvpe_path: Path
    index_rate: Literal[0.75] = 0.75
    pitch: StrictInt = 0
    protect: Literal[0.5] = 0.5
    f0_method: Literal["rmvpe"] = "rmvpe"
    audio_format: Literal["WAV"] = "WAV"
    epoch: StrictInt = 200
    approved_files: tuple[ArtifactFile, ...] = Field(default=(), repr=False)

    @model_validator(mode="after")
    def _validate_selection(self) -> "RvcSettings":
        if self.epoch != 200:
            raise ValueError("RVC settings must select epoch 200")
        if self.pitch != 0:
            raise ValueError("RVC settings must use pitch 0")
        expected_paths = {
            self.weight.resolve(strict=False),
            self.index.resolve(strict=False),
            (self.contentvec_root / "config.json").resolve(strict=False),
            (self.contentvec_root / "pytorch_model.bin").resolve(strict=False),
            self.rmvpe_path.resolve(strict=False),
        }
        approved = {entry.path for entry in self.approved_files}
        if self.approved_files and (len(self.approved_files) != 5 or len(approved) != 5):
            raise ValueError("RVC settings require the five manifest-approved artifacts")
        return self

    @classmethod
    def from_runtime_paths(
        cls, paths: RuntimePaths, manifest: VoiceArtifactManifest | None = None
    ) -> "RvcSettings":
        """Build the only production selection from the checked voice artifact manifest."""
        root = paths.voice_root.resolve(strict=False)
        if manifest is None:
            manifest_path = root / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            manifest = VoiceArtifactManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        if manifest.rvc.epoch != 200:
            raise ValueError("RVC artifact must select epoch 200")
        weight_entry = manifest.file_entry(manifest.rvc.weight)
        index_entry = manifest.file_entry(manifest.rvc.index)
        if weight_entry.sha256 != EPOCH_200_WEIGHT_SHA256:
            raise ValueError("voice manifest must select the epoch-200 weight hash")
        if index_entry.sha256 != EPOCH_200_INDEX_SHA256:
            raise ValueError("voice manifest must select the epoch-200 index hash")
        required_relatives = (
            manifest.rvc.weight,
            manifest.rvc.index,
            manifest.contentvec.config,
            manifest.contentvec.model,
            manifest.rmvpe,
        )
        entries = tuple(manifest.file_entry(relative) for relative in required_relatives)
        resolved = tuple((root / entry.path).resolve(strict=False) for entry in entries)
        if any(not path.is_relative_to(root) for path in resolved):
            raise ValueError("RVC manifest path escapes the runtime voice root")
        return cls(
            weight=resolved[0],
            index=resolved[1],
            contentvec_root=resolved[2].parent,
            rmvpe_path=resolved[4],
            index_rate=manifest.rvc.index_rate,
            pitch=manifest.rvc.pitch,
            protect=manifest.rvc.protect,
            f0_method=manifest.rvc.f0_method,
            audio_format=manifest.audio_format,
            epoch=manifest.rvc.epoch,
            approved_files=entries,
        )


class ApplioVoiceConverter:
    """The pinned Applio wrapper; imports occur only when a conversion is loaded."""

    def __init__(self, vendor_root: Path, *, work_root: Path | None = None) -> None:
        self.vendor_root = Path(vendor_root).resolve(strict=False)
        self.work_root = Path(work_root).resolve(strict=False) if work_root else None
        self._converter: object | None = None
        self._overlay_root: Path | None = None

    def _create_overlay(self, settings: RvcSettings) -> Path:
        if self._overlay_root is not None:
            return self._overlay_root
        parent = self.work_root or Path(tempfile.gettempdir())
        parent.mkdir(parents=True, exist_ok=True)
        overlay = Path(tempfile.mkdtemp(prefix="youtuber-rvc-", dir=parent))
        try:
            configs = self.vendor_root / "rvc" / "configs"
            if not configs.is_dir():
                raise FileNotFoundError(configs)
            for source in configs.rglob("*"):
                if source.is_dir():
                    continue
                target = overlay / source.relative_to(self.vendor_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            rmvpe_target = overlay / "rvc" / "models" / "predictors" / "rmvpe.pt"
            rmvpe_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(settings.rmvpe_path, rmvpe_target)
            self._overlay_root = overlay
            return overlay
        except BaseException:
            shutil.rmtree(overlay, ignore_errors=True)
            raise

    def _discard_overlay(self) -> None:
        overlay, self._overlay_root = self._overlay_root, None
        if overlay is not None:
            shutil.rmtree(overlay, ignore_errors=True)

    @contextmanager
    def _isolated_environment(self):
        if self._overlay_root is None:
            raise RuntimeError("Applio overlay is not prepared")
        with _isolated_applio_state(self.vendor_root, self._overlay_root):
            yield

    def load(self, settings: RvcSettings | None = None) -> None:
        if self._converter is not None:
            return
        if settings is None:
            raise ValueError("Applio load requires manifest-selected RVC settings")
        with _BACKEND_LOCK:
            if self._converter is not None:
                return
            try:
                self._create_overlay(settings)
                with self._isolated_environment():
                    module = importlib.import_module("rvc.infer.infer")
                    converter = getattr(module, "VoiceConverter")
                    self._converter = converter()
            except BaseException:
                self._converter = None
                self._discard_overlay()
                raise

    def convert_audio(self, **kwargs: object) -> object:
        with _BACKEND_LOCK:
            if self._converter is None:  # defensive: a failed load must never be reused
                raise RuntimeError("Applio backend is not loaded")
            try:
                with self._isolated_environment():
                    method = getattr(self._converter, "convert_audio")
                    return method(**kwargs)
            except BaseException:
                try:
                    self.unload()
                except BaseException:
                    pass
                raise

    def unload(self) -> None:
        with _BACKEND_LOCK:
            converter = self._converter
            if converter is None:
                self._discard_overlay()
                return
            try:
                with self._isolated_environment():
                    cleanup = getattr(converter, "cleanup_model", None)
                    if callable(cleanup):
                        cleanup()
            finally:
                self._converter = None
                self._discard_overlay()


MetadataLoader = Callable[[Path], Mapping[str, object]]
HashFile = Callable[[Path], str]


class RvcEngine:
    """Lifecycle-managed RVC converter that fails closed on artifact or WAV validation."""

    def __init__(
        self,
        settings: RvcSettings,
        backend: VoiceConverterBackend | None = None,
        *,
        vendor_root: Path | None = None,
        work_root: Path | None = None,
        metadata_loader: MetadataLoader | None = None,
        hash_file: HashFile = _sha256_file,
    ) -> None:
        self.settings = settings
        self.vendor_root = Path(vendor_root or Path(__file__).resolve().parents[1] / "vendor" / "applio").resolve(strict=False)
        self.overlay_work_root = Path(work_root).resolve(strict=False) if work_root else None
        self._backend = backend or ApplioVoiceConverter(self.vendor_root, work_root=self.overlay_work_root)
        self._metadata_loader = metadata_loader or self._load_metadata_lazily
        self._hash_file = hash_file
        self._loaded = False

    @classmethod
    def from_runtime_paths(
        cls,
        paths: RuntimePaths,
        manifest: VoiceArtifactManifest | None = None,
        *,
        vendor_root: Path | None = None,
    ) -> "RvcEngine":
        settings = RvcSettings.from_runtime_paths(paths, manifest)
        return cls(
            settings,
            vendor_root=vendor_root or paths.code_root / "vendor" / "applio",
            work_root=paths.temporary_audio_dir / "rvc-applio",
        )

    @staticmethod
    def _load_metadata_lazily(path: Path) -> Mapping[str, object]:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping):
            raise ValueError("RVC checkpoint metadata must be a mapping")
        return payload

    def _validate_artifacts(self) -> None:
        by_path = {entry.path: entry for entry in self.settings.approved_files}
        required = (
            (self.settings.weight, by_path.get(self._relative_for(self.settings.weight)), "epoch-200 weight"),
            (self.settings.index, by_path.get(self._relative_for(self.settings.index)), "epoch-200 index"),
            (self.settings.contentvec_root / "config.json", by_path.get(self._relative_for(self.settings.contentvec_root / "config.json")), "ContentVec config"),
            (self.settings.contentvec_root / "pytorch_model.bin", by_path.get(self._relative_for(self.settings.contentvec_root / "pytorch_model.bin")), "ContentVec model"),
            (self.settings.rmvpe_path, by_path.get(self._relative_for(self.settings.rmvpe_path)), "RMVPE asset"),
        )
        for path, entry, label in required:
            if entry is None:
                raise ValueError(f"RVC settings lack a manifest identity for {label}")
            if not path.is_file():
                raise ValueError(f"{label} size mismatch")
            expected_hash = entry.sha256
            if path == self.settings.weight:
                expected_hash = EPOCH_200_WEIGHT_SHA256
            elif path == self.settings.index:
                expected_hash = EPOCH_200_INDEX_SHA256
            if self._hash_file(path) != expected_hash:
                raise ValueError(f"{label} hash mismatch")
            if path.stat().st_size != entry.size:
                raise ValueError(f"{label} size mismatch")

    def _relative_for(self, path: Path) -> str:
        roots = [self.settings.weight.parent]
        for entry in self.settings.approved_files:
            candidate = Path(entry.path)
            if candidate.name == path.name:
                return entry.path
        # The lookup is only reached for a malformed manually-built test instance.
        return ""

    def _validate_metadata(self, payload: Mapping[str, object]) -> None:
        config = payload.get("config")
        weights = payload.get("weight")
        if not isinstance(config, (list, tuple)) or len(config) < 3:
            raise ValueError("RVC checkpoint metadata requires a non-empty config")
        if not isinstance(weights, Mapping) or not weights:
            raise ValueError("RVC checkpoint metadata requires a non-empty weight mapping")
        if payload.get("version") != "v2" or payload.get("f0") is not True:
            raise ValueError("RVC checkpoint metadata must select RVC v2 with F0")
        if payload.get("epoch") != 200:
            raise ValueError("RVC checkpoint metadata must select epoch 200")
        if payload.get("sr") != 32_000:
            raise ValueError("RVC checkpoint metadata must select 32000 Hz")
        if payload.get("embedder_model") != "contentvec" or payload.get("vocoder") != "HiFi-GAN":
            raise ValueError("RVC checkpoint metadata does not match ContentVec/HiFi-GAN")

    def load(self) -> None:
        with _BACKEND_LOCK:
            if self._loaded:
                return
            self._validate_artifacts()
            self._validate_metadata(self._metadata_loader(self.settings.weight))
            load = getattr(self._backend, "load", None)
            if callable(load):
                load(self.settings)
            self._loaded = True

    def unload(self) -> None:
        with _BACKEND_LOCK:
            if not self._loaded:
                return
            try:
                unload = getattr(self._backend, "unload", None)
                if callable(unload):
                    unload()
            finally:
                self._loaded = False

    def convert(self, source: Path, output: Path) -> AudioArtifact:
        source = Path(source)
        output = Path(output)
        with _BACKEND_LOCK:
            source_rate, source_frames = self._validate_wav(source, "source", clipping_guard=None)
            self.load()
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name + ".partial")
            temporary.unlink(missing_ok=True)
            try:
                self._backend.convert_audio(
                    audio_input_path=str(source),
                    audio_output_path=str(temporary),
                    model_path=str(self.settings.weight),
                    index_path=str(self.settings.index),
                    pitch=self.settings.pitch,
                    f0_method=self.settings.f0_method,
                    index_rate=self.settings.index_rate,
                    protect=self.settings.protect,
                    embedder_model="custom",
                    embedder_model_custom=str(self.settings.contentvec_root),
                    export_format="WAV",
                )
                output_rate, output_frames = self._validate_wav(
                    temporary, "converted output", clipping_guard=0.001
                )
                ratio = (output_frames / output_rate) / (source_frames / source_rate)
                if not 0.90 <= ratio <= 1.10:
                    raise ValueError(f"converted output duration ratio must be 0.90..1.10, got {ratio:.3f}")
                os.replace(temporary, output)
            except BaseException:
                try:
                    self.unload()
                except BaseException:
                    pass
                temporary.unlink(missing_ok=True)
                raise
            return AudioArtifact(
                path=output,
                sample_rate=output_rate,
                duration_seconds=output_frames / output_rate,
                sha256=self._hash_file(output),
            )

    @staticmethod
    def _validate_wav(path: Path, label: str, *, clipping_guard: float | None) -> tuple[int, int]:
        try:
            info = sf.info(path)
            samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        except RuntimeError as error:
            raise ValueError(f"{label} is not a readable WAV: {path}") from error
        waveform = np.asarray(samples, dtype=np.float32)
        if info.format != "WAV" or info.channels != 1 or info.frames == 0 or sample_rate <= 0:
            raise ValueError(f"{label} is not a readable WAV: {path}")
        if waveform.shape != (info.frames, 1) or not np.isfinite(waveform).all():
            raise ValueError(f"{label} must contain finite mono audio")
        if clipping_guard is not None:
            clipped_fraction = float(np.count_nonzero(np.abs(waveform) >= 1.0)) / waveform.size
            if clipped_fraction >= clipping_guard:
                raise ValueError(f"{label} exceeds clipping guard")
        return sample_rate, info.frames
