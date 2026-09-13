"""Behavior tests for the epoch-200, manifest-selected RVC adapter."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from runtime.paths import RuntimePaths
from voice.models import VoiceArtifactManifest
from voice.rvc_engine import (
    EPOCH_200_INDEX_SHA256,
    EPOCH_200_WEIGHT_SHA256,
    ApplioVoiceConverter,
    RvcEngine,
    RvcSettings,
    probe_applio_runtime,
)


FIXTURE = Path("tests/fixtures/voice/sine-24000.wav")


class FakeVoiceConverter:
    """A deterministic stand-in at the external Applio boundary."""

    def __init__(self, *, fail: bool = False, concurrent: list[int] | None = None) -> None:
        self.fail = fail
        self.concurrent = concurrent
        self.call: dict[str, object] | None = None
        self.unloaded = 0

    def convert_audio(self, **kwargs: object) -> str:
        self.call = kwargs
        if self.concurrent is not None:
            self.concurrent[0] += 1
            self.concurrent[1] = max(self.concurrent[1], self.concurrent[0])
            time.sleep(0.03)
            self.concurrent[0] -= 1
        if self.fail:
            raise RuntimeError("backend conversion failed")
        shutil.copyfile(Path(str(kwargs["audio_input_path"])), Path(str(kwargs["audio_output_path"])))
        return str(kwargs["audio_output_path"])

    def unload(self) -> None:
        self.unloaded += 1


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "source.wav"
    shutil.copyfile(FIXTURE, source)
    return source


def _write_assets(root: Path) -> dict[str, Path]:
    paths = {
        "weight": root / "rvc" / "speaker_epoch_200.pth",
        "index": root / "rvc" / "speaker.index",
        "contentvec_config": root / "rvc" / "models" / "embedders" / "contentvec" / "config.json",
        "contentvec_model": root / "rvc" / "models" / "embedders" / "contentvec" / "pytorch_model.bin",
        "rmvpe": root / "rvc" / "models" / "predictors" / "rmvpe.pt",
    }
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode("ascii"))
    return paths


def _manifest(root: Path, assets: dict[str, Path]) -> VoiceArtifactManifest:
    files = []
    for path in assets.values():
        relative = path.relative_to(root).as_posix()
        files.append({"path": relative, "size": path.stat().st_size, "sha256": _hash(path)})
    files[0]["sha256"] = EPOCH_200_WEIGHT_SHA256
    files[1]["sha256"] = EPOCH_200_INDEX_SHA256
    xtts = root / "xtts"
    xtts.mkdir(exist_ok=True)
    for name in ("best_model.pth", "config.json", "vocab.json"):
        path = xtts / name
        path.write_bytes(name.encode("ascii"))
        files.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": _hash(path)})
    reference = root / "references" / "approved.wav"
    reference.parent.mkdir(exist_ok=True)
    sf.write(reference, np.zeros(2_400, dtype=np.float32), 24_000, subtype="PCM_16")
    files.append({"path": reference.relative_to(root).as_posix(), "size": reference.stat().st_size, "sha256": _hash(reference)})
    return VoiceArtifactManifest.model_validate(
        {
            "schema": "youtuber.voice.v1",
            "runtime_api": 1,
            "sample_rate": 24_000,
            "audio_format": "WAV",
            "files": files,
            "xtts": {
                "checkpoint": "xtts/best_model.pth",
                "config": "xtts/config.json",
                "vocab": "xtts/vocab.json",
                "reference_wavs": ["references/approved.wav"],
            },
            "rvc": {
                "weight": "rvc/speaker_epoch_200.pth",
                "index": "rvc/speaker.index",
                "epoch": 200,
                "index_rate": 0.75,
                "pitch": 0,
                "f0_method": "rmvpe",
                "protect": 0.5,
            },
            "contentvec": {
                "config": "rvc/models/embedders/contentvec/config.json",
                "model": "rvc/models/embedders/contentvec/pytorch_model.bin",
            },
            "rmvpe": "rvc/models/predictors/rmvpe.pt",
        }
    )


@pytest.fixture
def rvc_settings(tmp_path: Path) -> RvcSettings:
    paths = RuntimePaths.developer(tmp_path)
    assets = _write_assets(paths.voice_root)
    return RvcSettings.from_runtime_paths(paths, _manifest(paths.voice_root, assets))


@pytest.fixture
def expected_hashes(rvc_settings: RvcSettings) -> dict[Path, str]:
    return {
        rvc_settings.weight: EPOCH_200_WEIGHT_SHA256,
        rvc_settings.index: EPOCH_200_INDEX_SHA256,
        rvc_settings.contentvec_root / "config.json": _hash(rvc_settings.contentvec_root / "config.json"),
        rvc_settings.contentvec_root / "pytorch_model.bin": _hash(rvc_settings.contentvec_root / "pytorch_model.bin"),
        rvc_settings.rmvpe_path: _hash(rvc_settings.rmvpe_path),
    }


def _fake_hash(expected: dict[Path, str]):
    def calculate(path: Path) -> str:
        path = Path(path)
        actual = _hash(path)
        if path in expected and actual == hashlib.sha256(path.name.encode("ascii")).hexdigest():
            return expected[path]
        return actual

    return calculate


def _metadata() -> dict[str, object]:
    return {
        "config": [1, 2, 3],
        "weight": {"emb_g.weight": object()},
        "version": "v2",
        "f0": True,
        "epoch": 200,
        "sr": 32_000,
        "embedder_model": "contentvec",
        "vocoder": "HiFi-GAN",
    }


def _engine(settings: RvcSettings, backend: FakeVoiceConverter, expected: dict[Path, str]) -> RvcEngine:
    return RvcEngine(
        settings,
        backend=backend,
        metadata_loader=lambda path: _metadata(),
        hash_file=_fake_hash(expected),
        vendor_root=settings.weight.parent,
    )


def _write_vendor_configs(vendor: Path) -> None:
    configs = vendor / "rvc" / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "24000.json").write_text("{}", encoding="utf-8")
    (configs / "config.py").write_text("VALUE = 1\n", encoding="utf-8")


def _write_importable_vendor(vendor: Path) -> None:
    _write_vendor_configs(vendor)
    infer = vendor / "rvc" / "infer"
    infer.mkdir(parents=True, exist_ok=True)
    (infer / "infer.py").write_text(
        "class VoiceConverter:\n"
        "    def convert_audio(self, **kwargs):\n"
        "        return None\n"
        "    def cleanup_model(self):\n"
        "        return None\n",
        encoding="utf-8",
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _pinned_vendor_source_inventory(vendor: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for relative in ("rvc/configs", "rvc/infer", "rvc/lib"):
        root = vendor / relative
        for path in root.rglob("*"):
            if path.is_file():
                files[path.relative_to(vendor).as_posix()] = path.read_bytes()
    return files


def test_rvc_engine_forwards_locked_conversion_settings(
    tmp_path: Path, rvc_settings: RvcSettings, expected_hashes: dict[Path, str]
):
    """Removing a locked Applio argument must prevent the approved conversion."""
    backend = FakeVoiceConverter()
    source = copy_fixture(tmp_path)
    output = tmp_path / "converted.wav"

    _engine(rvc_settings, backend, expected_hashes).convert(source, output)

    assert backend.call is not None
    assert backend.call["index_rate"] == 0.75
    assert backend.call["pitch"] == 0
    assert backend.call["f0_method"] == "rmvpe"
    assert backend.call["protect"] == 0.5
    assert backend.call["export_format"] == "WAV"
    assert backend.call["embedder_model"] == "custom"
    assert backend.call["embedder_model_custom"] == str(rvc_settings.contentvec_root)
    assert output.is_file()
    assert Path.cwd() == Path(__file__).resolve().parents[2]


def test_rvc_engine_rejects_wrong_selected_weight_hash_before_metadata_load(
    rvc_settings: RvcSettings, expected_hashes: dict[Path, str]
):
    """Changing the selected weight must stop before any checkpoint parsing or backend state."""
    rvc_settings.weight.write_bytes(b"wrong")
    metadata_calls = 0

    def metadata_loader(path: Path) -> dict[str, object]:
        nonlocal metadata_calls
        metadata_calls += 1
        return _metadata()

    with pytest.raises(ValueError, match="epoch-200 weight hash"):
        RvcEngine(
            rvc_settings,
            backend=FakeVoiceConverter(),
            metadata_loader=metadata_loader,
            hash_file=_fake_hash(expected_hashes),
            vendor_root=rvc_settings.weight.parent,
        ).load()

    assert metadata_calls == 0


def test_rvc_settings_is_frozen_and_requires_manifest_selected_runtime_paths(tmp_path: Path):
    """Production settings must originate from an approved manifest under the voice root."""
    paths = RuntimePaths.developer(tmp_path)
    assets = _write_assets(paths.voice_root)
    settings = RvcSettings.from_runtime_paths(paths, _manifest(paths.voice_root, assets))

    with pytest.raises(ValidationError):
        settings.index_rate = 0.5  # type: ignore[misc]
    bad_manifest = _manifest(paths.voice_root, assets).model_dump(by_alias=True)
    for entry in bad_manifest["files"]:
        if entry["path"] == "rvc/speaker_epoch_200.pth":
            entry["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="epoch-200 weight hash"):
        RvcSettings.from_runtime_paths(
            paths,
            VoiceArtifactManifest.model_validate(bad_manifest),
        )


def test_rvc_settings_exposes_the_documented_locked_constructor_defaults(tmp_path: Path):
    """Callers can form the documented value, but only a manifest-selected value can load."""
    settings = RvcSettings(
        weight=tmp_path / "weight.pth",
        index=tmp_path / "index.faiss",
        contentvec_root=tmp_path / "contentvec",
        rmvpe_path=tmp_path / "rmvpe.pt",
    )

    assert (settings.index_rate, settings.pitch, settings.protect, settings.f0_method, settings.audio_format) == (
        0.75,
        0,
        0.5,
        "rmvpe",
        "WAV",
    )
    with pytest.raises(ValueError, match="manifest identity"):
        RvcEngine(settings, backend=FakeVoiceConverter()).load()


def test_rvc_settings_loads_the_production_manifest_from_runtime_voice_root(tmp_path: Path):
    """The production factory reads only the manifest colocated with the runtime voice root."""
    paths = RuntimePaths.developer(tmp_path)
    assets = _write_assets(paths.voice_root)
    manifest = _manifest(paths.voice_root, assets)
    (paths.voice_root / "manifest.json").write_text(manifest.model_dump_json(by_alias=True), encoding="utf-8")

    settings = RvcSettings.from_runtime_paths(paths)

    assert settings.weight == paths.voice_root / "rvc" / "speaker_epoch_200.pth"


def test_rvc_engine_runtime_factory_uses_the_runtime_temporary_audio_root(tmp_path: Path):
    """Packaged construction owns its overlay beneath RuntimePaths' writable cache root."""
    paths = RuntimePaths.developer(tmp_path)
    assets = _write_assets(paths.voice_root)
    engine = RvcEngine.from_runtime_paths(paths, _manifest(paths.voice_root, assets))

    assert engine.overlay_work_root == paths.temporary_audio_dir / "rvc-applio"


def test_rvc_engine_rejects_invalid_checkpoint_metadata_without_importing_torch(
    rvc_settings: RvcSettings,
    expected_hashes: dict[Path, str],
    forbid_torch_import,
):
    """A malformed checkpoint cannot reach the backend and unit tests need no Torch import."""
    with pytest.raises(ValueError, match="checkpoint metadata"):
        RvcEngine(
            rvc_settings,
            backend=FakeVoiceConverter(),
            metadata_loader=lambda path: {"config": [], "weight": {}},
            hash_file=_fake_hash(expected_hashes),
            vendor_root=rvc_settings.weight.parent,
        ).load()


def test_rvc_engine_restores_cwd_after_backend_failure(
    tmp_path: Path, rvc_settings: RvcSettings, expected_hashes: dict[Path, str]
):
    """A failing backend cannot leak Applio's vendor directory into its caller."""
    original = Path.cwd()
    with pytest.raises(RuntimeError, match="backend conversion failed"):
        _engine(rvc_settings, FakeVoiceConverter(fail=True), expected_hashes).convert(
            copy_fixture(tmp_path), tmp_path / "converted.wav"
        )
    assert Path.cwd() == original


def test_applio_import_failure_restores_cwd_and_stays_lazy(
    monkeypatch, tmp_path: Path, rvc_settings: RvcSettings, forbid_torch_import
):
    """A failed pinned import restores the caller CWD without importing Torch eagerly."""
    original = Path.cwd()

    vendor = tmp_path / "vendor"
    _write_vendor_configs(vendor)

    def fail_import(name: str):
        assert name == "rvc.infer.infer"
        assert Path.cwd() != vendor
        raise RuntimeError("pinned Applio import failed")

    monkeypatch.setattr(importlib, "import_module", fail_import)

    with pytest.raises(RuntimeError, match="pinned Applio import failed"):
        ApplioVoiceConverter(vendor, work_root=tmp_path / "overlays").load(rvc_settings)

    assert Path.cwd() == original
    assert not any((tmp_path / "overlays").iterdir())


def test_applio_runtime_probe_serializes_with_the_global_backend_lock_and_restores_state(
    monkeypatch, tmp_path: Path
):
    """Readiness cannot interleave CWD/sys.path mutations with an RVC backend operation."""
    import voice.rvc_engine as rvc_module

    vendor = tmp_path / "vendor"
    source = vendor / "rvc" / "infer" / "infer.py"
    source.parent.mkdir(parents=True)
    source.write_text("# probe fixture\n", encoding="utf-8")
    original_cwd, original_path, original_bytecode = Path.cwd(), list(sys.path), sys.dont_write_bytecode
    entered = threading.Event()
    release = threading.Event()

    def import_module(name: str):
        assert name == "rvc.infer.infer"
        entered.set()
        assert release.wait(2)
        return SimpleNamespace(__file__=str(source), VoiceConverter=object)

    monkeypatch.setattr(importlib, "import_module", import_module)
    thread = threading.Thread(target=lambda: probe_applio_runtime(vendor), daemon=True)
    with rvc_module._BACKEND_LOCK:
        thread.start()
        assert not entered.wait(0.1)
    assert entered.wait(2)
    release.set()
    thread.join(2)

    assert Path.cwd() == original_cwd
    assert sys.path == original_path
    assert sys.dont_write_bytecode is original_bytecode
    assert source.read_text(encoding="utf-8") == "# probe fixture\n"


def test_applio_backend_binds_the_validated_rmvpe_asset_before_import(
    monkeypatch, tmp_path: Path, rvc_settings: RvcSettings
):
    """The pinned backend must use the manifest-selected RMVPE file, not a download."""
    vendor = tmp_path / "vendor"
    _write_vendor_configs(vendor)
    target = vendor / "rvc" / "models" / "predictors" / "rmvpe.pt"

    def fail_after_binding(name: str):
        assert name == "rvc.infer.infer"
        assert (Path.cwd() / "rvc" / "models" / "predictors" / "rmvpe.pt").read_bytes() == rvc_settings.rmvpe_path.read_bytes()
        assert not target.exists()
        raise RuntimeError("stop after RMVPE binding")

    monkeypatch.setattr(importlib, "import_module", fail_after_binding)

    with pytest.raises(RuntimeError, match="stop after RMVPE binding"):
        ApplioVoiceConverter(vendor, work_root=tmp_path / "overlays").load(rvc_settings)

    assert not target.exists()


def test_applio_restores_the_entire_sys_path_after_upstream_import_and_convert_mutations(
    monkeypatch, tmp_path: Path, rvc_settings: RvcSettings
):
    """Upstream path mutations must not leak through any lazy backend boundary."""
    vendor = tmp_path / "readonly-vendor"
    overlay_root = tmp_path / "overlays"
    _write_vendor_configs(vendor)
    original_cwd = Path.cwd()
    original_path = list(sys.path)

    class Converter:
        def convert_audio(self, **kwargs: object) -> None:
            assert Path.cwd() != vendor
            sys.path.extend(["upstream-convert-a", "upstream-convert-b"])

        def cleanup_model(self) -> None:
            sys.path.append("upstream-cleanup")

    def import_module(name: str):
        assert name == "rvc.infer.infer"
        assert Path.cwd() != vendor
        sys.path.extend(["upstream-import-a", "upstream-import-b"])
        return SimpleNamespace(VoiceConverter=Converter)

    monkeypatch.setattr(importlib, "import_module", import_module)
    backend = ApplioVoiceConverter(vendor, work_root=overlay_root)
    vendor_before = _tree_bytes(vendor)
    vendor_files = tuple(path for path in vendor.rglob("*") if path.is_file())
    for path in vendor_files:
        path.chmod(stat.S_IREAD)
    try:
        backend.load(rvc_settings)
        assert sys.path == original_path
        assert Path.cwd() == original_cwd
        backend.convert_audio(audio_input_path="input", audio_output_path="output")
        assert sys.path == original_path
        assert Path.cwd() == original_cwd
        backend.unload()
    finally:
        for path in vendor_files:
            path.chmod(stat.S_IREAD | stat.S_IWRITE)

    assert sys.path == original_path
    assert Path.cwd() == original_cwd
    assert _tree_bytes(vendor) == vendor_before
    assert not any(overlay_root.iterdir())


def test_applio_real_vendor_import_does_not_write_bytecode_or_leak_interpreter_state(
    monkeypatch, tmp_path: Path, rvc_settings: RvcSettings
):
    """A real vendor import must leave the verified source tree and bytecode flag unchanged."""
    vendor = tmp_path / "vendor"
    _write_importable_vendor(vendor)
    for module_name in tuple(name for name in sys.modules if name == "rvc" or name.startswith("rvc.")):
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    importlib.invalidate_caches()
    before = _tree_bytes(vendor)
    before_flag = sys.dont_write_bytecode

    backend = ApplioVoiceConverter(vendor, work_root=tmp_path / "overlays")
    backend.load(rvc_settings)
    backend.convert_audio(audio_input_path="input", audio_output_path="output")
    backend.unload()

    assert _tree_bytes(vendor) == before
    assert sys.dont_write_bytecode is before_flag


def test_rvc_engine_reloads_after_a_backend_conversion_failure(
    tmp_path: Path, rvc_settings: RvcSettings, expected_hashes: dict[Path, str]
):
    """A failed conversion clears engine residency so the next conversion loads a fresh backend."""
    class ResettingBackend:
        def __init__(self) -> None:
            self.loaded = 0
            self.unloaded = 0
            self.active = False
            self.calls = 0

        def load(self, settings: RvcSettings) -> None:
            self.loaded += 1
            self.active = True

        def convert_audio(self, **kwargs: object) -> None:
            self.calls += 1
            if self.calls == 1:
                self.active = False
                raise RuntimeError("conversion failed")
            if not self.active:
                raise RuntimeError("backend was not reloaded")
            shutil.copyfile(Path(str(kwargs["audio_input_path"])), Path(str(kwargs["audio_output_path"])))

        def unload(self) -> None:
            self.unloaded += 1
            self.active = False

    backend = ResettingBackend()
    engine = RvcEngine(
        rvc_settings,
        backend=backend,
        metadata_loader=lambda path: _metadata(),
        hash_file=_fake_hash(expected_hashes),
        vendor_root=tmp_path,
    )
    output = tmp_path / "converted.wav"

    with pytest.raises(RuntimeError, match="conversion failed"):
        engine.convert(copy_fixture(tmp_path), output)
    artifact = engine.convert(copy_fixture(tmp_path), output)

    assert artifact.path == output
    assert backend.loaded == 2
    assert backend.unloaded == 1


def test_rvc_engine_preserves_conversion_error_when_backend_unload_also_fails(
    tmp_path: Path, rvc_settings: RvcSettings, expected_hashes: dict[Path, str]
):
    """A cleanup error cannot mask conversion failure or leave the engine marked loaded."""
    class CleanupFailureBackend:
        def load(self, settings: RvcSettings) -> None:
            return None

        def convert_audio(self, **kwargs: object) -> None:
            raise RuntimeError("conversion failed")

        def unload(self) -> None:
            raise RuntimeError("cleanup failed")

    engine = RvcEngine(
        rvc_settings,
        backend=CleanupFailureBackend(),
        metadata_loader=lambda path: _metadata(),
        hash_file=_fake_hash(expected_hashes),
        vendor_root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="conversion failed"):
        engine.convert(copy_fixture(tmp_path), tmp_path / "converted.wav")

    assert engine._loaded is False


def test_applio_cleanup_runs_once_and_clears_state_even_when_cleanup_raises(
    monkeypatch, tmp_path: Path, rvc_settings: RvcSettings
):
    """One unload calls cleanup once, clears state, restores process state, and cleans its overlay."""
    vendor = tmp_path / "vendor"
    overlay_root = tmp_path / "overlays"
    _write_vendor_configs(vendor)
    original_cwd = Path.cwd()
    original_path = list(sys.path)
    calls = 0

    class Converter:
        def cleanup_model(self) -> None:
            nonlocal calls
            calls += 1
            sys.path.append("upstream-cleanup-error")
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(
        importlib, "import_module", lambda name: SimpleNamespace(VoiceConverter=Converter)
    )
    backend = ApplioVoiceConverter(vendor, work_root=overlay_root)
    backend.load(rvc_settings)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        backend.unload()
    backend.unload()

    assert calls == 1
    assert backend._converter is None
    assert Path.cwd() == original_cwd
    assert sys.path == original_path
    assert not any(overlay_root.iterdir())


def test_rvc_engine_unloads_applio_cleanup_once_per_loaded_lifecycle(
    monkeypatch, tmp_path: Path, rvc_settings: RvcSettings, expected_hashes: dict[Path, str]
):
    """Engine lifecycle delegates exactly one real backend cleanup and remains idempotent."""
    vendor = tmp_path / "vendor"
    _write_vendor_configs(vendor)
    cleanup_calls = 0

    class Converter:
        def cleanup_model(self) -> None:
            nonlocal cleanup_calls
            cleanup_calls += 1

    monkeypatch.setattr(
        importlib, "import_module", lambda name: SimpleNamespace(VoiceConverter=Converter)
    )
    engine = RvcEngine(
        rvc_settings,
        backend=ApplioVoiceConverter(vendor, work_root=tmp_path / "overlays"),
        metadata_loader=lambda path: _metadata(),
        hash_file=_fake_hash(expected_hashes),
        vendor_root=vendor,
    )

    engine.load()
    engine.unload()
    engine.unload()

    assert cleanup_calls == 1


def test_rvc_engine_is_deterministically_unloaded_when_applio_cleanup_raises(
    monkeypatch, tmp_path: Path, rvc_settings: RvcSettings, expected_hashes: dict[Path, str]
):
    """Cleanup errors surface once without leaving engine state, CWD, or sys.path behind."""
    vendor = tmp_path / "vendor"
    _write_vendor_configs(vendor)
    original_cwd = Path.cwd()
    original_path = list(sys.path)
    cleanup_calls = 0

    class Converter:
        def cleanup_model(self) -> None:
            nonlocal cleanup_calls
            cleanup_calls += 1
            sys.path.append("cleanup-leak")
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr(
        importlib, "import_module", lambda name: SimpleNamespace(VoiceConverter=Converter)
    )
    engine = RvcEngine(
        rvc_settings,
        backend=ApplioVoiceConverter(vendor, work_root=tmp_path / "overlays"),
        metadata_loader=lambda path: _metadata(),
        hash_file=_fake_hash(expected_hashes),
        vendor_root=vendor,
    )
    engine.load()

    with pytest.raises(RuntimeError, match="cleanup failed"):
        engine.unload()
    engine.unload()

    assert cleanup_calls == 1
    assert engine._loaded is False
    assert Path.cwd() == original_cwd
    assert sys.path == original_path
    assert not any((tmp_path / "overlays").iterdir())


def test_rvc_engine_preserves_prior_valid_output_when_converted_audio_is_invalid(
    tmp_path: Path, rvc_settings: RvcSettings, expected_hashes: dict[Path, str]
):
    """Bad converted audio must not replace the caller's already valid WAV."""
    output = tmp_path / "converted.wav"
    shutil.copyfile(FIXTURE, output)
    original = output.read_bytes()

    class InvalidAudio(FakeVoiceConverter):
        def convert_audio(self, **kwargs: object) -> str:
            Path(str(kwargs["audio_output_path"])).write_bytes(b"not a WAV")
            return str(kwargs["audio_output_path"])

    with pytest.raises(ValueError, match="readable WAV"):
        _engine(rvc_settings, InvalidAudio(), expected_hashes).convert(copy_fixture(tmp_path), output)

    assert output.read_bytes() == original
    assert not output.with_name("converted.wav.partial").exists()


def test_rvc_engine_serializes_backend_use_and_unloads_once(
    tmp_path: Path, rvc_settings: RvcSettings, expected_hashes: dict[Path, str]
):
    """Concurrent callers cannot overlap mutable Applio state and unload is idempotent."""
    concurrency = [0, 0]
    backend = FakeVoiceConverter(concurrent=concurrency)
    engine = _engine(rvc_settings, backend, expected_hashes)
    errors: list[BaseException] = []

    def convert(index: int) -> None:
        try:
            engine.convert(copy_fixture(tmp_path), tmp_path / f"converted-{index}.wav")
        except BaseException as error:  # pragma: no cover - asserted below
            errors.append(error)

    threads = [threading.Thread(target=convert, args=(index,)) for index in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    engine.unload()
    engine.unload()

    assert errors == []
    assert concurrency[1] == 1
    assert backend.unloaded == 1


def test_rvc_engine_rejects_selected_index_size_before_backend_load(
    rvc_settings: RvcSettings, expected_hashes: dict[Path, str]
):
    """A truncated FAISS index cannot reach metadata parsing or conversion."""
    rvc_settings.index.write_bytes(b"short")

    def hash_file(path: Path) -> str:
        if Path(path) == rvc_settings.index:
            return EPOCH_200_INDEX_SHA256
        return _fake_hash(expected_hashes)(path)

    with pytest.raises(ValueError, match="epoch-200 index size"):
        RvcEngine(
            rvc_settings,
            backend=FakeVoiceConverter(),
            metadata_loader=lambda path: _metadata(),
            hash_file=hash_file,
            vendor_root=rvc_settings.weight.parent,
        ).load()


@pytest.mark.parametrize(
    ("writer", "message"),
    [
        (lambda path: path.write_bytes(b"not WAV"), "readable WAV"),
        (
            lambda path: sf.write(
                path, np.full(24_000, np.nan, dtype=np.float32), 24_000, format="WAV", subtype="FLOAT"
            ),
            "finite mono",
        ),
    ],
)
def test_rvc_engine_rejects_invalid_source_wav_before_conversion(
    tmp_path: Path,
    rvc_settings: RvcSettings,
    expected_hashes: dict[Path, str],
    writer,
    message: str,
):
    """Unreadable or non-finite caller audio must not be handed to Applio."""
    source = tmp_path / "source.wav"
    writer(source)
    backend = FakeVoiceConverter()

    with pytest.raises(ValueError, match=message):
        _engine(rvc_settings, backend, expected_hashes).convert(source, tmp_path / "output.wav")

    assert backend.call is None


def test_rvc_engine_rejects_duration_drift_and_preserves_existing_output(
    tmp_path: Path, rvc_settings: RvcSettings, expected_hashes: dict[Path, str]
):
    """A conversion that changes duration by more than ten percent cannot be promoted."""
    output = tmp_path / "converted.wav"
    shutil.copyfile(FIXTURE, output)
    original = output.read_bytes()

    class DriftedAudio(FakeVoiceConverter):
        def convert_audio(self, **kwargs: object) -> str:
            sf.write(
                Path(str(kwargs["audio_output_path"])),
                np.zeros(48_000, dtype=np.float32),
                24_000,
                format="WAV",
            )
            return str(kwargs["audio_output_path"])

    with pytest.raises(ValueError, match="duration ratio"):
        _engine(rvc_settings, DriftedAudio(), expected_hashes).convert(copy_fixture(tmp_path), output)

    assert output.read_bytes() == original


def _hard_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError as error:
        raise AssertionError(f"slow smoke requires a same-volume hard link: {source}") from error


@pytest.mark.slow
def test_rvc_engine_real_epoch_200_smoke(voice_root: Path | None):
    """The exact --voice-root command performs a real isolated pinned-Applio conversion."""
    if voice_root is None:
        pytest.skip("pass --voice-root to run the real epoch-200 smoke")
    supplied_root = voice_root.resolve()
    artifacts = supplied_root / "artifacts"
    runtime = supplied_root.parent / "runtime"
    vendor = runtime / "Applio"
    rvc_python = runtime / ".venv-rvc" / "Scripts" / "python.exe"
    weight = artifacts / "speaker_rvc_full_v1_200e_15800s.pth"
    index = artifacts / "speaker_rvc_full_v1.index"
    contentvec = vendor / "rvc" / "models" / "embedders" / "contentvec"
    rmvpe = vendor / "rvc" / "models" / "predictors" / "rmvpe.pt"
    for required in (weight, index, contentvec / "config.json", contentvec / "pytorch_model.bin", rmvpe, vendor / "rvc" / "configs", rvc_python):
        assert required.exists(), f"required real smoke asset is missing: {required}"
    vendor_before = _pinned_vendor_source_inventory(vendor)

    workspace = Path(__file__).resolve().parents[2]
    build = workspace / "build"
    build.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="task6-rvc-smoke-", dir=build) as temporary:
        root = Path(temporary)
        voice = root / "models"
        links = {
            "rvc/speaker_epoch_200.pth": weight,
            "rvc/speaker.index": index,
            "rvc/models/embedders/contentvec/config.json": contentvec / "config.json",
            "rvc/models/embedders/contentvec/pytorch_model.bin": contentvec / "pytorch_model.bin",
            "rvc/models/predictors/rmvpe.pt": rmvpe,
            "references/approved.wav": FIXTURE.resolve(),
        }
        for relative, source in links.items():
            _hard_link(source, voice / relative)
        for relative, contents in {
            "xtts/best_model.pth": b"smoke-placeholder\n",
            "xtts/config.json": b"{}\n",
            "xtts/vocab.json": b"{}\n",
        }.items():
            path = voice / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
        files = [
            {"path": path.relative_to(voice).as_posix(), "size": path.stat().st_size, "sha256": _hash(path)}
            for path in sorted(voice.rglob("*"))
            if path.is_file()
        ]
        for entry in files:
            if entry["path"] == "rvc/speaker_epoch_200.pth":
                assert entry["sha256"] == EPOCH_200_WEIGHT_SHA256
            if entry["path"] == "rvc/speaker.index":
                assert entry["sha256"] == EPOCH_200_INDEX_SHA256
        manifest = {
            "schema": "youtuber.voice.v1",
            "runtime_api": 1,
            "sample_rate": 24_000,
            "audio_format": "WAV",
            "files": files,
            "xtts": {"checkpoint": "xtts/best_model.pth", "config": "xtts/config.json", "vocab": "xtts/vocab.json", "reference_wavs": ["references/approved.wav"]},
            "rvc": {"weight": "rvc/speaker_epoch_200.pth", "index": "rvc/speaker.index", "epoch": 200, "index_rate": 0.75, "pitch": 0, "f0_method": "rmvpe", "protect": 0.5},
            "contentvec": {"config": "rvc/models/embedders/contentvec/config.json", "model": "rvc/models/embedders/contentvec/pytorch_model.bin"},
            "rmvpe": "rvc/models/predictors/rmvpe.pt",
        }
        (voice / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        output = root / "converted.wav"
        code = (
            "from pathlib import Path; from types import SimpleNamespace; "
            "from voice.rvc_engine import RvcEngine, RvcSettings; "
            f"root=Path(r'{root}'); settings=RvcSettings.from_runtime_paths(SimpleNamespace(voice_root=root/'models')); "
            f"engine=RvcEngine(settings, vendor_root=Path(r'{vendor}'), work_root=root/'overlay'); "
            f"engine.convert(Path(r'{FIXTURE.resolve()}'), Path(r'{output}')); engine.unload()"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(workspace)
        completed = subprocess.run(
            [str(rvc_python), "-c", code],
            cwd=workspace,
            env=environment,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        samples, sample_rate = sf.read(output, dtype="float32", always_2d=True)
        source, source_rate = sf.read(FIXTURE, dtype="float32", always_2d=True)
        ratio = (samples.shape[0] / sample_rate) / (source.shape[0] / source_rate)
        clipped = float(np.count_nonzero(np.abs(samples) >= 1.0)) / samples.size
        assert samples.shape[1] == 1 and samples.size > 0 and np.isfinite(samples).all()
        assert 0.90 <= ratio <= 1.10
        assert clipped < 0.001
    assert _pinned_vendor_source_inventory(vendor) == vendor_before
