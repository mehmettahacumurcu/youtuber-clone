from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from runtime.paths import RuntimePaths
from voice.models import ArtifactFile, VoiceArtifactManifest, XttsSettings
from voice.xtts_engine import XttsEngine


class FakeBackend:
    def __init__(self, waveform: np.ndarray | None = None, sample_rate: int = 24_000):
        self.waveform = np.zeros(24_000, dtype=np.float32) if waveform is None else waveform
        self.sample_rate = sample_rate
        self.loaded = 0
        self.unloaded = 0

    def load(self, settings: XttsSettings) -> None:
        self.loaded += 1

    def synthesize(
        self, text: str, language: str, reference_wavs: tuple[Path, ...]
    ) -> tuple[np.ndarray, int]:
        assert text == "Merhaba dünya."
        assert language == "tr"
        assert reference_wavs
        return self.waveform, self.sample_rate

    def unload(self) -> None:
        self.unloaded += 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_reference(path: Path) -> None:
    sf.write(path, np.zeros(24_000, dtype=np.float32), 24_000, subtype="PCM_16")


@pytest.fixture
def xtts_settings(tmp_path: Path) -> XttsSettings:
    checkpoint_dir = tmp_path / "voice" / "xtts"
    checkpoint_dir.mkdir(parents=True)
    for name in ("best_model.pth", "config.json", "vocab.json"):
        (checkpoint_dir / name).write_text(name, encoding="utf-8")
    reference = tmp_path / "voice" / "references" / "approved.wav"
    reference.parent.mkdir()
    _write_reference(reference)
    return XttsSettings(checkpoint_dir=checkpoint_dir, reference_wavs=(reference,))


def test_xtts_engine_writes_a_valid_mono_wav(tmp_path: Path, xtts_settings: XttsSettings):
    output = tmp_path / "raw.wav"

    artifact = XttsEngine(xtts_settings, backend=FakeBackend()).synthesize(
        "Merhaba dünya.", output
    )

    assert artifact.path == output
    assert artifact.sample_rate == 24_000
    assert artifact.duration_seconds == pytest.approx(1.0)
    assert artifact.sha256 == _sha256(output)
    assert sf.info(output).channels == 1
    assert sf.info(output).samplerate == 24_000


def test_xtts_engine_reloads_after_unload_without_importing_torch(
    tmp_path: Path, xtts_settings: XttsSettings, forbid_torch_import
):
    backend = FakeBackend()
    engine = XttsEngine(xtts_settings, backend=backend)

    engine.load()
    engine.unload()
    engine.synthesize("Merhaba dünya.", tmp_path / "raw.wav")

    assert backend.loaded == 2
    assert backend.unloaded == 1


@pytest.mark.parametrize("text", ["", " \n\t "])
def test_xtts_engine_rejects_blank_text(
    tmp_path: Path, xtts_settings: XttsSettings, text: str
):
    with pytest.raises(ValueError, match="text must not be blank"):
        XttsEngine(xtts_settings, backend=FakeBackend()).synthesize(text, tmp_path / "raw.wav")


def test_xtts_engine_rejects_missing_checkpoint_before_backend_load(
    tmp_path: Path, xtts_settings: XttsSettings
):
    (xtts_settings.checkpoint_dir / "vocab.json").unlink()
    backend = FakeBackend()

    with pytest.raises(FileNotFoundError, match="vocab.json"):
        XttsEngine(xtts_settings, backend=backend).load()

    assert backend.loaded == 0


def test_xtts_settings_rejects_non_wav_reference(tmp_path: Path):
    checkpoint_dir = tmp_path / "xtts"
    checkpoint_dir.mkdir()
    reference = tmp_path / "approved.mp3"
    reference.write_bytes(b"not wav")

    with pytest.raises(ValidationError, match="WAV"):
        XttsSettings(checkpoint_dir=checkpoint_dir, reference_wavs=(reference,))


def test_xtts_engine_rejects_a_reference_that_is_not_a_readable_wav(
    tmp_path: Path, xtts_settings: XttsSettings
):
    xtts_settings.reference_wavs[0].write_bytes(b"not a wav file")

    with pytest.raises(ValueError, match="readable WAV"):
        XttsEngine(xtts_settings, backend=FakeBackend()).load()


@pytest.mark.parametrize(
    ("waveform", "message"),
    [
        (np.array([0.0, np.nan], dtype=np.float32), "finite"),
        (np.array([0.0, np.inf], dtype=np.float32), "finite"),
        (np.ones(24_000, dtype=np.float32), "clipping"),
        (np.zeros(2_399, dtype=np.float32), "100 ms"),
        (np.zeros((24_000, 2), dtype=np.float32), "mono"),
    ],
)
def test_xtts_engine_rejects_invalid_backend_audio(
    tmp_path: Path,
    xtts_settings: XttsSettings,
    waveform: np.ndarray,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        XttsEngine(xtts_settings, backend=FakeBackend(waveform)).synthesize(
            "Merhaba dünya.", tmp_path / "raw.wav"
        )


def test_xtts_engine_rejects_unexpected_sample_rate(
    tmp_path: Path, xtts_settings: XttsSettings
):
    with pytest.raises(ValueError, match="24000 Hz"):
        XttsEngine(xtts_settings, backend=FakeBackend(sample_rate=22_050)).synthesize(
            "Merhaba dünya.", tmp_path / "raw.wav"
        )


def test_xtts_engine_rejects_pcm16_clipping_after_wav_readback(
    tmp_path: Path, xtts_settings: XttsSettings
):
    output = tmp_path / "raw.wav"
    waveform = np.full(24_000, -0.99999, dtype=np.float32)

    with pytest.raises(ValueError, match="clipping"):
        XttsEngine(xtts_settings, backend=FakeBackend(waveform)).synthesize(
            "Merhaba dünya.", output
        )

    assert not output.exists()
    assert not output.with_name("raw.wav.partial").exists()


def test_xtts_engine_preserves_existing_output_when_pcm16_validation_fails(
    tmp_path: Path, xtts_settings: XttsSettings
):
    output = tmp_path / "raw.wav"
    sf.write(output, np.zeros(24_000, dtype=np.float32), 24_000, subtype="PCM_16")
    original_bytes = output.read_bytes()
    waveform = np.full(24_000, -0.99999, dtype=np.float32)

    with pytest.raises(ValueError, match="clipping"):
        XttsEngine(xtts_settings, backend=FakeBackend(waveform)).synthesize(
            "Merhaba dünya.", output
        )

    assert not output.with_name("raw.wav.partial").exists()
    assert output.read_bytes() == original_bytes


def test_manifest_selects_only_hashed_xtts_assets_below_runtime_voice_root(tmp_path: Path):
    paths = RuntimePaths.developer(tmp_path)
    voice_root = paths.voice_root
    xtts = voice_root / "xtts"
    references = voice_root / "references"
    xtts.mkdir(parents=True)
    references.mkdir(parents=True)
    for name in ("best_model.pth", "config.json", "vocab.json"):
        (xtts / name).write_text(name, encoding="utf-8")
    reference = references / "approved.wav"
    _write_reference(reference)
    files = []
    for path in (xtts / "best_model.pth", xtts / "config.json", xtts / "vocab.json", reference):
        files.append(
            {
                "path": path.relative_to(voice_root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    for path in (
        "rvc/speaker_epoch_200.pth",
        "rvc/speaker.index",
        "rvc/models/embedders/contentvec/config.json",
        "rvc/models/embedders/contentvec/pytorch_model.bin",
        "rvc/models/predictors/rmvpe.pt",
    ):
        files.append({"path": path, "size": 1, "sha256": "d" * 64})
    manifest = VoiceArtifactManifest.model_validate(
        {
            "schema": "youtuber.voice.v1",
            "runtime_api": 1,
            "sample_rate": 16_000,
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

    settings = XttsSettings.from_runtime_paths(paths, manifest)

    assert settings.checkpoint_dir == xtts
    assert settings.reference_wavs == (reference,)
    assert settings.sample_rate == 16_000
    assert settings.audio_format == "WAV"
    artifact = XttsEngine(
        settings, backend=FakeBackend(np.zeros(16_000, dtype=np.float32), 16_000)
    ).synthesize("Merhaba dünya.", tmp_path / "manifest-rate.wav")
    assert artifact.sample_rate == 16_000


def test_manifest_rejects_reference_not_declared_as_a_hashed_file():
    payload = {
        "schema": "youtuber.voice.v1",
        "runtime_api": 1,
        "sample_rate": 24_000,
        "audio_format": "WAV",
        "files": [
            {"path": "xtts/best_model.pth", "size": 1, "sha256": "a" * 64},
            {"path": "xtts/config.json", "size": 1, "sha256": "b" * 64},
            {"path": "xtts/vocab.json", "size": 1, "sha256": "c" * 64},
            {"path": "rvc/speaker_epoch_200.pth", "size": 1, "sha256": "d" * 64},
            {"path": "rvc/speaker.index", "size": 1, "sha256": "e" * 64},
            {"path": "rvc/models/embedders/contentvec/config.json", "size": 1, "sha256": "f" * 64},
            {"path": "rvc/models/embedders/contentvec/pytorch_model.bin", "size": 1, "sha256": "0" * 64},
            {"path": "rvc/models/predictors/rmvpe.pt", "size": 1, "sha256": "1" * 64},
        ],
        "xtts": {
            "checkpoint": "xtts/best_model.pth",
            "config": "xtts/config.json",
            "vocab": "xtts/vocab.json",
            "reference_wavs": ["references/unapproved.wav"],
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

    with pytest.raises(ValidationError, match="approved file"):
        VoiceArtifactManifest.model_validate(json.loads(json.dumps(payload)))


def test_voice_manifest_rejects_windows_style_artifact_paths():
    with pytest.raises(ValidationError, match="safe relative POSIX"):
        ArtifactFile(path=r"references\\approved.wav", size=1, sha256="a" * 64)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("audio_format", "FLAC"),
        ("runtime_api", True),
        ("runtime_api", 1.0),
        ("sample_rate", True),
        ("sample_rate", 24_000.0),
        ("rvc.epoch", True),
        ("rvc.epoch", 200.0),
        ("rvc.pitch", False),
        ("rvc.pitch", 0.0),
    ],
)
def test_manifest_rejects_non_json_integer_values(field: str, value: object):
    payload = {
        "schema": "youtuber.voice.v1",
        "runtime_api": 1,
        "sample_rate": 24_000,
        "audio_format": "WAV",
        "files": [
            {"path": "xtts/best_model.pth", "size": 1, "sha256": "a" * 64},
            {"path": "xtts/config.json", "size": 1, "sha256": "b" * 64},
            {"path": "xtts/vocab.json", "size": 1, "sha256": "c" * 64},
            {"path": "references/approved.wav", "size": 1, "sha256": "d" * 64},
            {"path": "rvc/speaker_epoch_200.pth", "size": 1, "sha256": "e" * 64},
            {"path": "rvc/speaker.index", "size": 1, "sha256": "f" * 64},
            {"path": "rvc/models/embedders/contentvec/config.json", "size": 1, "sha256": "0" * 64},
            {"path": "rvc/models/embedders/contentvec/pytorch_model.bin", "size": 1, "sha256": "1" * 64},
            {"path": "rvc/models/predictors/rmvpe.pt", "size": 1, "sha256": "2" * 64},
        ],
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
    parent, _, child = field.partition(".")
    if child:
        payload[parent][child] = value
    else:
        payload[parent] = value

    with pytest.raises(ValidationError):
        VoiceArtifactManifest.model_validate(payload)


def test_manifest_defaults_serialize_schema_and_round_trip():
    manifest = VoiceArtifactManifest.model_validate(
        {
            "schema": "youtuber.voice.v1",
            "runtime_api": 1,
            "sample_rate": 24_000,
            "audio_format": "WAV",
            "files": [
                {"path": "xtts/best_model.pth", "size": 1, "sha256": "a" * 64},
                {"path": "xtts/config.json", "size": 1, "sha256": "b" * 64},
                {"path": "xtts/vocab.json", "size": 1, "sha256": "c" * 64},
                {"path": "references/approved.wav", "size": 1, "sha256": "d" * 64},
                {"path": "rvc/speaker_epoch_200.pth", "size": 1, "sha256": "e" * 64},
                {"path": "rvc/speaker.index", "size": 1, "sha256": "f" * 64},
                {"path": "rvc/models/embedders/contentvec/config.json", "size": 1, "sha256": "0" * 64},
                {"path": "rvc/models/embedders/contentvec/pytorch_model.bin", "size": 1, "sha256": "1" * 64},
                {"path": "rvc/models/predictors/rmvpe.pt", "size": 1, "sha256": "2" * 64},
            ],
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

    serialized = manifest.model_dump()
    assert serialized["schema"] == "youtuber.voice.v1"
    assert VoiceArtifactManifest.model_validate_json(manifest.model_dump_json()) == manifest
