"""Lazy, validated XTTS synthesis with a small injectable backend boundary."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np
import soundfile as sf

from voice.models import AudioArtifact, XttsSettings


class XttsBackend(Protocol):
    def load(self, settings: XttsSettings) -> None: ...

    def synthesize(
        self, text: str, language: str, reference_wavs: tuple[Path, ...]
    ) -> tuple[np.ndarray, int]: ...

    def unload(self) -> None: ...


class CoquiXttsBackend:
    """The real backend; heavyweight Coqui/Torch imports occur only in ``load``."""

    def __init__(self) -> None:
        self._model: object | None = None
        self._gpt_latent: object | None = None
        self._speaker_embedding: object | None = None

    def load(self, settings: XttsSettings) -> None:
        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        config = XttsConfig()
        config.load_json(str(settings.config_file))
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_path=str(settings.checkpoint_file),
            vocab_path=str(settings.vocab_file),
            use_deepspeed=False,
        )
        if torch.cuda.is_available():
            model.cuda()
        self._gpt_latent, self._speaker_embedding = model.get_conditioning_latents(
            audio_path=[str(path) for path in settings.reference_wavs],
            gpt_cond_len=30,
            max_ref_length=30,
            sound_norm_refs=True,
        )
        self._model = model

    def synthesize(
        self, text: str, language: str, reference_wavs: tuple[Path, ...]
    ) -> tuple[np.ndarray, int]:
        if self._model is None or self._gpt_latent is None or self._speaker_embedding is None:
            raise RuntimeError("XTTS backend is not loaded")
        result = self._model.inference(
            text,
            language,
            self._gpt_latent,
            self._speaker_embedding,
            temperature=0.7,
            repetition_penalty=1.3,
            length_penalty=1.0,
        )
        return np.asarray(result["wav"], dtype=np.float32), 24_000

    def unload(self) -> None:
        self._model = None
        self._gpt_latent = None
        self._speaker_embedding = None


class XttsEngine:
    """Lifecycle-managed XTTS engine that rejects invalid model and audio artifacts."""

    def __init__(self, settings: XttsSettings, backend: XttsBackend | None = None) -> None:
        self.settings = settings
        self._backend = backend or CoquiXttsBackend()
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        required = (
            self.settings.checkpoint_file,
            self.settings.config_file,
            self.settings.vocab_file,
        )
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        for path in self.settings.reference_wavs:
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.suffix.lower() != ".wav":
                raise ValueError("reference paths must be WAV files")
            try:
                info = sf.info(path)
            except RuntimeError as error:
                raise ValueError(f"reference is not a readable WAV: {path}") from error
            if info.format != "WAV" or info.channels != 1 or info.frames == 0:
                raise ValueError(f"reference is not a readable WAV: {path}")
        self._backend.load(self.settings)
        self._loaded = True

    def unload(self) -> None:
        if not self._loaded:
            return
        try:
            self._backend.unload()
        finally:
            self._loaded = False

    def synthesize(self, text: str, output: Path) -> AudioArtifact:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must not be blank")
        self.load()
        samples, sample_rate = self._backend.synthesize(
            text, self.settings.language, self.settings.reference_wavs
        )
        waveform = np.asarray(samples, dtype=np.float32)
        self._validate_waveform(waveform, sample_rate)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".partial")
        temporary.unlink(missing_ok=True)
        try:
            sf.write(temporary, waveform, sample_rate, subtype="PCM_16", format="WAV")
            info = sf.info(temporary)
            if info.channels != 1 or info.samplerate != sample_rate or info.frames != waveform.size:
                raise ValueError("generated WAV readback is not mono at the expected sample rate")
            readback, readback_rate = sf.read(
                temporary, dtype="float32", always_2d=False
            )
            self._validate_waveform(np.asarray(readback, dtype=np.float32), readback_rate)
            temporary.replace(output)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return AudioArtifact(
            path=output,
            sample_rate=sample_rate,
            duration_seconds=waveform.size / sample_rate,
            sha256=self._sha256_file(output),
        )

    def _validate_waveform(self, waveform: np.ndarray, sample_rate: int) -> None:
        if sample_rate != self.settings.sample_rate:
            raise ValueError(
                f"generated audio must be {self.settings.sample_rate} Hz, got {sample_rate} Hz"
            )
        if waveform.ndim != 1:
            raise ValueError("generated audio must be mono")
        if waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError("generated audio must contain finite mono samples")
        if waveform.size / sample_rate < 0.1:
            raise ValueError("generated audio must be at least 100 ms")
        clipped_fraction = float(np.count_nonzero(np.abs(waveform) >= 1.0)) / waveform.size
        if clipped_fraction > self.settings.clipping_guard:
            raise ValueError("generated audio exceeds clipping guard")

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
