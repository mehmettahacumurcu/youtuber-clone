"""Silero VAD over a stage-1 WAV. Emits per-video speech-region JSON.

Output schema (``data/vad/{video_id}.json``)::

    {
      "video_id": "sample00059",
      "sample_rate_hz": 16000,
      "duration_s": 2772.2,
      "config": {"threshold": 0.5, "min_speech_ms": 250, ...},
      "regions": [{"start": 0.5, "end": 3.2}, ...],   # seconds
      "total_speech_s": 1834.7,
      "speech_fraction": 0.66,
      "ran_at_utc": "2026-05-22T07:00:00+00:00"
    }

Silero VAD is tiny (CPU-fine); we deliberately don't use the GPU here.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import Settings
from pipeline.download import audio_path_for, meta_path_for

log = logging.getLogger("pipeline.vad")


def vad_path_for(video_id: str, settings: Settings) -> Path:
    return settings.paths.vad_dir / f"{video_id}.json"


def run_vad(video_id: str, settings: Settings, *, force: bool = False) -> Path:
    """Run Silero VAD on ``data/audio/{video_id}.wav`` and write JSON. Returns the JSON path."""
    audio = audio_path_for(video_id, settings)
    if not audio.exists():
        raise FileNotFoundError(f"audio not found: {audio} (run pipeline.download first)")

    out = vad_path_for(video_id, settings)
    if out.exists() and not force:
        log.info("skip VAD for %s (already exists at %s)", video_id, out)
        return out

    # Lazy imports — keeps `pipeline.config` cheap to import and avoids loading torch unnecessarily.
    import numpy as np
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    sr = settings.download.sample_rate_hz
    log.info("reading audio %s", audio)
    samples, file_sr = sf.read(str(audio), dtype="float32", always_2d=False)
    if file_sr != sr:
        raise ValueError(
            f"audio sample rate mismatch: file is {file_sr} Hz, expected {sr} Hz "
            f"(re-run pipeline.download to regenerate {audio})"
        )
    if samples.ndim > 1:
        # downmix to mono just in case
        samples = samples.mean(axis=1).astype(np.float32, copy=False)
    wav = torch.from_numpy(samples)
    duration_s = float(wav.shape[-1]) / sr

    log.info("loading Silero VAD model")
    model = load_silero_vad()

    cfg = settings.vad
    log.info(
        "running VAD on %.1fs of audio (threshold=%.2f min_speech_ms=%d min_silence_ms=%d pad_ms=%d)",
        duration_s,
        cfg.threshold,
        cfg.min_speech_ms,
        cfg.min_silence_ms,
        cfg.speech_pad_ms,
    )
    raw = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sr,
        threshold=cfg.threshold,
        min_speech_duration_ms=cfg.min_speech_ms,
        min_silence_duration_ms=cfg.min_silence_ms,
        speech_pad_ms=cfg.speech_pad_ms,
        return_seconds=True,
    )
    regions = [{"start": float(r["start"]), "end": float(r["end"])} for r in raw]
    total = sum(r["end"] - r["start"] for r in regions)

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_id": video_id,
        "sample_rate_hz": sr,
        "duration_s": round(duration_s, 3),
        "config": {
            "threshold": cfg.threshold,
            "min_speech_ms": cfg.min_speech_ms,
            "min_silence_ms": cfg.min_silence_ms,
            "speech_pad_ms": cfg.speech_pad_ms,
        },
        "regions": regions,
        "total_speech_s": round(total, 3),
        "speech_fraction": round(total / duration_s, 4) if duration_s else 0.0,
        "ran_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(
        "wrote %s — %d regions, %.1fs speech (%.1f%% of duration)",
        out,
        len(regions),
        total,
        100.0 * total / duration_s if duration_s else 0.0,
    )
    return out


def discover_video_ids(settings: Settings) -> list[str]:
    """Return all video ids that have an audio file in data/audio/."""
    return sorted(p.stem for p in settings.paths.audio_dir.glob("*.wav"))
