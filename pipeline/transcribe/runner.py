"""faster-whisper Large-v3 over Stage-4 ``keep`` segments only.

We intentionally do NOT transcribe the entire audio file — only segments that
Stage 4 classified as him. That cuts compute roughly in half on this corpus
and avoids burning GPU on non-his speech that gets dropped downstream anyway.

For each kept segment we extract the audio slice in memory and call
``faster_whisper.WhisperModel.transcribe`` with internal VAD disabled (we've
already gated upstream). The sub-segments faster-whisper emits within each
slice are concatenated into a single transcript string and stored alongside
the original segment metadata.

Schema (``data/transcribe/{video_id}.json``)::

    {
      "video_id": "...",
      "model": "large-v3",
      "language": "tr",
      "device": "cuda",
      "compute_type": "float16",
      "beam_size": 5,
      "input_segments": 1681,                  # how many keep segments existed
      "transcribed_segments": 1620,
      "skipped_too_short": 50,
      "errors": 11,
      "segments": [
        {
          "start": 12.5, "end": 18.7, "speaker": "SPEAKER_03",
          "similarity": 0.81, "duration_s": 6.2,
          "text": "...",
          "avg_logprob": -0.34,
          "no_speech_prob": 0.01,
          "language_prob": 0.99
        },
        ...
      ],
      "elapsed_s": ...,
      "ran_at_utc": "..."
    }

Speed: ~5-7x real-time on RTX 3060 Ti at float16. For our calibration corpus
(~175 min of kept speech) total runtime is ~30 min.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from pipeline.config import Settings
from pipeline.download import audio_path_for
from pipeline.identify import identify_path_for
from pipeline.identify.runner import _read_mono_16k

log = logging.getLogger("pipeline.transcribe")

MIN_TRANSCRIBE_S = 2.0  # CLAUDE.md filter.min_segment_s for LLM use


def transcribe_path_for(video_id: str, settings: Settings) -> Path:
    return settings.paths.transcribe_dir / f"{video_id}.json"


@lru_cache(maxsize=1)
def _load_whisper_model(model_size: str, device: str, compute_type: str):
    """Cache the faster-whisper model across calls in one process."""
    from faster_whisper import WhisperModel

    log.info("loading faster-whisper %s on %s (%s)", model_size, device, compute_type)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return model


def _resolve_device_compute(requested_device: str | None, requested_compute: str | None) -> tuple[str, str]:
    import torch

    if requested_device:
        dev = requested_device
    else:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_compute:
        comp = requested_compute
    else:
        comp = "float16" if dev == "cuda" else "int8"
    return dev, comp


def _kept_segments(identify_doc: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in identify_doc["segments"] if s.get("decision") == "keep"]


def run_transcribe(
    video_id: str,
    settings: Settings,
    *,
    force: bool = False,
    device: str | None = None,
    compute_type: str | None = None,
    beam_size: int | None = None,
) -> Path:
    """Transcribe all ``keep`` segments of ``video_id``. Returns the JSON path."""
    idn = identify_path_for(video_id, settings)
    if not idn.exists():
        raise FileNotFoundError(
            f"identify output not found: {idn} (run pipeline.identify first)"
        )
    audio = audio_path_for(video_id, settings)
    if not audio.exists():
        raise FileNotFoundError(f"audio not found: {audio}")

    out = transcribe_path_for(video_id, settings)
    if out.exists() and not force:
        log.info("skip transcribe for %s (already at %s)", video_id, out)
        return out

    cfg = settings.transcribe
    dev, comp = _resolve_device_compute(device, compute_type)
    bs = beam_size if beam_size is not None else cfg.batch_size

    idn_doc = json.loads(idn.read_text(encoding="utf-8"))
    kept = _kept_segments(idn_doc)
    log.info("video %s — %d keep segments to transcribe", video_id, len(kept))
    if not kept:
        log.warning("no keep segments for %s — writing empty output", video_id)

    sr = settings.download.sample_rate_hz
    log.info("reading audio %s", audio)
    audio_arr = _read_mono_16k(audio, sr)

    model = _load_whisper_model(cfg.model, dev, comp)

    t0 = datetime.now(timezone.utc)
    out_segments: list[dict[str, Any]] = []
    too_short = 0
    errors = 0
    transcribed = 0
    last_log = 0
    for i, seg in enumerate(kept):
        dur = seg["end"] - seg["start"]
        if dur < MIN_TRANSCRIBE_S:
            too_short += 1
            continue
        start_idx = max(0, int(seg["start"] * sr))
        end_idx = min(len(audio_arr), int(seg["end"] * sr))
        clip = audio_arr[start_idx:end_idx]
        try:
            sub_segments, info = model.transcribe(
                clip,
                language=cfg.language,
                beam_size=bs,
                vad_filter=False,  # already segmented upstream
                word_timestamps=False,
                condition_on_previous_text=False,
            )
            sub_list = list(sub_segments)  # generator -> list
            text = " ".join(s.text.strip() for s in sub_list if s.text).strip()
            avg_logprob = (
                sum(s.avg_logprob for s in sub_list) / len(sub_list) if sub_list else 0.0
            )
            no_speech_prob = max((s.no_speech_prob for s in sub_list), default=1.0)
            rec = {
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg.get("speaker"),
                "similarity": seg.get("similarity"),
                "duration_s": round(dur, 3),
                "text": text,
                "avg_logprob": round(avg_logprob, 4),
                "no_speech_prob": round(no_speech_prob, 4),
                "language_prob": round(info.language_probability, 4),
            }
            out_segments.append(rec)
            transcribed += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("transcribe failed for seg %d (%.2f-%.2f): %s", i, seg["start"], seg["end"], exc)
            errors += 1

        if (i + 1) - last_log >= 100 or (i + 1) == len(kept):
            elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
            done_s = sum(s["duration_s"] for s in out_segments)
            ratio = done_s / elapsed if elapsed else 0
            log.info(
                "  ... %d / %d segments  (%.1fs audio in %.1fs, %.1fx real-time)",
                i + 1, len(kept), done_s, elapsed, ratio,
            )
            last_log = i + 1

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()

    payload = {
        "video_id": video_id,
        "model": cfg.model,
        "language": cfg.language,
        "device": dev,
        "compute_type": comp,
        "beam_size": bs,
        "input_segments": len(kept),
        "transcribed_segments": transcribed,
        "skipped_too_short": too_short,
        "errors": errors,
        "segments": out_segments,
        "elapsed_s": round(elapsed, 1),
        "ran_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    total_chars = sum(len(s["text"]) for s in out_segments)
    log.info(
        "wrote %s — %d segments, %d chars, %.1fs elapsed",
        out, len(out_segments), total_chars, elapsed,
    )
    return out


def discover_video_ids(settings: Settings) -> list[str]:
    """Videos that have an identify JSON."""
    return sorted(p.stem for p in settings.paths.identify_dir.glob("*.json"))
