"""Export per-segment audio clips for kept/borderline segments before WAVs are discarded.

Reads ``data/identify/{video_id}.json`` (segment decisions) + the (ephemeral) WAV and
writes small clips to ``data/clips/{video_id}/{segment_id}.wav``. These are committed to
git so the local review UI (Stage 8) and TTS (Phase 5) have audio without the 100 GB
corpus. ``segment_id`` is positional: ``{index:04d}_{start:.2f}-{end:.2f}``.

CLI: ``python -m pipeline.clips --config config.yaml --video-id X``
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from pipeline.config import Settings, load_settings

log = logging.getLogger("pipeline.clips")

EXPORT_DECISIONS = ("keep", "borderline")


def clips_dir_for(video_id: str, settings: Settings) -> Path:
    return settings.paths.data_dir / "clips" / video_id


def segment_id(index: int, start: float, end: float) -> str:
    return f"{index:04d}_{start:.2f}-{end:.2f}"


def export_clips(
    video_id: str,
    settings: Settings,
    *,
    decisions: tuple[str, ...] = EXPORT_DECISIONS,
    force: bool = False,
) -> list[Path]:
    """Write clips for segments whose decision is in `decisions`. Returns clip paths."""
    import numpy as np
    import soundfile as sf

    # Imported lazily to keep `segment_id`/`clips_dir_for` importable without the chain.
    from pipeline.download.runner import audio_path_for
    from pipeline.identify.runner import identify_path_for

    idn_path = identify_path_for(video_id, settings)
    if not idn_path.exists():
        raise FileNotFoundError(f"identify output not found: {idn_path}")
    audio_path = audio_path_for(video_id, settings)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    sr = settings.download.sample_rate_hz
    samples, file_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if file_sr != sr:
        raise ValueError(f"{audio_path} sample rate {file_sr} != expected {sr}")
    if samples.ndim > 1:
        samples = samples.mean(axis=1).astype(np.float32, copy=False)

    doc = json.loads(idn_path.read_text(encoding="utf-8"))
    out_dir = clips_dir_for(video_id, settings)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for i, seg in enumerate(doc.get("segments", [])):
        if seg.get("decision") not in decisions:
            continue
        start = float(seg["start"])
        end = float(seg["end"])
        dst = out_dir / f"{segment_id(i, start, end)}.wav"
        if dst.exists() and not force:
            written.append(dst)
            continue
        a = max(0, int(start * sr))
        b = min(len(samples), int(end * sr))
        if b <= a:
            continue
        sf.write(str(dst), samples[a:b], sr, subtype="PCM_16")
        written.append(dst)

    log.info("wrote %d clips for %s -> %s", len(written), video_id, out_dir)
    return written


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline.clips", description=__doc__)
    p.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    p.add_argument("--video-id", required=True, help="Video id to export clips for")
    p.add_argument("--force", action="store_true", help="Overwrite existing clips")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings(args.config)
    written = export_clips(args.video_id, settings, force=args.force)
    print(f"Exported {len(written)} clips for {args.video_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
