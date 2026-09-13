"""Stage 2 — Silero VAD on 16 kHz mono WAVs.

Public API:
    run_vad(video_id, settings) -> Path  # writes data/vad/{video_id}.json
    vad_path_for(video_id, settings) -> Path
"""
from pipeline.vad.runner import run_vad, vad_path_for

__all__ = ["run_vad", "vad_path_for"]
