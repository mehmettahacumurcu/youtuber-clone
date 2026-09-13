"""Stage 3 — speaker diarization via pyannote.audio.

Public API:
    run_diarization(video_id, settings) -> Path  # writes data/diarize/{video_id}.json
    diarize_path_for(video_id, settings) -> Path
"""
from pipeline.diarize.runner import diarize_path_for, run_diarization

__all__ = ["diarize_path_for", "run_diarization"]
