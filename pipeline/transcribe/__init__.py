"""Stage 5 — Whisper transcription (faster-whisper, Turkish forced).

Transcribes only the segments that Stage 4 tagged ``decision=keep``, since the
goal is his-only training data. Output: ``data/transcribe/{video_id}.json``.

Public API:
    run_transcribe(video_id, settings) -> Path
    transcribe_path_for(video_id, settings) -> Path
"""
from pipeline.transcribe.runner import run_transcribe, transcribe_path_for

__all__ = ["run_transcribe", "transcribe_path_for"]
