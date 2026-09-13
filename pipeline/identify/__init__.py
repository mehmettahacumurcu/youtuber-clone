"""Stage 4 — speaker identification: cosine similarity to the reference clip.

For each diarization segment, embed the audio slice with ECAPA-TDNN and compare
to the precomputed reference embedding. Tag each segment with a keep/borderline/
drop decision based on configured thresholds.

Public API:
    run_identify(video_id, settings) -> Path  # writes data/identify/{video_id}.json
    identify_path_for(video_id, settings) -> Path
"""
from pipeline.identify.runner import identify_path_for, run_identify

__all__ = ["identify_path_for", "run_identify"]
