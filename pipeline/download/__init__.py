"""Stage 1 — download audio from YouTube via yt-dlp.

Public API:
    download_video(url, settings, force=False) -> Path
    download_channel(channel_url, settings, limit, force=False) -> list[Path]
    meta_path_for(video_id, settings) -> Path
    audio_path_for(video_id, settings) -> Path

CLI entry point is :mod:`pipeline.download.__main__`.
"""
from pipeline.download.runner import (
    audio_path_for,
    download_channel,
    download_video,
    meta_path_for,
)

__all__ = [
    "audio_path_for",
    "download_channel",
    "download_video",
    "meta_path_for",
]
