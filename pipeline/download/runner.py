"""yt-dlp wrapper that downloads channel/video audio as 16 kHz mono WAV.

Each successful download writes two files:
    - ``data/audio/{video_id}.wav``  — 16 kHz mono PCM
    - ``data/meta/{video_id}.json``  — slim metadata captured at download time

The presence of BOTH files is the idempotency signal: subsequent runs skip
already-downloaded videos unless ``force=True``.
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from yt_dlp import YoutubeDL

from pipeline.config import Settings

log = logging.getLogger("pipeline.download")

# Where to look for a yt-dlp cookies.txt when download.cookies_file is unset.
# "/content/cookies.txt" is where Colab's "Upload to Colab" drops the file; the relative
# entries cover local/cwd runs. YouTube bot-gates downloads from datacenter IPs (Colab),
# so authenticated cookies are required there; a home IP usually needs none.
_DEFAULT_COOKIE_LOCATIONS = ("/content/cookies.txt", "cookies.txt", "data/cookies.txt")


def _resolve_cookiefile(explicit: str | None) -> str | None:
    """First existing cookies file: the configured path, else a known default location.

    Returns None if none exist — yt-dlp then runs without cookies (fine for IPs that
    aren't bot-gated).
    """
    for candidate in ([explicit] if explicit else []) + list(_DEFAULT_COOKIE_LOCATIONS):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


# YouTube now obfuscates stream URLs behind an "n challenge" that must be solved by
# running its JavaScript. yt-dlp can do this only if (a) a JS runtime like Deno is on
# PATH and (b) it is allowed to fetch the solver scripts from the official yt-dlp/ejs
# GitHub release — this opt grants (b). Without it yt-dlp sees only image formats and
# the download fails with "Requested format is not available". Harmless off YouTube.
_EJS_REMOTE_COMPONENTS = ["ejs:github"]


def audio_path_for(video_id: str, settings: Settings) -> Path:
    return settings.paths.audio_dir / f"{video_id}.wav"


def meta_path_for(video_id: str, settings: Settings) -> Path:
    return settings.paths.meta_dir / f"{video_id}.json"


def _ydl_opts(settings: Settings, *, no_playlist: bool, out_dir: Path) -> dict[str, Any]:
    """yt-dlp downloads bestaudio + extracts to native-rate WAV in a temp dir.

    We then ffmpeg-resample to 16 kHz mono ourselves — keeps the yt-dlp call simple
    and avoids version-fragile postprocessor_args key syntax.
    """
    opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": no_playlist,
        "remote_components": list(_EJS_REMOTE_COMPONENTS),  # solve YouTube's n-challenge
        "ignoreerrors": True,
        "retries": 5,
        "fragment_retries": 5,
        "concurrent_fragment_downloads": 4,
        "quiet": False,
        "no_warnings": False,
        "writeinfojson": False,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": settings.download.audio_format,  # wav
                "preferredquality": settings.download.audio_quality,
            }
        ],
    }
    cookiefile = _resolve_cookiefile(settings.download.cookies_file)
    if cookiefile:
        opts["cookiefile"] = cookiefile
    # Throttle to dodge bot-gating: a randomized pause before each download (and between
    # extraction requests). Only set when configured > 0, so default behavior is unchanged.
    si = settings.download.sleep_interval_s
    if si and si > 0:
        msi = settings.download.max_sleep_interval_s
        opts["sleep_interval"] = si
        opts["max_sleep_interval"] = max(msi, si) if msi else si
        opts["sleep_interval_requests"] = 1
    return opts


def _resample_to_target(src: Path, dst: Path, sample_rate_hz: int, channels: int) -> None:
    """ffmpeg-resample src -> dst (16 kHz mono PCM s16le by default).

    Raises subprocess.CalledProcessError if ffmpeg fails; caller decides what to do.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    # -y overwrite; -hide_banner cleaner log; pcm_s16le is the canonical WAV format for downstream tools.
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(src),
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate_hz),
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def _write_meta(info: dict[str, Any], settings: Settings) -> Path:
    video_id = info["id"]
    audio = audio_path_for(video_id, settings)
    meta = {
        "id": video_id,
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "uploader_id": info.get("uploader_id"),
        "channel": info.get("channel"),
        "channel_url": info.get("channel_url"),
        "webpage_url": info.get("webpage_url"),
        "duration_s": info.get("duration"),
        "upload_date": info.get("upload_date"),  # YYYYMMDD
        "view_count": info.get("view_count"),
        "language": info.get("language"),
        "extractor": info.get("extractor"),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "audio_path": str(audio.relative_to(settings.paths.data_dir.parent)).replace("\\", "/"),
        "sample_rate_hz": settings.download.sample_rate_hz,
        "channels": settings.download.channels,
    }
    out = meta_path_for(video_id, settings)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _already_have(video_id: str, settings: Settings) -> bool:
    return audio_path_for(video_id, settings).exists() and meta_path_for(video_id, settings).exists()


def _ensure_dirs(settings: Settings) -> None:
    settings.paths.audio_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.meta_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.logs_dir.mkdir(parents=True, exist_ok=True)


def download_video(url: str, settings: Settings, *, force: bool = False) -> Path | None:
    """Download a single video URL. Returns the final audio Path, or None on failure."""
    _ensure_dirs(settings)

    # Probe (no download) to learn the id, for idempotency.
    probe_opts: dict[str, Any] = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "remote_components": list(_EJS_REMOTE_COMPONENTS),  # solve YouTube's n-challenge
        "ignoreerrors": True,
    }
    cookiefile = _resolve_cookiefile(settings.download.cookies_file)
    if cookiefile:
        probe_opts["cookiefile"] = cookiefile
    with YoutubeDL(probe_opts) as ydl:
        try:
            probe = ydl.extract_info(url, download=False, process=False)
        except Exception as exc:  # noqa: BLE001
            log.error("probe failed for %s: %s", url, exc)
            return None
    if probe is None:
        log.error("probe returned no info for %s", url)
        return None
    video_id = probe.get("id")
    if not video_id:
        log.error("probe missing id for %s", url)
        return None

    if not force and _already_have(video_id, settings):
        log.info("skip %s (already downloaded)", video_id)
        return audio_path_for(video_id, settings)

    # Download to a temp dir, then ffmpeg-resample into the canonical path.
    with tempfile.TemporaryDirectory(prefix=f"ytdl_{video_id}_") as tmp:
        tmp_dir = Path(tmp)
        opts = _ydl_opts(settings, no_playlist=True, out_dir=tmp_dir)
        with YoutubeDL(opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
            except Exception as exc:  # noqa: BLE001
                log.error("download failed for %s: %s", url, exc)
                return None
        if info is None:
            return None

        # yt-dlp's FFmpegExtractAudio produces {id}.{audio_format} in tmp_dir.
        src = tmp_dir / f"{video_id}.{settings.download.audio_format}"
        if not src.exists():
            # Fallback: any audio file matching the id (some PPs use a different ext).
            cands = list(tmp_dir.glob(f"{video_id}.*"))
            if not cands:
                log.error("no audio file produced for %s in %s", video_id, tmp_dir)
                return None
            src = cands[0]
        dst = audio_path_for(video_id, settings)
        try:
            _resample_to_target(
                src,
                dst,
                sample_rate_hz=settings.download.sample_rate_hz,
                channels=settings.download.channels,
            )
        except subprocess.CalledProcessError as exc:
            log.error("ffmpeg resample failed for %s: %s", video_id, exc)
            return None

    _write_meta(info, settings)
    log.info("downloaded %s -> %s", video_id, dst)
    return dst


def _enumerate_channel(channel_url: str, limit: int, cookies_file: str | None = None) -> list[str]:
    """Return up to `limit` video URLs from a channel/playlist, newest first (yt-dlp default)."""
    flat_opts: dict[str, Any] = {
        "extract_flat": "in_playlist",
        "skip_download": True,
        "playlistend": limit,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }
    cookiefile = _resolve_cookiefile(cookies_file)
    if cookiefile:
        flat_opts["cookiefile"] = cookiefile
    with YoutubeDL(flat_opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    if info is None:
        return []
    entries = info.get("entries") or []
    urls: list[str] = []
    for e in entries:
        if not e:
            continue
        u = e.get("url") or e.get("webpage_url")
        if not u:
            continue
        # extract_flat sometimes gives bare ids — promote to full URL
        if not u.startswith("http"):
            u = f"https://www.youtube.com/watch?v={u}"
        urls.append(u)
        if len(urls) >= limit:
            break
    return urls


def _id_from_url(url: str) -> str | None:
    """Extract the YouTube video id from a watch URL, or None if not present."""
    q = parse_qs(urlparse(url).query)
    vals = q.get("v")
    return vals[0] if vals else None


def enumerate_channel_ids(channel_url: str, limit: int, cookies_file: str | None = None) -> list[str]:
    """Return up to `limit` video ids from a channel/playlist (newest first)."""
    ids: list[str] = []
    for u in _enumerate_channel(channel_url, limit, cookies_file):
        vid = _id_from_url(u)
        if vid:
            ids.append(vid)
    return ids


def download_channel(
    channel_url: str,
    settings: Settings,
    *,
    limit: int,
    force: bool = False,
) -> list[Path]:
    """Download up to `limit` videos from the channel. Returns audio paths of successful pulls."""
    _ensure_dirs(settings)
    urls = _enumerate_channel(channel_url, limit, settings.download.cookies_file)
    log.info("enumerated %d video URLs from %s", len(urls), channel_url)
    out: list[Path] = []
    for u in urls:
        p = download_video(u, settings, force=force)
        if p is not None:
            out.append(p)
    return out
