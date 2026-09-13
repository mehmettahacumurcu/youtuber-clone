"""CLI: ``python -m pipeline.download [flags]``.

Defaults to the channel URL + max_videos from config.yaml. Use ``--url`` or
``--video-id`` to fetch a single video (handy for the reference clip).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pipeline.config import load_settings
from pipeline.download.runner import _already_have, download_channel, download_video


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline.download", description=__doc__)
    p.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--url", help="Single video URL to download")
    g.add_argument("--video-id", help="Single YouTube video id (e.g. sample00059)")
    p.add_argument("--limit", type=int, default=None, help="Override download.max_videos")
    p.add_argument("--force", action="store_true", help="Re-download even if outputs already exist")
    p.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings(args.config)

    if args.url or args.video_id:
        # Short-circuit a known video-id whose audio+meta already exist, BEFORE any network
        # call. download_video() otherwise probes YouTube first (just to learn the id) even
        # when the file is present — which re-trips the bot-gate when a processing host (e.g.
        # Colab) re-runs the download stage on audio pre-staged from Drive. With both files
        # already present there is nothing to fetch, so never touch the network.
        if args.video_id and not args.force and _already_have(args.video_id, settings):
            print(f"skip {args.video_id} (already downloaded)")
            return 0
        url = args.url or f"https://www.youtube.com/watch?v={args.video_id}"
        path = download_video(url, settings, force=args.force)
        return 0 if path is not None else 1

    channel_url = settings.download.channel_url
    if not channel_url:
        if settings.download.video_urls:
            results = [download_video(u, settings, force=args.force) for u in settings.download.video_urls]
            return 0 if any(r is not None for r in results) else 1
        print("ERROR: no --url / --video-id and no channel_url / video_urls in config.", file=sys.stderr)
        return 2

    limit = args.limit if args.limit is not None else settings.download.max_videos
    paths = download_channel(channel_url, settings, limit=limit, force=args.force)
    print(f"Downloaded {len(paths)} videos to {settings.paths.audio_dir}", file=sys.stderr)
    return 0 if paths else 1


if __name__ == "__main__":
    raise SystemExit(main())
