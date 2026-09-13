"""Build/maintain ``data/manifest.json`` — the full list of target video ids.

The manifest is the denominator for global progress (``pipeline.status``) and the
work-list for the Colab batch driver (``colab/run_batch.py``). It is committed to git
so progress is known without the audio present.

CLI: ``python -m pipeline.manifest --config config.yaml [--limit N]``
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import Settings, load_settings

log = logging.getLogger("pipeline.manifest")


def manifest_path_for(settings: Settings) -> Path:
    return settings.paths.data_dir / "manifest.json"


def load_manifest(settings: Settings) -> list[str]:
    p = manifest_path_for(settings)
    if not p.exists():
        return []
    doc = json.loads(p.read_text(encoding="utf-8"))
    return [str(v) for v in doc.get("video_ids", [])]


def save_manifest(video_ids: list[str], settings: Settings, *, generated_at: str) -> Path:
    seen: set[str] = set()
    ordered: list[str] = []
    for v in video_ids:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    p = manifest_path_for(settings)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {"generated_at_utc": generated_at, "count": len(ordered), "video_ids": ordered}
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def pending_ids(settings: Settings, output_dir: Path, *, suffix: str = ".json") -> list[str]:
    """Manifest ids lacking an output file (``<id><suffix>``) in ``output_dir``."""
    done = {p.stem for p in output_dir.glob(f"*{suffix}")}
    return [v for v in load_manifest(settings) if v not in done]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline.manifest", description=__doc__)
    p.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    p.add_argument("--limit", type=int, default=None, help="Max videos to enumerate (default: config max_videos)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings(args.config)
    channel_url = settings.download.channel_url
    if not channel_url:
        print("ERROR: download.channel_url not set in config.", flush=True)
        return 2
    # Imported here so the pure helpers above stay importable without yt-dlp at module load.
    from pipeline.download.runner import enumerate_channel_ids

    limit = args.limit if args.limit is not None else settings.download.max_videos
    ids = enumerate_channel_ids(channel_url, limit, settings.download.cookies_file)
    out = save_manifest(
        ids, settings, generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    print(f"Wrote {out} with {len(ids)} video ids", flush=True)
    return 0 if ids else 1


if __name__ == "__main__":
    raise SystemExit(main())
