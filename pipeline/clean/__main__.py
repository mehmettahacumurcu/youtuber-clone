"""CLI: ``python -m pipeline.clean --config config.yaml (--video-id X | --all) [--force]``."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pipeline.config import load_settings
from pipeline.clean.runner import discover_video_ids, run_clean


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline.clean", description=__doc__)
    p.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video-id", help="Run on this id only")
    g.add_argument("--all", action="store_true", help="Run on every video with a filter JSON")
    p.add_argument("--force", action="store_true", help="Re-run even if JSON already exists")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings(args.config)
    ids = [args.video_id] if args.video_id else discover_video_ids(settings)
    if not ids:
        print("ERROR: no filter JSONs to clean.", file=sys.stderr)
        return 2
    failures = 0
    for vid in ids:
        try:
            run_clean(vid, settings, force=args.force)
        except Exception as exc:  # noqa: BLE001
            logging.exception("clean failed for %s: %s", vid, exc)
            failures += 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
