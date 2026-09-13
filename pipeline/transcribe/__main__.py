"""CLI: ``python -m pipeline.transcribe [--video-id X | --all]``."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pipeline.config import load_settings
from pipeline.transcribe.runner import discover_video_ids, run_transcribe


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline.transcribe", description=__doc__)
    p.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video-id", help="Run on this id only")
    g.add_argument("--all", action="store_true", help="Run on every video with an identify JSON")
    p.add_argument("--force", action="store_true", help="Re-run even if JSON already exists")
    p.add_argument("--device", default=None, help="Force cpu/cuda")
    p.add_argument("--compute-type", default=None, help="float16/int8_float16/int8")
    p.add_argument("--beam-size", type=int, default=None)
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
        print("ERROR: no identify JSONs to transcribe.", file=sys.stderr)
        return 2

    failures = 0
    for vid in ids:
        try:
            run_transcribe(
                vid, settings,
                force=args.force,
                device=args.device,
                compute_type=args.compute_type,
                beam_size=args.beam_size,
            )
        except Exception as exc:  # noqa: BLE001
            logging.exception("transcribe failed for %s: %s", vid, exc)
            failures += 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
