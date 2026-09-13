"""CLI: ``python -m pipeline.assemble [--config config.yaml]`` — build llm_corpus.jsonl."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pipeline.config import load_settings
from pipeline.assemble.runner import discover_video_ids, run_assemble


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline.assemble", description=__doc__)
    p.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    p.add_argument("--source", choices=["clean", "filter"], default="clean",
                   help="clean = ASR-corrected canonical corpus (llm_corpus.jsonl); "
                        "filter = RAW corpus (llm_corpus.raw.jsonl)")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = load_settings(args.config)
    if not discover_video_ids(settings):
        print("ERROR: no filter JSONs to assemble.", file=sys.stderr)
        return 2
    run_assemble(settings, source=args.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
