"""Colab batch driver — run the next N un-transcribed videos through Stages 1-6 (download…filter).

Run on the Colab runtime after cloning the repo and generating the manifest:
    python colab/run_batch.py --config config.yaml --n 10 [--dry-run]

Each video is run through the existing per-stage CLIs, which are idempotent: finished
sub-steps skip. Raw WAVs land on the ephemeral VM; only the small JSON outputs persist
(pushed to git by the notebook). A stage failure logs and skips the rest of that video.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Ensure the repo root is importable when run as a script (python colab/run_batch.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import load_settings  # noqa: E402
from pipeline.manifest import load_manifest  # noqa: E402

STAGES = ("download", "vad", "diarize", "identify", "transcribe", "filter")


def select_batch(manifest_ids: list[str], filter_dir: Path, n: int) -> list[str]:
    """First `n` manifest ids without a filter output (the batch's terminal stage)."""
    done = {p.stem for p in filter_dir.glob("*.json")} if filter_dir.exists() else set()
    pending = [v for v in manifest_ids if v not in done]
    return pending[:n]


def commands_for(video_id: str, config_path: str | Path | None = None) -> list[list[str]]:
    """Build the per-stage CLI commands for one video.

    Uses ``sys.executable`` (the interpreter running this driver) rather than a bare
    ``"python"`` so each spawned stage runs under the same environment/deps as the driver.
    """
    cfg = ["--config", str(config_path)] if config_path else []
    return [[sys.executable, "-m", f"pipeline.{stage}", *cfg, "--video-id", video_id] for stage in STAGES]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_batch", description=__doc__)
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--n", type=int, default=10, help="How many videos this batch")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = load_settings(args.config)
    batch = select_batch(load_manifest(settings), settings.paths.filter_dir, args.n)
    print(f"Selected {len(batch)} videos: {batch}", flush=True)
    for vid in batch:
        print(f"\n=== {vid} ===", flush=True)
        for cmd in commands_for(vid, args.config):
            print("  $ " + " ".join(cmd), flush=True)
            if args.dry_run:
                continue
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  ! failed (rc={result.returncode}); skipping rest of {vid}", file=sys.stderr, flush=True)
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
