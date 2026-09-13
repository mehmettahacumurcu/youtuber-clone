"""Colab corpus driver — drain ALL pending videos through Stages 1-6, checkpoint to git.

Unlike run_batch (fixed N, no git), this processes the whole pending list so one
unattended Colab session makes maximal progress, committing after each video so a
disconnect loses at most --commit-every videos. A --max-minutes wall-clock budget stops
cleanly (checked BEFORE each video) so the next session resumes (stages are idempotent;
finished videos skip).

    python colab/run_corpus.py --config config.yaml [--max-minutes 600]
                               [--commit-every 1] [--no-checkpoint] [--dry-run]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Ensure the repo root is importable when run as a script.
sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import load_settings  # noqa: E402
from pipeline.manifest import load_manifest  # noqa: E402
from colab.run_batch import commands_for  # noqa: E402


def select_pending(manifest_ids: list[str], filter_dir: Path) -> list[str]:
    """Manifest ids lacking a filter output (the terminal stage of this driver)."""
    done = {p.stem for p in filter_dir.glob("*.json")} if filter_dir.exists() else set()
    return [v for v in manifest_ids if v not in done]


def run_video(video_id: str, config_path, *, dry_run: bool) -> bool:
    """Run every stage for one video. Returns False if a stage failed."""
    print(f"\n=== {video_id} ===", flush=True)
    for cmd in commands_for(video_id, config_path):
        print("  $ " + " ".join(cmd), flush=True)
        if dry_run:
            continue
        result = subprocess.run(cmd)
        if result.returncode != 0:
            stage = cmd[2] if len(cmd) > 2 else "?"
            print(
                f"  ! {stage} failed (rc={result.returncode}); skipping rest of {video_id}",
                file=sys.stderr,
                flush=True,
            )
            return False
    return True


def git_checkpoint(n: int) -> None:
    """Commit + push the small JSON outputs so progress survives a disconnect.

    All git calls run with cwd=REPO_ROOT so they work regardless of the process cwd.
    """
    paths = ["data/manifest.json", "data/meta", "data/vad", "data/diarize",
             "data/identify", "data/transcribe", "data/filter"]
    subprocess.run(["git", "add", *paths], cwd=REPO_ROOT, check=False)
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT).returncode != 0
    if not staged:
        return
    subprocess.run(
        ["git", "commit", "-m", f"data: checkpoint after {n} videos on Colab"],
        cwd=REPO_ROOT,
        check=False,
    )
    push = subprocess.run(["git", "push"], cwd=REPO_ROOT, check=False)
    if push.returncode != 0:
        print(
            f"  ! git push failed (rc={push.returncode}); progress committed locally but NOT pushed.",
            file=sys.stderr,
            flush=True,
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_corpus", description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--max-minutes", type=float, default=None, help="Stop cleanly after this wall-clock budget")
    p.add_argument("--limit", type=int, default=None,
                   help="Attempt at most N pending videos this run (then stop). Batches a drain so a "
                        "bot-gated IP doesn't burn the session fast-failing the whole pending list.")
    p.add_argument("--commit-every", type=int, default=1, help="git checkpoint every K videos")
    p.add_argument("--no-checkpoint", action="store_true", default=True, help="Do not git commit/push (default)")
    p.add_argument("--checkpoint", action="store_false", dest="no_checkpoint", help="Explicitly enable Git checkpoints only in a separate private data repository")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = load_settings(args.config)
    pending = select_pending(load_manifest(settings), settings.paths.filter_dir)
    if args.limit is not None and args.limit >= 0:
        pending = pending[: args.limit]
    print(f"{len(pending)} videos this run (pending list capped by --limit={args.limit}): {pending}"
          if args.limit is not None else f"{len(pending)} videos pending: {pending}", flush=True)

    checkpoint = not (args.dry_run or args.no_checkpoint)
    start = time.monotonic()
    processed = 0
    failed = 0
    for vid in pending:
        if args.max_minutes is not None and (time.monotonic() - start) / 60.0 >= args.max_minutes:
            print(
                f"Reached --max-minutes before {vid}; stopping cleanly after {processed} videos.",
                flush=True,
            )
            break
        ok = run_video(vid, args.config, dry_run=args.dry_run)
        if not ok:
            failed += 1
        processed += 1
        if checkpoint and processed % args.commit_every == 0:
            git_checkpoint(processed)
    if checkpoint:
        git_checkpoint(processed)  # final flush of any uncommitted progress
    if failed:
        print(f"WARNING: {failed}/{processed} videos had stage failures.", file=sys.stderr, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
