"""Download audio for NEVER-SEEN videos — no filter output AND no local WAV.

Run locally from a home/residential IP to stage audio without tripping the Colab
datacenter bot-gate. The processing stages never touch YouTube, so this is the only
stage that needs the ungated IP.

Targets exactly: manifest ids  MINUS  {videos with a filter output}  MINUS  {videos
with a local data/audio/*.wav}. That excludes both Colab-processed videos and
already-downloaded-local videos, per the user's "just what we didn't see" scope.

In-process loop (one settings load, no per-video subprocess spawn), resumable
(re-running recomputes the todo set so finished downloads drop out), and it keeps
going past per-video failures.

    uv run python scripts/download_pending.py --config config.yaml [--limit N]

Throttle (sleep before each download) comes from config.download.sleep_interval_s;
override for a home IP via env, e.g. YOUTUBER_DOWNLOAD__SLEEP_INTERVAL_S=3.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import load_settings  # noqa: E402
from pipeline.manifest import load_manifest  # noqa: E402
from pipeline.download.runner import download_video  # noqa: E402


def compute_todo(settings):
    ids = load_manifest(settings)
    fdir, adir = settings.paths.filter_dir, settings.paths.audio_dir
    done = {p.stem for p in fdir.glob("*.json")} if fdir.exists() else set()
    wav = {p.stem for p in adir.glob("*.wav")} if adir.exists() else set()
    seen = done | wav
    return [v for v in ids if v not in seen], len(ids), len(done), len(wav)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="download_pending", description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--limit", type=int, default=None, help="Attempt at most N this run")
    args = p.parse_args(argv)

    settings = load_settings(args.config)
    todo, n_manifest, n_done, n_wav = compute_todo(settings)
    if args.limit is not None:
        todo = todo[: args.limit]
    print(
        f"manifest={n_manifest} processed_filter={n_done} local_wav={n_wav} "
        f"-> {len(todo)} to download",
        flush=True,
    )

    ok = fail = 0
    failed_ids: list[str] = []
    start = time.monotonic()
    for i, vid in enumerate(todo, 1):
        url = f"https://www.youtube.com/watch?v={vid}"
        print(f"[{i}/{len(todo)}] {vid}", flush=True)
        try:
            path = download_video(url, settings)
        except Exception as exc:  # noqa: BLE001
            path = None
            print(f"  ! exception {vid}: {exc}", file=sys.stderr, flush=True)
        if path is not None:
            ok += 1
        else:
            fail += 1
            failed_ids.append(vid)
            print(f"  ! FAILED {vid}", file=sys.stderr, flush=True)
        if i % 10 == 0:
            el = (time.monotonic() - start) / 60.0
            print(
                f"  ... {ok} ok / {fail} fail / {len(todo) - i} left / {el:.1f} min elapsed",
                flush=True,
            )

    el = (time.monotonic() - start) / 60.0
    print(f"DONE: {ok} downloaded, {fail} failed, {len(todo)} attempted, {el:.1f} min", flush=True)
    if failed_ids:
        print("FAILED IDS: " + " ".join(failed_ids), file=sys.stderr, flush=True)
    return 1 if (fail and not ok) else 0


if __name__ == "__main__":
    raise SystemExit(main())
