"""In-process, STAGE-MAJOR Colab driver — load each heavy model ONCE, loop all videos.

Why this exists
---------------
The per-stage CLIs (``python -m pipeline.<stage>``) each cache their heavy model with
``@lru_cache`` *per process*. ``colab/run_corpus.py`` spawns a fresh subprocess per stage
PER VIDEO, so every model (Silero VAD, pyannote segmentation, ECAPA, faster-whisper) reloads
for every single video — ~5 h of pure model-load overhead across ~468 videos.

This driver fixes that by being STAGE-MAJOR and IN-PROCESS: for each stage it loops over all
target videos calling that stage's ``run_*`` function directly in one Python process, so the
``@lru_cache`` keeps the model resident across every video. Each model loads exactly once per
stage per process.

Multi-session A100 workflow
---------------------------
WAVs are staged to Google Drive once (by the download notebook). For processing, open
``colab/run_processing.ipynb`` in N separate Colab sessions (one per Google account, each on an
A100 40 GB). Give each session a different ``--shard i`` with the same ``--num-shards N``. The
shard filter is ``sha1(video_id) % N == i`` — a STABLE hash (NOT Python's ``hash()``, which is
salted per process), so the same video always lands in the same shard across sessions/machines.
Each session copies only its shard's WAVs from Drive to local disk (huge I/O win) then runs:

    python colab/run_stages.py --config config.yaml --shard i --num-shards N

Stages are idempotent (a video whose output exists is skipped unless ``--force``), so a session
that dies mid-shard just resumes on restart. After each stage the small JSON outputs are
git-committed + pushed (push failure is non-fatal); shards touch disjoint files, so parallel
pushes only need an occasional ``git pull --rebase`` between sessions.

    python colab/run_stages.py --config config.yaml [--shard i --num-shards N]
                               [--stages vad,diarize,identify,transcribe,filter]
                               [--force] [--max-minutes 600] [--commit-every 0]
                               [--no-checkpoint] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Ensure the repo root is importable when run as a script (python colab/run_stages.py).
sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import load_settings  # noqa: E402
from pipeline.manifest import load_manifest  # noqa: E402

log = logging.getLogger("run_stages")

# Ordered stage registry. Each entry lazily imports the stage's runner so --dry-run (and
# building the worklists) never imports torch/pyannote/whisper. The closures wrap the public
# run_* function and its *_path_for so we never reach into a runner's privates.
#   run(video_id, settings, *, force)  -> calls the stage in-process (model stays cached)
#   out_path(video_id, settings)       -> the stage's output JSON path (for skip detection)
STAGE_ORDER = ("vad", "diarize", "identify", "transcribe", "filter")


def _vad_runner():
    from pipeline.vad.runner import run_vad, vad_path_for

    def run(vid, settings, *, force):
        run_vad(vid, settings, force=force)

    return run, vad_path_for


def _diarize_runner():
    from pipeline.diarize.runner import run_diarization, diarize_path_for

    def run(vid, settings, *, force):
        run_diarization(vid, settings, force=force)

    return run, diarize_path_for


def _identify_runner():
    from pipeline.identify.runner import run_identify, identify_path_for

    def run(vid, settings, *, force):
        run_identify(vid, settings, force=force)

    return run, identify_path_for


def _transcribe_runner():
    from pipeline.transcribe.runner import run_transcribe, transcribe_path_for

    def run(vid, settings, *, force):
        run_transcribe(vid, settings, force=force)

    return run, transcribe_path_for


def _filter_runner():
    from pipeline.filter.runner import run_filter, filter_path_for

    def run(vid, settings, *, force):
        run_filter(vid, settings, force=force)

    return run, filter_path_for


STAGE_FACTORIES = {
    "vad": _vad_runner,
    "diarize": _diarize_runner,
    "identify": _identify_runner,
    "transcribe": _transcribe_runner,
    "filter": _filter_runner,
}

# Stage -> module that exposes its *_path_for, imported ONLY for path computation (no model
# load). vad/diarize/identify/transcribe path helpers live in the runner; filter's lives in
# pipeline.filter.runner too. Importing a runner module pulls torch transitively but does NOT
# load any model (the @lru_cache loaders only fire when called), so this is dry-run safe.


def _out_path_for(stage: str, video_id: str, settings):
    """The output JSON path for ``stage``/``video_id`` — used for skip detection in --dry-run
    and the live loop, without loading the stage model."""
    if stage == "vad":
        from pipeline.vad.runner import vad_path_for as f
    elif stage == "diarize":
        from pipeline.diarize.runner import diarize_path_for as f
    elif stage == "identify":
        from pipeline.identify.runner import identify_path_for as f
    elif stage == "transcribe":
        from pipeline.transcribe.runner import transcribe_path_for as f
    elif stage == "filter":
        from pipeline.filter.runner import filter_path_for as f
    else:  # pragma: no cover - guarded by parse_stages
        raise ValueError(f"unknown stage {stage!r}")
    return f(video_id, settings)


def in_shard(video_id: str, shard: int, num_shards: int) -> bool:
    """Stable shard membership: sha1(video_id) % num_shards == shard.

    Uses sha1 — NOT Python's builtin ``hash()``, which is per-process salted (PYTHONHASHSEED)
    and would put a video in different shards across sessions, double-processing some and
    skipping others.
    """
    h = int(hashlib.sha1(video_id.encode("utf-8")).hexdigest(), 16)
    return h % num_shards == shard


def shard_filter(video_ids: list[str], shard: int | None, num_shards: int | None) -> list[str]:
    if num_shards is None or shard is None:
        return list(video_ids)
    return [v for v in video_ids if in_shard(v, shard, num_shards)]


def stage_worklist(stage: str, candidates: list[str], settings, *, force: bool) -> list[str]:
    """Videos in ``candidates`` this stage should run.

    Mirrors each runner's own skip rule: skip a video whose output JSON already exists unless
    ``--force``. For the FIRST stage (vad) ``candidates`` is already restricted to ids with a
    local WAV; later stages assume the previous stage's output exists (a missing prerequisite
    surfaces as a per-video FileNotFoundError that is logged and skipped).
    """
    if force:
        return list(candidates)
    work: list[str] = []
    for vid in candidates:
        if not _out_path_for(stage, vid, settings).exists():
            work.append(vid)
    return work


def git_checkpoint(message: str) -> None:
    """Commit + push the small JSON output dirs so progress survives a disconnect.

    Mirrors run_corpus.git_checkpoint: all git calls use cwd=REPO_ROOT so they work regardless
    of the process cwd; a failed push is logged and non-fatal (progress is committed locally).
    """
    paths = [
        "data/manifest.json",
        "data/vad",
        "data/diarize",
        "data/identify",
        "data/transcribe",
        "data/filter",
    ]
    subprocess.run(["git", "add", *paths], cwd=REPO_ROOT, check=False)
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT
    ).returncode != 0
    if not staged:
        return
    subprocess.run(["git", "commit", "-m", message], cwd=REPO_ROOT, check=False)
    push = subprocess.run(["git", "push"], cwd=REPO_ROOT, check=False)
    if push.returncode != 0:
        log.warning(
            "git push failed (rc=%d); progress committed locally but NOT pushed.",
            push.returncode,
        )


def parse_stages(raw: str) -> list[str]:
    stages = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGE_FACTORIES]
    if unknown:
        raise SystemExit(
            f"unknown stage(s) {unknown}; valid: {', '.join(STAGE_ORDER)}"
        )
    # Preserve the canonical processing order regardless of how the user listed them.
    return [s for s in STAGE_ORDER if s in stages]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_stages", description=__doc__)
    p.add_argument("--config", default=None, help="Path to config.yaml")
    p.add_argument("--shard", type=int, default=None, help="This session's shard index i")
    p.add_argument("--num-shards", type=int, default=None, help="Total shards N (sha1 % N == i)")
    p.add_argument(
        "--stages",
        default=",".join(STAGE_ORDER),
        help="Comma list of stages to run (default: all, in pipeline order)",
    )
    p.add_argument("--force", action="store_true", help="Re-run even if output JSON exists")
    p.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="Stop cleanly between videos after this wall-clock budget",
    )
    p.add_argument(
        "--commit-every",
        type=int,
        default=0,
        help="git checkpoint every K videos WITHIN a stage (0 = only after each stage)",
    )
    p.add_argument("--no-checkpoint", action="store_true", default=True, help="Do not git commit/push (default)")
    p.add_argument("--checkpoint", action="store_false", dest="no_checkpoint", help="Explicitly enable Git checkpoints only in a separate private data repository")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan + per-stage shard worklist counts; load no model, touch no git",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if (args.shard is None) != (args.num_shards is None):
        raise SystemExit("--shard and --num-shards must be given together (or neither)")
    if args.num_shards is not None and not (0 <= args.shard < args.num_shards):
        raise SystemExit(f"--shard must be in [0, {args.num_shards}); got {args.shard}")

    settings = load_settings(args.config)
    stages = parse_stages(args.stages)

    # Worklist universe = manifest, sharded. The FIRST stage (vad) is further restricted to ids
    # whose WAV is on local disk (downloaded / copied from Drive); later stages don't need the
    # WAV present here — they read the prior stage's JSON, but in practice the WAV is required
    # for diarize/identify/transcribe too, so those will FileNotFoundError-skip if it's absent.
    manifest_ids = load_manifest(settings)
    shard_ids = shard_filter(manifest_ids, args.shard, args.num_shards)
    audio_dir = settings.paths.audio_dir
    have_wav = [v for v in shard_ids if (audio_dir / f"{v}.wav").exists()]

    shard_desc = (
        f"shard {args.shard}/{args.num_shards}"
        if args.num_shards is not None
        else "no sharding (all shards)"
    )
    print(
        f"manifest={len(manifest_ids)}  {shard_desc} -> {len(shard_ids)} ids  "
        f"({len(have_wav)} with local WAV in {audio_dir})",
        flush=True,
    )
    print(f"stages: {stages}  force={args.force}", flush=True)

    # First stage starts from the local-WAV set; subsequent stages start from the full shard set
    # (their input is the prior stage's JSON, which they may or may not have produced — a missing
    # prerequisite is a logged per-video skip, not a crash).
    for stage in stages:
        candidates = have_wav if stage == STAGE_ORDER[0] else shard_ids
        work = stage_worklist(stage, candidates, settings, force=args.force)
        done = len(candidates) - len(work)
        print(
            f"  [{stage}] candidates={len(candidates)} already_done={done} to_run={len(work)}",
            flush=True,
        )

    if args.dry_run:
        print("\n--dry-run: plan printed; no models loaded, no git, nothing executed.", flush=True)
        return 0

    checkpoint = not args.no_checkpoint
    start = time.monotonic()
    stopped_early = False

    for stage in stages:
        candidates = have_wav if stage == STAGE_ORDER[0] else shard_ids
        work = stage_worklist(stage, candidates, settings, force=args.force)
        if not work:
            print(f"\n=== stage {stage}: nothing to do ===", flush=True)
            continue

        print(f"\n=== stage {stage}: {len(work)} videos ===", flush=True)
        run_fn, _ = STAGE_FACTORIES[stage]()  # imports + caches the model on first call below

        processed = 0
        failed = 0
        for vid in work:
            if args.max_minutes is not None and (time.monotonic() - start) / 60.0 >= args.max_minutes:
                print(
                    f"Reached --max-minutes during stage {stage} before {vid}; "
                    f"stopping cleanly after {processed} videos in this stage.",
                    flush=True,
                )
                stopped_early = True
                break
            try:
                run_fn(vid, settings, force=args.force)
            except Exception as exc:  # noqa: BLE001
                log.exception("stage %s failed for %s: %s", stage, vid, exc)
                failed += 1
            processed += 1
            if checkpoint and args.commit_every and processed % args.commit_every == 0:
                git_checkpoint(
                    f"data: stage {stage} checkpoint ({shard_desc}) [{processed} videos]"
                )

        if failed:
            log.warning("stage %s: %d/%d videos failed", stage, failed, processed)
        if checkpoint:
            git_checkpoint(f"data: stage {stage} checkpoint ({shard_desc})")
        if stopped_early:
            break

    if stopped_early:
        print("\nStopped early on --max-minutes; re-run to resume (finished videos skip).", flush=True)
    else:
        print("\nAll selected stages complete for this shard.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
