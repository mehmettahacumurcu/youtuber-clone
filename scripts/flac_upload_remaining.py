"""Finish the Drive upload faster by sending the not-yet-uploaded WAVs as lossless FLAC.

FLAC is lossless — it decompresses to bit-identical PCM, so speaker-ID/ASR see exactly the
same audio, but the files are ~half the size → ~2x faster upload. The 124 WAVs already on
Drive are left as-is (also lossless). Drive ends up with a mix of .wav + .flac (Colab will
decompress the .flac back to .wav before processing).

Resumable + idempotent: re-listing the Drive folder recomputes what's left, FLAC encoding skips
files already encoded, and rclone skips files already uploaded. Safe to re-run.

    python scripts/flac_upload_remaining.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

RCLONE = r"C:\Users\Developer\.local\bin\rclone.exe"
REMOTE = "gdrive:youtuber_audio"
AUDIO = Path("data/audio")
FLAC = Path("data/audio_flac")
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


def on_drive() -> set[str]:
    out = subprocess.run([RCLONE, "lsf", REMOTE], capture_output=True, text=True)
    done: set[str] = set()
    for f in out.stdout.splitlines():
        f = f.strip()
        if f.endswith(".wav"):
            done.add(f[:-4])
        elif f.endswith(".flac"):
            done.add(f[:-5])
    return done


def main() -> int:
    FLAC.mkdir(parents=True, exist_ok=True)
    done = on_drive()
    local = sorted(AUDIO.glob("*.wav"))
    remaining = [p for p in local if p.stem not in done]
    print(f"on Drive: {len(done)}  local WAVs: {len(local)}  remaining to FLAC+upload: {len(remaining)}", flush=True)

    # 1) encode remaining WAV -> lossless FLAC (skip already-encoded)
    enc = 0
    for i, p in enumerate(remaining, 1):
        out_flac = FLAC / f"{p.stem}.flac"
        if out_flac.exists():
            continue
        r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(p),
                            "-c:a", "flac", "-compression_level", "8", str(out_flac)])
        if r.returncode == 0:
            enc += 1
        else:
            print(f"  ! encode failed: {p.stem}", file=sys.stderr, flush=True)
        if i % 25 == 0:
            print(f"  encoded {i}/{len(remaining)} ...", flush=True)
    n_flac = len(list(FLAC.glob("*.flac")))
    print(f"encoded {enc} new FLACs (audio_flac now holds {n_flac})", flush=True)

    # 2) upload the FLACs, RESILIENT to network drops. rclone copy is idempotent (skips files
    # already on Drive), so we loop: each pass resumes where the last left off. rclone's own
    # --retries / --low-level-retries ride out short blips within a pass; the outer loop rides
    # out full failures (e.g. a DNS/connection drop that kills the whole pass) by waiting and
    # re-running until it exits 0 (everything uploaded) or we hit the attempt cap.
    print("uploading FLACs to Drive (resilient mode) ...", flush=True)
    for attempt in range(1, 41):
        print(f"upload attempt {attempt} ...", flush=True)
        up = subprocess.run([RCLONE, "copy", str(FLAC), REMOTE,
                             "--transfers", "4", "--checkers", "8", "--drive-chunk-size", "64M",
                             "--retries", "10", "--retries-sleep", "20s", "--low-level-retries", "20",
                             "--log-file", "logs/rclone_flac_upload.log", "--log-level", "INFO"])
        if up.returncode == 0:
            print(f"upload complete on attempt {attempt} (exit 0).", flush=True)
            print("DONE.", flush=True)
            return 0
        print(f"  attempt {attempt} failed (exit {up.returncode}) — likely a network drop; "
              f"waiting 90s then resuming (already-uploaded files are skipped)", flush=True)
        time.sleep(90)
    print("gave up after 40 attempts; re-run scripts/run_flac_upload.cmd to continue.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
