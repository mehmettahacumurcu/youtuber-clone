"""Quick pipeline status — global progress from the manifest, per-stage counts.

Works in the cloud model where local WAVs may be absent: the corpus denominator
comes from ``data/manifest.json`` (falling back to the union of existing output ids).
The audio-duration section is best-effort over whatever WAVs happen to be present.

Run with: ``uv run --no-sync python -m pipeline.status``
"""
from __future__ import annotations

from pathlib import Path

from pipeline.config import Settings, load_settings
from pipeline.manifest import load_manifest

STAGES = ("meta", "vad", "diarize", "identify", "transcribe", "filter")


def _ids(d: Path) -> set[str]:
    return {p.stem for p in d.glob("*.json")} if d.exists() else set()


def compute_status(settings: Settings) -> dict:
    s = settings
    stage_ids = {
        "meta": _ids(s.paths.meta_dir),
        "vad": _ids(s.paths.vad_dir),
        "diarize": _ids(s.paths.diarize_dir),
        "identify": _ids(s.paths.identify_dir),
        "transcribe": _ids(s.paths.transcribe_dir),
        "filter": _ids(s.paths.filter_dir),
    }
    manifest = set(load_manifest(s))
    if manifest:
        corpus = manifest
    else:
        corpus = set().union(*stage_ids.values()) if any(stage_ids.values()) else set()

    stages = {name: len(ids & corpus) for name, ids in stage_ids.items()}
    pending = {name: sorted(corpus - ids) for name, ids in stage_ids.items()}
    return {
        "corpus": len(corpus),
        "manifest_known": bool(manifest),
        "stages": stages,
        "pending": pending,
    }


def _print_status(st: dict, settings: Settings) -> None:
    n = st["corpus"]
    src = "manifest" if st["manifest_known"] else "outputs (no manifest)"
    print(f"=== youtuber-clone pipeline status  (corpus = {n} videos, from {src}) ===\n")

    def _bar(done: int) -> str:
        pct = done / n * 100 if n else 0
        return f"{done:>4}/{n:<4} ({pct:>5.1f}%)"

    labels = {
        "meta": "Stage 1 Download ",
        "vad": "Stage 2 VAD      ",
        "diarize": "Stage 3 Diarize  ",
        "identify": "Stage 4 Identify ",
        "transcribe": "Stage 5 Transcribe",
        "filter": "Stage 6 Filter   ",
    }
    for name in STAGES:
        print(f"  {labels[name]} {_bar(st['stages'][name])}")
    print()

    # Best-effort local audio duration (WAVs are usually absent in the cloud model).
    try:
        import soundfile as sf  # optional; absent on minimal installs

        wavs = list(settings.paths.audio_dir.glob("*.wav")) if settings.paths.audio_dir.exists() else []
        if wavs:
            total_s = 0.0
            for w in wavs:
                try:
                    info = sf.info(str(w))
                    total_s += info.frames / info.samplerate
                except Exception:
                    pass
            print(f"  Local audio present: {len(wavs)} files, {total_s / 3600:.1f} h\n")
    except Exception:
        pass

    pend = st["pending"]
    todo = {k: v for k, v in pend.items() if k in ("diarize", "identify", "transcribe", "filter") and v}
    if todo:
        print("Pending:")
        for name in ("diarize", "identify", "transcribe", "filter"):
            if pend[name]:
                print(f"  {labels[name].strip()} needs {len(pend[name])} videos")
    elif n and st["stages"]["filter"] == n:
        print("All stages complete.")


def main() -> int:
    s = load_settings()
    _print_status(compute_status(s), s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
