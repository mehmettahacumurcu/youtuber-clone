"""Add ONE user-supplied reference clip (clean him-only span) to the speaker pool.

The user hand-picks clips of just him talking (video + start/end). Each new clip from
DIFFERENT acoustics raises his ECAPA similarity uniformly (identify uses MAX cosine across
all active pool members), strengthening Stage-4 speaker-ID before the full A100 run.

What it does, idempotently, for one clip:
  1. Ensure the source audio is local (download_video -> data/audio/{vid}.wav, 16 kHz mono).
  2. ffmpeg-slice [start,end] -> data/reference/pool/candidates/{idx}_user_{vid}_{a}-{b}.wav.
  3. ECAPA-embed it; print sim-to-seed as a SANITY CHECK (genuine him ~0.6-0.75; a low score
     means the span likely caught music/another voice -> re-check the timestamps).
  4. Register as an ALWAYS-ACTIVE pool member: append to candidates.json (new candidate_index),
     append the L2-normalized embedding row to candidate_embeddings.npy (kept in index order
     for the UI), and add the index to active.json.

identify RE-EMBEDS the active wavs at runtime (it does NOT read the .npy), so the wav +
candidates.json + active.json are what actually drive identification; the .npy is updated only
to keep the UI consistent.

After all clips are added, rebuild the Colab bundle: python colab/make_reference_bundle.py

    python scripts/add_reference_clip.py --video-id xgdLBF7K2rI --start 0:15 --end 1:30
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import REPO_ROOT, load_settings  # noqa: E402
from pipeline.download import audio_path_for  # noqa: E402
from pipeline.download.runner import download_video  # noqa: E402
from pipeline.identify.runner import (  # noqa: E402
    _cosine, _embed, _load_embedding_model, _read_mono_16k, _resolve_device,
)
from pipeline.identify.pool import (  # noqa: E402
    _slice_wav, candidate_embeddings_path, candidates_dir, candidates_manifest_path,
)
from ui.state import load_active_pool, save_active_pool  # noqa: E402

LOW_SIM_WARN = 0.45


def parse_ts(s: str) -> float:
    """Parse 'M:SS' / 'H:MM:SS' / plain seconds -> float seconds."""
    s = s.strip()
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        sec = 0.0
        for p in parts:
            sec = sec * 60 + p
        return sec
    return float(s)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="add_reference_clip", description=__doc__)
    p.add_argument("--config", default=None)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video-id", help="YouTube id of the source video")
    g.add_argument("--url", help="Full YouTube URL")
    p.add_argument("--start", required=True, help="Clip start (e.g. 0:15 or 15)")
    p.add_argument("--end", required=True, help="Clip end (e.g. 1:30 or 90)")
    p.add_argument("--note", default="", help="Optional note stored on the candidate")
    args = p.parse_args(argv)

    import numpy as np

    settings = load_settings(args.config)
    sr = settings.download.sample_rate_hz
    vid = args.video_id or args.url.rsplit("=", 1)[-1].rsplit("/", 1)[-1]
    url = args.url or f"https://www.youtube.com/watch?v={vid}"
    start, end = parse_ts(args.start), parse_ts(args.end)
    if end <= start:
        print(f"ERROR: end ({end}) <= start ({start})", file=sys.stderr)
        return 2

    # 1) ensure audio local (download is a no-op if already present)
    audio = audio_path_for(vid, settings)
    if not audio.exists():
        print(f"downloading {vid} (home IP, ungated) ...", flush=True)
        got = download_video(url, settings)
        if got is None or not audio.exists():
            print(f"ERROR: download failed for {vid}", file=sys.stderr)
            return 1
    dur_total = len(_read_mono_16k(audio, sr)) / sr
    if end > dur_total + 0.5:
        print(f"ERROR: end {end:.1f}s exceeds video length {dur_total:.1f}s", file=sys.stderr)
        return 2

    # 2) determine the next candidate_index and slice the clip
    manifest_path = candidates_manifest_path(settings)
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    cands = doc["candidates"]
    next_idx = max(c["candidate_index"] for c in cands) + 1
    dst = candidates_dir(settings) / f"{next_idx:02d}_user_{vid}_{start:.1f}-{end:.1f}.wav"
    _slice_wav(audio, start, end, dst, sr)

    # 3) embed + sim-to-seed sanity check
    dev = _resolve_device()
    model = _load_embedding_model(settings.identify.embedding_model, dev)
    clip_emb = _embed(model, _read_mono_16k(dst, sr), dev)
    seed_emb = _embed(model, _read_mono_16k(settings.paths.reference_clip, sr), dev)
    sim_seed = _cosine(clip_emb, seed_emb)
    # also max-sim vs the currently-active pool (how much NEW acoustic coverage it adds)
    active = load_active_pool(settings)["active_indices"]
    by_idx = {c["candidate_index"]: c for c in cands}
    max_active = -1.0
    for i in active:
        f = REPO_ROOT / by_idx[i]["file"]
        if f.exists():
            max_active = max(max_active, _cosine(clip_emb, _embed(model, _read_mono_16k(f, sr), dev)))

    # 4) register: candidates.json + candidate_embeddings.npy + active.json
    cands.append({
        "candidate_index": next_idx,
        "is_seed": False,
        "user_supplied": True,
        "source_video": vid,
        "source_speaker_label": None,
        "start": round(start, 2),
        "end": round(end, 2),
        "duration_s": round(end - start, 2),
        "sim_to_seed": round(sim_seed, 4),
        "note": args.note,
        "file": str(dst.relative_to(REPO_ROOT)).replace("\\", "/"),
    })
    doc["pool_size_actual"] = len(cands)
    doc["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    # append L2-normalized embedding row in candidate_index order (UI sidecar)
    npy_path = candidate_embeddings_path(settings)
    e = clip_emb.flatten().float()
    e = (e / (e.norm() + 1e-8)).cpu().numpy().astype(np.float32)
    if npy_path.exists():
        arr = np.load(str(npy_path))
        if arr.shape[0] == next_idx and arr.shape[1] == e.shape[0]:
            arr = np.vstack([arr, e[None, :]])
        else:
            print(f"  WARN: npy shape {arr.shape} not aligned with next_idx {next_idx}; "
                  f"UI sidecar left as-is (identify re-embeds, so identification is unaffected)")
            arr = None
    else:
        arr = e[None, :]
    if arr is not None:
        np.save(str(npy_path), arr)

    save_active_pool(settings, active + [next_idx])

    flag = "  <-- LOW: re-check this span (music/other voice?)" if sim_seed < LOW_SIM_WARN else ""
    print(f"\nadded reference clip #{next_idx}: {vid} [{start:.1f}-{end:.1f}s] ({end-start:.1f}s)")
    print(f"  sim_to_seed = {sim_seed:.3f}{flag}")
    print(f"  max sim vs active pool = {max_active:.3f}  (lower = more NEW acoustic coverage)")
    print(f"  file: {dst.relative_to(REPO_ROOT)}")
    print(f"  active pool now has {len(active) + 1} members")
    print("  -> when all clips are in, run: python colab/make_reference_bundle.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
