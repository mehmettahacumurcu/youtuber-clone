"""Curate the TTS clip set per the XTTS recipe (research-backed).

From the extracted clips, keep only 3-11s clips ending in terminal punctuation, rank by speaker-ID
similarity (cleanest = most his voice), keep the best ~TARGET_MIN minutes, then for each kept clip:
resample 24k -> 22.05k with a HIGH-QUALITY resampler (soxr, reduces brightness aliasing), trim
leading/trailing silence to a uniform ~150ms pad, and RMS-normalize loudness (consistent boundaries).

  .venv-tts\\Scripts\\python.exe training\\curate_tts.py
"""
from __future__ import annotations

import collections
import glob
import json
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

TTS = "E:/youtuber-clone/data/tts"
CLEAN = "E:/youtuber-clone/data/clean"
TARGET_MIN = 110         # richer: cover his full register (was 80, too narrow -> sounded thin/young)
MIN_SIM = 0.55           # relaxed from 0.60 to admit more of his natural vocal range
SR = 22_050
TARGET_RMS_DBFS = -20.0


def res_type() -> str:
    try:
        import soxr  # noqa: F401
        return "soxr_hq"
    except Exception:
        return "kaiser_best"


def main() -> None:
    # text -> (similarity, avg_logprob) from the clean-stage metrics
    metrics: dict[tuple[str, str], tuple[float, float]] = {}
    for fp in glob.glob(f"{CLEAN}/*.json"):
        if fp.endswith(".review.json"):
            continue
        d = json.load(open(fp, encoding="utf-8"))
        vid = d["video_id"]
        for s in d.get("segments", []):
            t = (s.get("text") or "").strip()
            if t:
                metrics.setdefault((vid, t), (s.get("similarity", 0), s.get("avg_logprob", -9)))

    rows = [l for l in open(f"{TTS}/metadata.csv", encoding="utf-8").read().splitlines() if l.strip()]
    cand = []
    for l in rows:
        name, text, _ = l.split("|")
        vid = name.rsplit("_", 1)[0]
        wav = f"{TTS}/wavs/{name}.wav"
        if not os.path.exists(wav):
            continue
        info = sf.info(wav)
        dur = info.frames / info.samplerate
        tt = text.strip()
        if not (3 <= dur <= 11) or tt[-1:] not in ".!?":
            continue
        sim, lp = metrics.get((vid, tt), (0.5, -0.5))
        if sim < MIN_SIM:
            continue
        cand.append((name, text, wav, dur, sim, lp))

    # RICHER selection: round-robin across videos (best-first within each) so the set covers his
    # full range of contexts/registers, not just whichever video has the highest similarity.
    byvid: dict[str, list] = collections.defaultdict(list)
    for c in cand:
        byvid[c[0].rsplit("_", 1)[0]].append(c)
    for v in byvid:
        byvid[v].sort(key=lambda c: (-c[4], -c[5]))
    vids = list(byvid)
    idx = {v: 0 for v in vids}
    kept, tot = [], 0.0
    while tot < TARGET_MIN * 60 and any(idx[v] < len(byvid[v]) for v in vids):
        for v in vids:
            if idx[v] < len(byvid[v]) and tot < TARGET_MIN * 60:
                c = byvid[v][idx[v]]
                idx[v] += 1
                kept.append(c)
                tot += c[3]

    rt = res_type()
    target_rms = 10 ** (TARGET_RMS_DBFS / 20)
    keptnames = set()
    for name, text, wav, dur, sim, lp in kept:
        y, sr = librosa.load(wav, sr=None)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR, res_type=rt)
        y, _ = librosa.effects.trim(y, top_db=30)
        pad = int(0.15 * SR)
        y = np.concatenate([np.zeros(pad), y, np.zeros(pad)]).astype(np.float32)
        rms = float(np.sqrt(np.mean(y ** 2))) or 1e-6
        y = y * (target_rms / rms)
        pk = float(np.max(np.abs(y)))
        if pk > 0.97:
            y = y * (0.97 / pk)          # clip guard
        sf.write(wav, y, SR, subtype="PCM_16")
        keptnames.add(name)

    for w in glob.glob(f"{TTS}/wavs/*.wav"):
        if Path(w).stem not in keptnames:
            os.remove(w)
    open(f"{TTS}/metadata.csv", "w", encoding="utf-8").write(
        "\n".join(f"{n}|{t}|{t}" for n, t, _, _, _, _ in kept) + "\n")

    byv = collections.Counter(n.rsplit("_", 1)[0] for n, _, _, _, _, _ in kept)
    print(f"curated: {len(kept)} clips / {tot/60:.1f} min across {len(byv)} videos "
          f"(resample={rt}, sr={SR}, rms={TARGET_RMS_DBFS}dBFS)")


if __name__ == "__main__":
    main()
