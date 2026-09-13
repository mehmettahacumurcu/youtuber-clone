"""Acoustically inspect the §6b checkpoint-compare samples to pick the sweet-spot XTTS epoch.

For every checkpoint tag x 8 probe sentences in xtts_cmp/, measure what the EAR is judging:
  - CER  : faster-whisper transcribes the clip; char-error vs the known probe text. Low = clear,
           no stutter/garble. s1 ("amina koyayim") is the overtraining canary.
  - F0   : median pitch (his real clips ~ computed below). High = the "too young" problem.
  - cen  : spectral centroid / brightness (his real ~ computed). High = bright/young timbre.
  - sim  : MFCC-cosine to his REAL voice profile (data/tts/wavs). High = sounds like him.
  - sil  : internal-silence ratio after trim. High = gappy/choppy prosody (stutter signature).
  - s/ch : seconds per character. Abnormally high = dragging/breaking.

Pick: the EARLIEST tag where sim is already high + F0/cen near his real AND CER/sil are still low
(before overtraining re-introduces the stutter).

  .venv-tts\\Scripts\\python.exe training\\inspect_cmp.py
"""
from __future__ import annotations

import glob
import os
import re

import librosa
import numpy as np
import soundfile as sf

CMP = "E:/youtuber-clone/xtts_cmp"
REAL = "E:/youtuber-clone/data/tts/wavs"
SR = 22_050

TEXTS = [
    "Türkiye'nin başkenti Ankara'dır ve en büyük şehri İstanbul'dur.",
    "Bu meseleyi yıllardır konuşuyoruz ama kimse dinlemiyor amına koyayım.",
    "Ya bak abi, bu işin aslı tamamen başka, sana açık açık anlatayım.",
    "Sence bu ülkede gerçekten adalet var mı, yoksa hepsi palavra mı?",
    "Şu çocuğun yaptığı şey gerçekten çok yanlıştı, değil mi?",
    "İki bin yirmi altı yılında işler iyice değişti arkadaşlar.",
    "Atatürk Cumhuriyet'i kurduğunda herkes buna inanmıyordu aslında.",
    "Geçen hafta arkadaşlarımla buluştuk, uzun uzun konuştuk, sonunda hiçbir şeye varamadık.",
]


def norm(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-zçğıöşü0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def f0_voiced(y):
    f0, _, _ = librosa.pyin(y, fmin=70, fmax=400, sr=SR, frame_length=1024)
    ok = ~np.isnan(f0)
    med = float(np.median(f0[ok])) if ok.any() else 0.0
    return med, float(ok.mean())


def centroid(y):
    return float(librosa.feature.spectral_centroid(y=y, sr=SR).mean())


def mfcc_vec(y):
    return librosa.feature.mfcc(y=y, sr=SR, n_mfcc=20).mean(axis=1)


def sil_ratio(y):
    yt, _ = librosa.effects.trim(y, top_db=30)
    if len(yt) < SR * 0.2:
        return 0.0
    rms = librosa.feature.rms(y=yt, frame_length=1024, hop_length=256)[0]
    thr = rms.max() * (10 ** (-30 / 20))
    return float((rms < thr).mean())


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main() -> None:
    # ---- his REAL voice profile (ground truth "him") ----
    real = sorted(glob.glob(f"{REAL}/*.wav"))[:80]
    rf0, rcen, rmfcc = [], [], []
    for w in real:
        y, _ = librosa.load(w, sr=SR)
        m, _ = f0_voiced(y)
        if m:
            rf0.append(m)
        rcen.append(centroid(y))
        rmfcc.append(mfcc_vec(y))
    REF_F0, REF_CEN = float(np.median(rf0)), float(np.median(rcen))
    REF_MFCC = np.mean(rmfcc, axis=0)
    print(f"HIS REAL voice (n={len(real)}): F0={REF_F0:.0f}Hz  centroid={REF_CEN:.0f}Hz\n")

    # ---- faster-whisper for CER ----
    from faster_whisper import WhisperModel
    try:
        asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
    except Exception as e:
        print("cuda whisper failed, cpu int8:", e)
        asr = WhisperModel("large-v3", device="cpu", compute_type="int8")

    import jiwer

    tags = sorted({os.path.basename(p).rsplit("_s", 1)[0] for p in glob.glob(f"{CMP}/*_s*.wav")},
                  key=lambda t: (t != "best_model", int(t.split("_")[-1]) if t[-1].isdigit() else 0))

    rows = []
    for tag in tags:
        cers, f0s, cens, sims, sils, spc = [], [], [], [], [], []
        s1_cer = None
        for j, txt in enumerate(TEXTS):
            wav = f"{CMP}/{tag}_s{j}.wav"
            if not os.path.exists(wav):
                continue
            y, _ = librosa.load(wav, sr=SR)
            dur = len(y) / SR
            segs, _ = asr.transcribe(wav, language="tr", beam_size=5)
            hyp = " ".join(s.text for s in segs)
            c = jiwer.cer(norm(txt), norm(hyp)) if norm(hyp) else 1.0
            m, _ = f0_voiced(y)
            cers.append(c)
            if j == 1:
                s1_cer = c
            f0s.append(m)
            cens.append(centroid(y))
            sims.append(cos(mfcc_vec(y), REF_MFCC))
            sils.append(sil_ratio(y))
            spc.append(dur / max(len(norm(txt)), 1))
        rows.append((tag, np.mean(cers), s1_cer, np.median([x for x in f0s if x]),
                     np.mean(cens), np.mean(sims), np.mean(sils), np.mean(spc)))
        print(f"  done {tag}")

    print(f"\n{'tag':16s}{'CER':>7s}{'s1CER':>7s}{'F0':>7s}{'cen':>7s}{'sim':>7s}{'sil':>7s}{'s/ch':>7s}")
    print(f"{'HIS REAL':16s}{'-':>7s}{'-':>7s}{REF_F0:>7.0f}{REF_CEN:>7.0f}{'1.00':>7s}{'-':>7s}{'-':>7s}")
    print("-" * 64)
    for tag, cer, s1, f0, cen, sim, sil, sc in rows:
        print(f"{tag:16s}{cer:>7.3f}{(s1 or 0):>7.3f}{f0:>7.0f}{cen:>7.0f}{sim:>7.3f}{sil:>7.3f}{sc:>7.3f}")

    print("\nLower CER/s1CER/sil = smoother. F0/cen closer to HIS REAL = more him (less young/bright). "
          "Higher sim = more him. Sweet spot = earliest tag with high sim + near-real F0/cen + low CER/sil.")


if __name__ == "__main__":
    main()
