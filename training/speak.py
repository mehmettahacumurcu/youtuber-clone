"""Local XTTS-v2 inference for the Speaker voice clone, with a pitch (F0) diagnostic.

The model came out high-pitched; this script (a) generates speech from the fine-tuned model and
(b) measures the median F0 of the output vs his real clips, so we can see exactly how many semitones
off it is and whether it's the 24k-vs-22.05k sample-rate bug (ratio ~1.088).

Run with the isolated TTS env:
  .venv-tts\\Scripts\\python.exe training\\speak.py "Merhaba, nasilsin?"            # synth -> out.wav
  .venv-tts\\Scripts\\python.exe training\\speak.py --diagnose                       # F0 gen vs real
  .venv-tts\\Scripts\\python.exe training\\speak.py "metin" --out-sr 22050           # test SR-label fix
  .venv-tts\\Scripts\\python.exe training\\speak.py "metin" --pitch -1.5             # pitch-shift fix
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import soundfile as sf
import torch

MODEL_DIR = "E:/youtuber-clone/models/xtts_speaker_v2"
WAVS = "E:/youtuber-clone/data/tts/wavs"
LANG = "tr"
DEFAULT_TEXT = "Bak şimdi sana açık açık söyleyeyim, bu işin aslı tamamen başka."


def load_model():
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    cfg = XttsConfig()
    cfg.load_json(f"{MODEL_DIR}/config.json")
    m = Xtts.init_from_config(cfg)
    m.load_checkpoint(cfg, checkpoint_path=f"{MODEL_DIR}/best_model.pth",
                      vocab_path=f"{MODEL_DIR}/vocab.json", use_deepspeed=False)
    if torch.cuda.is_available():
        m.cuda()
    out_sr = getattr(cfg, "audio", None)
    out_sr = getattr(out_sr, "output_sample_rate", 24000) if out_sr else 24000
    return m, out_sr


def ref_clips(n=6):
    return sorted(glob.glob(f"{WAVS}/*.wav"))[:n]


def f0_median(wav: np.ndarray, sr: int) -> float:
    """Autocorrelation median F0 over voiced frames (75-400 Hz). No librosa needed."""
    wav = np.asarray(wav, dtype=np.float64)
    if wav.ndim > 1:
        wav = wav[:, 0]
    frame, hop = int(0.04 * sr), int(0.02 * sr)
    lo, hi = int(sr / 400), int(sr / 75)
    f0s = []
    for i in range(0, max(0, len(wav) - frame), hop):
        fr = wav[i:i + frame]
        if np.sqrt(np.mean(fr ** 2)) < 0.01:
            continue
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, "full")[frame - 1:]
        if ac[0] <= 0:
            continue
        seg = ac[lo:hi]
        if not len(seg):
            continue
        peak = int(np.argmax(seg)) + lo
        if ac[peak] / ac[0] > 0.3:
            f0s.append(sr / peak)
    return float(np.median(f0s)) if f0s else 0.0


def generate(model, out_sr, text, refs, pitch=0.0, out_sr_override=None):
    gpt_lat, spk = model.get_conditioning_latents(audio_path=refs)
    res = model.inference(text, LANG, gpt_lat, spk, temperature=0.7)
    wav = np.asarray(res["wav"], dtype=np.float32)
    write_sr = out_sr_override or out_sr
    if pitch:
        import librosa
        wav = librosa.effects.pitch_shift(wav, sr=out_sr, n_steps=pitch)
    return wav, write_sr


def main():
    ap = argparse.ArgumentParser(description="XTTS Speaker local synth + pitch diagnostic")
    ap.add_argument("text", nargs="?", default=DEFAULT_TEXT)
    ap.add_argument("--out", default="E:/youtuber-clone/data/tts/_sample/out.wav")
    ap.add_argument("--diagnose", action="store_true", help="measure F0 of output vs his real clips")
    ap.add_argument("--out-sr", type=int, default=None, help="write at this SR (test 22050 SR-label fix)")
    ap.add_argument("--pitch", type=float, default=0.0, help="pitch-shift output by N semitones")
    ap.add_argument("--refs", type=int, default=6, help="how many real clips as the voice reference")
    args = ap.parse_args()

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    refs = ref_clips(args.refs)

    if args.diagnose:
        # His real pitch from the reference clips
        real = []
        for r in refs:
            w, sr = sf.read(r)
            f = f0_median(w, sr)
            if f:
                real.append(f)
        real_f0 = float(np.median(real)) if real else 0.0
        print(f"his real median F0: {real_f0:.1f} Hz  (from {len(real)} clips)")

    print("loading model...")
    model, out_sr = load_model()
    print(f"model output_sample_rate = {out_sr}")
    wav, write_sr = generate(model, out_sr, args.text, refs, pitch=args.pitch, out_sr_override=args.out_sr)
    sf.write(args.out, wav, write_sr)
    print(f"wrote {args.out}  (sr={write_sr}, {len(wav)/write_sr:.1f}s)")

    if args.diagnose:
        gen_f0 = f0_median(wav, out_sr)   # measure at the TRUE generation sr
        print(f"generated median F0: {gen_f0:.1f} Hz")
        if gen_f0 and real_f0:
            ratio = gen_f0 / real_f0
            semis = 12 * np.log2(ratio)
            print(f"\n  ratio gen/real = {ratio:.3f}  ({semis:+.2f} semitones)")
            print(f"  24000/22050 = {24000/22050:.3f}  (the SR-mismatch signature)")
            if abs(ratio - 24000 / 22050) < 0.03:
                print("  -> matches the SR bug. Fix: write at 22050 (--out-sr 22050).")
            elif ratio > 1.03:
                print(f"  -> high but not the SR signature. Fix: --pitch {-semis:.1f}")
            else:
                print("  -> pitch is close; the issue may be timbre/fluency, not pitch.")


if __name__ == "__main__":
    main()
