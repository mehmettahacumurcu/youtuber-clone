"""LLM -> TTS end to end: ask Speaker a question, get his words (LLM via Ollama), speak them in his
voice (fine-tuned XTTS). The two halves of the clone wired together.

  .venv-tts\\Scripts\\python.exe training\\say.py "Kemal Kilicdaroglu hakkinda ne dusunuyorsun?"
  .venv-tts\\Scripts\\python.exe training\\say.py "Bir cumle." --text       # speak text directly (skip LLM)
  .venv-tts\\Scripts\\python.exe training\\say.py "soru" --eq 6              # stronger brightness cut
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import requests
import soundfile as sf
import torch

MODEL_DIR = "E:/youtuber-clone/models/xtts_speaker_v2"
WAVS = "E:/youtuber-clone/data/tts/wavs"
LANG = "tr"
OLLAMA = "http://localhost:11434/api/chat"
LLM_MODEL = "speaker-qwen3-run2-ep2"
SYS = ("Sen izinli egitim verisindeki anlatim uslubuna uyarlanmis bir "
       "dil modelisin. Sana bir soru sorulur; ogrendigin anlatim uslubuyla, "
       "KONUYA SADIK kalarak akici ve net Turkce yanit ver. Lafi dagitma, sorulani cevapla.")


def llm(question: str, num_predict: int = 220) -> str:
    body = {"model": LLM_MODEL, "messages": [{"role": "system", "content": SYS},
            {"role": "user", "content": question}], "stream": False, "keep_alive": 0,  # unload after -> free GPU for XTTS
            "options": {"temperature": 0.7, "top_p": 0.85, "repeat_penalty": 1.2, "num_predict": num_predict}}
    r = requests.post(OLLAMA, json=body, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()


def split_sentences(text: str, maxlen: int = 200) -> list[str]:
    out = []
    for p in re.split(r"(?<=[.!?])\s+", text.strip()):
        p = p.strip()
        if not p:
            continue
        if len(p) <= maxlen:
            out.append(p)
            continue
        buf = ""                                   # break over-long on commas
        for chunk in re.split(r"(?<=,)\s+", p):
            if len(buf) + len(chunk) + 1 <= maxlen:
                buf = (buf + " " + chunk).strip()
            else:
                if buf:
                    out.append(buf)
                buf = chunk
        if buf:
            out.append(buf)
    return [s if s[-1:] in ".!?" else s + "." for s in out]


def centroid(y, sr):
    import librosa
    return float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())


def pick_dark_refs(n=6, min_dur=6.0):
    """Research: darker/calmer reference clips -> less bright clone. Pick lowest-centroid long clips."""
    import librosa
    cand = []
    for w in sorted(glob.glob(f"{WAVS}/*.wav"))[:160]:
        info = sf.info(w)
        if info.frames / info.samplerate < min_dur:
            continue
        y, sr = librosa.load(w, sr=None)
        cand.append((centroid(y, sr), w))
    cand.sort(key=lambda x: x[0])
    return [w for _, w in cand[:n]]


def high_shelf(y, sr, fc=5500, cut_db=4.0):
    import librosa
    S = librosa.stft(y, n_fft=1024)
    f = librosa.fft_frequencies(sr=sr, n_fft=1024)
    g = np.ones_like(f)
    g[f > fc] = 10 ** (-cut_db / 20)
    ramp = (f >= fc - 1000) & (f <= fc)
    g[ramp] = np.linspace(1.0, 10 ** (-cut_db / 20), int(ramp.sum()))
    return librosa.istft(S * g[:, None], length=len(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--text", action="store_true", help="speak the input directly (skip the LLM)")
    ap.add_argument("--out", default="E:/youtuber-clone/data/tts/_sample/speaker_say.wav")
    ap.add_argument("--eq", type=float, default=4.0, help="high-shelf cut dB above 5.5kHz (0=off)")
    ap.add_argument("--refs", type=int, default=6)
    args = ap.parse_args()

    text = args.question if args.text else llm(args.question)
    print("=== TEXT (his words) ===\n", text, "\n")
    sents = split_sentences(text)
    print(f"{len(sents)} sentence(s) -> TTS\n")

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    cfg = XttsConfig()
    cfg.load_json(f"{MODEL_DIR}/config.json")
    m = Xtts.init_from_config(cfg)
    m.load_checkpoint(cfg, checkpoint_path=f"{MODEL_DIR}/best_model.pth",
                      vocab_path=f"{MODEL_DIR}/vocab.json", use_deepspeed=False)
    if torch.cuda.is_available():
        m.cuda()

    refs = pick_dark_refs(args.refs)
    print("refs (darkest/calmest):", [os.path.basename(r) for r in refs])
    gpt_lat, spk = m.get_conditioning_latents(audio_path=refs, gpt_cond_len=30,
                                              max_ref_length=30, sound_norm_refs=True)

    sr = 24000
    gap = np.zeros(int(0.18 * sr), dtype=np.float32)
    chunks = []
    for i, s in enumerate(sents):
        res = m.inference(s, LANG, gpt_lat, spk, temperature=0.7,
                          repetition_penalty=1.3, length_penalty=1.0)  # 1.3 smooths the stutter vs 2.0
        w = np.asarray(res["wav"], dtype=np.float32)
        chunks.extend([w, gap])
        print(f"  [{i+1}/{len(sents)}] {len(w)/sr:.1f}s  {s[:60]}")
    y = np.concatenate(chunks)
    if args.eq > 0:
        y = high_shelf(y, sr, cut_db=args.eq).astype(np.float32)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sf.write(args.out, y, sr)
    print(f"\nwrote {args.out}  ({len(y)/sr:.1f}s) | centroid {centroid(y, sr):.0f} Hz (his ~1805)")


if __name__ == "__main__":
    main()
