"""TTS intelligibility gate: synthesize fixed Turkish sentences with BASE XTTS vs OUR fine-tune,
transcribe with Whisper-large-v3, compute CER vs the input text. Lower CER = cleaner Turkish.

If ours >> base  -> overtraining erased Turkish pronunciation -> retrain gentle / pick earlier ckpt.
If ours ~= base  -> pronunciation is fine; the perceived "garble" is timbre/brightness, not phonemes.

  .venv-tts\\Scripts\\python.exe training\\tts_cer.py
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import soundfile as sf
import torch
from jiwer import cer

MODEL_DIR = "E:/youtuber-clone/models/xtts_speaker_v2"
BASE_DIR = "E:/youtuber-clone/models/xtts_base"
WAVS = "E:/youtuber-clone/data/tts/wavs"
SMP = "E:/youtuber-clone/data/tts/_sample"
LANG = "tr"
SENTS = [
    "Bugün hava çok güzel, dışarı çıkıp biraz yürüyüş yapalım.",
    "Türkiye'nin başkenti Ankara'dır ve en büyük şehri İstanbul'dur.",
    "Ekonomik durum gittikçe kötüleşiyor, herkes bundan şikayetçi.",
    "Bu meseleyi yıllardır konuşuyoruz ama kimse çözüm bulamıyor.",
    "Siyasetçiler her zaman aynı şeyleri söyler ama hiçbir şey değişmez.",
    "Geçen hafta arkadaşlarımla buluştuk ve uzun uzun sohbet ettik.",
]


def norm(s: str) -> str:
    return re.sub(r"[^\wçğıöşü ]", "", s.lower()).strip()


def load_xtts(model_pth):
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    cfg = XttsConfig()
    cfg.load_json(f"{MODEL_DIR}/config.json")
    m = Xtts.init_from_config(cfg)
    m.load_checkpoint(cfg, checkpoint_path=model_pth, vocab_path=f"{MODEL_DIR}/vocab.json", use_deepspeed=False)
    if torch.cuda.is_available():
        m.cuda()
    return m


def synth_all(model_pth, tag, refs):
    m = load_xtts(model_pth)
    gpt, spk = m.get_conditioning_latents(audio_path=refs, gpt_cond_len=30, sound_norm_refs=True)
    for i, s in enumerate(SENTS):
        res = m.inference(s, LANG, gpt, spk, temperature=0.7, repetition_penalty=2.0)
        sf.write(f"{SMP}/_cer_{tag}_{i}.wav", np.asarray(res["wav"], dtype=np.float32), 24000)
    del m
    torch.cuda.empty_cache()


def main():
    os.makedirs(SMP, exist_ok=True)
    os.makedirs(BASE_DIR, exist_ok=True)
    base_pth = f"{BASE_DIR}/model.pth"
    if not os.path.exists(base_pth):
        from huggingface_hub import hf_hub_download
        print("downloading base XTTS model.pth (~1.9GB)...")
        hf_hub_download("coqui/XTTS-v2", "model.pth", local_dir=BASE_DIR)

    refs = sorted(glob.glob(f"{WAVS}/*.wav"))[:4]
    print("synth OURS..."); synth_all(f"{MODEL_DIR}/best_model.pth", "ours", refs)
    print("synth BASE..."); synth_all(base_pth, "base", refs)

    from faster_whisper import WhisperModel
    w = WhisperModel("large-v3", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")

    results = {}
    for tag in ("ours", "base"):
        cers = []
        print(f"\n=== {tag.upper()} ===")
        for i, s in enumerate(SENTS):
            segs, _ = w.transcribe(f"{SMP}/_cer_{tag}_{i}.wav", language="tr")
            hyp = " ".join(seg.text for seg in segs).strip()
            c = cer(norm(s), norm(hyp))
            cers.append(c)
            print(f"  CER {c:.2f} | {hyp[:72]}")
        results[tag] = float(np.mean(cers))

    print(f"\n{'='*50}\nMEAN CER   ours={results['ours']:.3f}   base={results['base']:.3f}")
    r = results["ours"] / max(results["base"], 1e-3)
    if results["ours"] > 0.25 and r > 1.6:
        print("-> OURS much worse: overtraining erased Turkish. Retrain gentle / earlier ckpt.")
    elif results["ours"] <= 0.18:
        print("-> OURS is intelligible: NOT a phoneme problem — the 'garble' is timbre/brightness.")
    else:
        print("-> moderate degradation; gentle re-finetune likely helps.")


if __name__ == "__main__":
    main()
