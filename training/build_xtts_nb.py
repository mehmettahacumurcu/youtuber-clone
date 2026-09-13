"""Generate training/xtts_finetune.ipynb — a Colab notebook that fine-tunes XTTS-v2 on Speaker's voice.

Based on the official Coqui recipe (recipes/ljspeech/xtts_v2/train_gpt_xtts.py), adapted for Colab +
our data. Input is the LJSpeech-format set built by training/build_tts_dataset.py
(data/tts/wavs/*.wav 24 kHz + data/tts/metadata.csv as `name|text|text`).

  .venv\\Scripts\\python.exe training/build_xtts_nb.py
"""
from __future__ import annotations

import json
from pathlib import Path

CELLS: list[dict] = []


def md(src: str) -> None:
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)})


def code(src: str) -> None:
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.splitlines(keepends=True)})


md("""# Speaker XTTS-v2 voice fine-tune (Colab)

Fine-tunes **XTTS-v2** on his isolated 24 kHz clips. Use an **A100/L4** runtime (T4 works but slow).
Upload `tts.zip` (built locally by `training/build_tts_dataset.py`) when asked. Output: a fine-tuned
XTTS checkpoint you download to `models/` and run locally.""")

md("""## 1. Install coqui-tts  →  THEN restart the runtime
`coqui-tts` rebuilds numpy, which corrupts the running kernel (`numpy.char` / sanity-check error).
That's expected: after this cell finishes, **Runtime → Restart session**, then run **cell 1b** (do
NOT re-run this install). deepspeed is intentionally omitted (inference-only, slow source build).""")
code(r"""# Pin the EXACT validated combo: coqui-tts 0.26.2 + transformers 4.48.3. (Newer coqui-tts 0.27.5
# demands transformers>=4.57 -> 'is_torchcodec_available' import error; older transformers dropped
# 'isin_mps_friendly'. 0.26.2 + 4.48.3 is the stable island.)
!pip install -q coqui-tts==0.26.2 "transformers==4.48.3"
print("\n\n>>> INSTALL DONE. Now: Runtime -> Restart session, then run cell 1b below (skip this cell).")""")
md("### 1b. Run this AFTER restarting (import only — verifies the env)")
code(r"""import torch, TTS
print("torch", torch.__version__, "| TTS", TTS.__version__, "| cuda", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU ONLY — stop, pick a GPU runtime")""")

md("""## 2. Upload data
Run, then upload **`tts.zip`** (it has `wavs/` + `metadata.csv`). Unzips to `/content/tts/`.""")
code(r"""import os, zipfile
if not os.path.exists("/content/tts/metadata.csv"):
    # Prefer a tts.zip already uploaded via the Files panel (faster than the upload widget).
    if os.path.exists("/content/tts.zip"):
        z = "/content/tts.zip"
    else:
        from google.colab import files
        z = next(iter(files.upload()))   # pick tts.zip
    with zipfile.ZipFile(z) as zf: zf.extractall("/content/tts")
# normalize: ensure metadata.csv + wavs/ sit directly under /content/tts
import glob
if not os.path.exists("/content/tts/metadata.csv"):
    hit = glob.glob("/content/tts/**/metadata.csv", recursive=True)
    assert hit, "metadata.csv not found in the zip"
    base = os.path.dirname(hit[0])
    if base != "/content/tts":
        os.system(f'cp -r "{base}"/* /content/tts/')
n = sum(1 for _ in open("/content/tts/metadata.csv", encoding="utf-8"))
nw = len(glob.glob("/content/tts/wavs/*.wav"))
print(f"clips in metadata: {n} | wav files: {nw}")
assert nw > 200, "too few wavs — re-zip data/tts and re-upload"
""")

md("## 3. Download the XTTS-v2 base checkpoint + DVAE (fine-tune starts from these)")
code(r"""import os
from huggingface_hub import hf_hub_download   # reliable; coqui/XTTS-v2 is public, no auth
OUT = "/content/xtts_base"; os.makedirs(OUT, exist_ok=True)
for fn in ["model.pth", "config.json", "vocab.json", "dvae.pth", "mel_stats.pth"]:
    if not os.path.exists(f"{OUT}/{fn}"):
        print("downloading", fn); hf_hub_download(repo_id="coqui/XTTS-v2", filename=fn, local_dir=OUT)
DVAE = f"{OUT}/dvae.pth"; MEL = f"{OUT}/mel_stats.pth"
TOKENIZER = f"{OUT}/vocab.json"; XTTS_CKPT = f"{OUT}/model.pth"
print("base files ready:", os.listdir(OUT))""")

md("## 4. Config — fine-tune GPT on Turkish, conservative for Colab")
code(r"""from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig   # moved here in newer coqui-tts
from TTS.tts.datasets import load_tts_samples

RUN_NAME = "speaker_xtts_v2"
OUT_PATH = "/content/run"
LANG = "tr"

dataset = BaseDatasetConfig(formatter="ljspeech", dataset_name="speaker",
                            path="/content/tts", meta_file_train="metadata.csv", language=LANG)

# XTTS audio rates: GPT conditioning 22050, HiFiGAN output 24000 (our wavs are 24k -> fine).
audio = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)

model_args = GPTArgs(
    max_conditioning_length=132300, min_conditioning_length=66150,
    max_wav_length=255995,          # ~11.6s @22050 — our clips are <=15s, loader trims
    max_text_length=200,
    mel_norm_file=MEL, dvae_checkpoint=DVAE, xtts_checkpoint=XTTS_CKPT, tokenizer_file=TOKENIZER,
    gpt_num_audio_tokens=1026, gpt_start_audio_token=1024, gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True, gpt_use_perceiver_resampler=True,
)

BATCH = 8            # A100: 6-8 (T4: 2-3)
GRAD_ACUMM = 4       # GENTLE recipe: 30ep/lr1e-5/ga1 OVERTRAINED (good timbre, broken prosody/stutter).
                     # ga4 (eff-batch 32) + lr5e-6 + 12ep = slower, so timbre forms before prosody breaks.
config = GPTTrainerConfig(
    run_name=RUN_NAME, project_name="speaker", output_path=OUT_PATH,
    model_args=model_args, audio=audio,
    epochs=12, batch_size=BATCH, batch_group_size=48, eval_batch_size=BATCH,
    save_step=115, save_n_checkpoints=12, save_checkpoints=True,   # ~1 ckpt/epoch, keep all -> pick the sweet spot
    print_step=50, plot_step=200, log_model_step=1000, eval_split_max_size=64,
    optimizer="AdamW", optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
    lr=5e-6, lr_scheduler="MultiStepLR",   # gentle LR; milestones never fire on a small set (flat)
    lr_scheduler_params={"milestones": [900000, 2700000, 5400000], "gamma": 0.5, "last_epoch": -1},
    test_sentences=[],
)   # NB: grad_accum_steps goes on TrainerArgs (cell 5). DON'T just take best_model.pth (min eval loss
    # = likely overtrained); use the §6b compare cell to pick the earliest checkpoint with good timbre.
print("effective batch ~", BATCH * GRAD_ACUMM, "| GENTLE: 12ep, lr5e-6, save every epoch")""")

md("## 5. Train")
code(r"""from trainer import Trainer, TrainerArgs
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTTrainer

model = GPTTrainer.init_from_config(config)
train_samples, eval_samples = load_tts_samples([dataset], eval_split=True, eval_split_max_size=64,
                                               eval_split_size=0.05)   # ~5% held out for early-stop
print(f"train clips: {len(train_samples)} | eval clips: {len(eval_samples)}")
trainer = Trainer(
    TrainerArgs(restore_path=None, skip_train_epoch=False, start_with_eval=False,
                grad_accum_steps=GRAD_ACUMM),
    config, output_path=OUT_PATH, model=model,
    train_samples=train_samples, eval_samples=eval_samples,
)
trainer.fit()
print("best model:", trainer.best_loss, "->", OUT_PATH)""")

md("""## 6. Quick synthesis test
Loads the fine-tuned weights and speaks a sentence in his voice, conditioned on a real clip of his.""")
code(r"""import glob, torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

run_dir = sorted(glob.glob(f"{OUT_PATH}/{RUN_NAME}-*"))[-1]
ckpt = f"{run_dir}/best_model.pth"
cfg = XttsConfig(); cfg.load_json(f"/content/xtts_base/config.json")
m = Xtts.init_from_config(cfg)
m.load_checkpoint(cfg, checkpoint_path=ckpt, vocab_path=TOKENIZER, use_deepspeed=False)
m.cuda()

ref = sorted(glob.glob("/content/tts/wavs/*.wav"))[:6]   # a few of his clips as the voice reference
gpt_lat, spk_emb = m.get_conditioning_latents(audio_path=ref)
text = "Bak şimdi sana açık açık söyleyeyim, bu işin aslı tamamen başka."
out = m.inference(text, LANG, gpt_lat, spk_emb, temperature=0.7)
import soundfile as sf
sf.write("/content/speaker_sample.wav", out["wav"], 24000)
from IPython.display import Audio
Audio("/content/speaker_sample.wav")""")

md("""## 6b. Checkpoint compare — pick the TIMBRE-vs-PROSODY sweet spot
best_model.pth (min eval loss) is usually OVERtrained (good voice, stuttery prosody). This synthesizes
the stutter-prone profane line from EVERY epoch checkpoint so you pick the EARLIEST one where the voice
is already his AND 'amına koyayım' comes out smooth.""")
code(r"""from google.colab import drive; drive.mount("/content/drive")
import glob, os, torch, soundfile as sf
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
run_dir = sorted(glob.glob(f"{OUT_PATH}/{RUN_NAME}-*"))[-1]
CMP = "/content/drive/MyDrive/speaker_models/xtts_cmp"; os.makedirs(CMP, exist_ok=True)
cks = sorted(glob.glob(f"{run_dir}/checkpoint_*.pth"), key=lambda p: int(p.split('_')[-1].split('.')[0]))
cks += [f"{run_dir}/best_model.pth"]
ref = sorted(glob.glob("/content/tts/wavs/*.wav"))[:4]
# 8 fixed probes (same at every epoch) covering: clean, profanity/stutter, his style, question,
# diacritics (ş/ç/ğ), numbers, names, long run-on. Listen across epochs to SEE where it overtrains.
texts = [
    "Türkiye'nin başkenti Ankara'dır ve en büyük şehri İstanbul'dur.",
    "Bu meseleyi yıllardır konuşuyoruz ama kimse dinlemiyor amına koyayım.",
    "Ya bak abi, bu işin aslı tamamen başka, sana açık açık anlatayım.",
    "Sence bu ülkede gerçekten adalet var mı, yoksa hepsi palavra mı?",
    "Şu çocuğun yaptığı şey gerçekten çok yanlıştı, değil mi?",
    "İki bin yirmi altı yılında işler iyice değişti arkadaşlar.",
    "Atatürk Cumhuriyet'i kurduğunda herkes buna inanmıyordu aslında.",
    "Geçen hafta arkadaşlarımla buluştuk, uzun uzun konuştuk, sonunda hiçbir şeye varamadık.",
]
cfg = XttsConfig(); cfg.load_json("/content/xtts_base/config.json")
for ck in cks:
    tag = os.path.basename(ck).replace('.pth', '')   # epoch tag (checkpoint_<step> / best_model)
    m = Xtts.init_from_config(cfg); m.load_checkpoint(cfg, checkpoint_path=ck, vocab_path=TOKENIZER, use_deepspeed=False); m.cuda()
    g, s = m.get_conditioning_latents(audio_path=ref, gpt_cond_len=30, sound_norm_refs=True)
    for j, t in enumerate(texts):
        out = m.inference(t, LANG, g, s, temperature=0.7, repetition_penalty=1.3)
        sf.write(f"{CMP}/{tag}_s{j}.wav", out["wav"], 24000)
    del m; torch.cuda.empty_cache(); print("synth", tag)
print("\n-> Download /MyDrive/speaker_models/xtts_cmp (8 sentences x every checkpoint).")
print("-> Listen across epochs: voice should sharpen then prosody (esp. s1 'amina koyayim') starts to")
print("   stutter when overtrained. Pick the EARLIEST tag where voice=his AND all 8 read smooth -> CHOSEN.")""")

md("## 7. Slim-save the CHOSEN checkpoint to Drive (~1.9GB, optimizer stripped)")
code(r"""CHOSEN = "best_model"   # <- after listening to §6b, set e.g. "checkpoint_575"
import glob, os, shutil, torch
run_dir = sorted(glob.glob(f"{OUT_PATH}/{RUN_NAME}-*"))[-1]
DST = "/content/drive/MyDrive/speaker_models/xtts_speaker_v3"; os.makedirs(DST, exist_ok=True)
ck = torch.load(f"{run_dir}/{CHOSEN}.pth", map_location="cpu")
torch.save({"model": ck["model"], "config": ck.get("config")}, f"{DST}/best_model.pth")  # slim, fast download
shutil.copy("/content/xtts_base/config.json", f"{DST}/config.json")
shutil.copy(TOKENIZER, f"{DST}/vocab.json")
print("saved CHOSEN", CHOSEN, "->", DST, os.listdir(DST),
      "|", round(os.path.getsize(f'{DST}/best_model.pth')/1e9, 2), "GB")""")

nb = {"cells": CELLS,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
out = Path(__file__).with_name("xtts_finetune.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {out} ({len(CELLS)} cells)")
