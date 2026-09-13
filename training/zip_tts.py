"""Re-zip the curated/reviewed TTS set into data/tts.zip for the Colab XTTS notebook.

Layout the notebook expects: metadata.csv at the zip root + wavs/{name}.wav.
Only wavs referenced in metadata.csv are included (drops orphans), and total duration is
reported so we can confirm we still have enough data (XTTS-v2 wants ~60-90+ min cleanest).

  .venv-tts\\Scripts\\python.exe training\\zip_tts.py
"""
from __future__ import annotations

import os
import zipfile

import soundfile as sf

TTS = "E:/youtuber-clone/data/tts"
META = f"{TTS}/metadata.csv"
OUT = "E:/youtuber-clone/data/tts.zip"


def main() -> None:
    names = []
    for ln in open(META, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            names.append(ln.split("|", 1)[0])

    total = 0.0
    missing = []
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.write(META, "metadata.csv")
        for n in names:
            wav = f"{TTS}/wavs/{n}.wav"
            if not os.path.exists(wav):
                missing.append(n)
                continue
            info = sf.info(wav)
            total += info.frames / info.samplerate
            z.write(wav, f"wavs/{n}.wav")

    nvid = len({n.rsplit("_", 1)[0] for n in names})
    size_mb = os.path.getsize(OUT) / 1e6
    print(f"zipped: {len(names) - len(missing)} clips / {total/60:.1f} min "
          f"across {nvid} videos -> {OUT} ({size_mb:.0f} MB)")
    if missing:
        print(f"WARNING: {len(missing)} metadata rows had no wav: {missing[:8]}")


if __name__ == "__main__":
    main()
