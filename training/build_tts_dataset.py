"""Build a 24 kHz LJSpeech-style TTS dataset from the cleanest videos.

Re-downloads each approved video's audio at full quality, resamples to 24 kHz mono (the TTS
master rate — the pipeline's 16 kHz WAVs are ASR-only), and cuts it into clips using the
TTS-grade segment timestamps already computed in data/clean/{vid}.json. No re-transcription,
no re-running VAD/diarize/speaker-ID — we reuse the timestamps + transcripts we already have.

Reads:  data/tts/candidates.txt   (one video_id per line; '#' comments allowed)
Writes: data/tts/wavs/{vid}_{i:04d}.wav   (24 kHz mono PCM, 3-15 s)
        data/tts/metadata.csv             (LJSpeech: name|text|text)

Idempotent: a video whose clips already exist is skipped. Profanity is preserved (never filtered).

  .venv\\Scripts\\python.exe -m training.build_tts_dataset
  .venv\\Scripts\\python.exe -m training.build_tts_dataset --list data/tts/candidates.txt
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf
from yt_dlp import YoutubeDL

from pipeline.config import load_settings
from pipeline.download.runner import _EJS_REMOTE_COMPONENTS, _resolve_cookiefile

SR = 22_050  # XTTS conditioning rate (research: feed 22050 so the loader doesn't cheap-resample 24k->22k)

# TTS-grade segment filter (XTTS recipe: 3-11s, terminal punctuation, <=225 char).
MIN_DUR, MAX_DUR = 3.0, 11.0   # <3s breaks end-of-turn token (runaway); >11.6s silently truncated
MIN_SIM = 0.60
MAX_NO_SPEECH = 0.20
MIN_LOGPROB = -0.45
MIN_CHARS = 8
MAX_CHARS = 225   # XTTS hard-caps tr text at 226 chars; longer clips train with truncated text


def _read_list(path: Path) -> list[str]:
    vids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            vids.append(line)
    return vids


def _tts_segments(clean_json: Path) -> list[dict]:
    d = json.loads(clean_json.read_text(encoding="utf-8"))
    out = []
    for s in d.get("segments", []):
        if (MIN_DUR <= s.get("duration_s", 0) <= MAX_DUR
                and s.get("similarity", 0) >= MIN_SIM
                and s.get("no_speech_prob", 1) < MAX_NO_SPEECH
                and s.get("avg_logprob", -9) > MIN_LOGPROB
                and MIN_CHARS <= len((s.get("text") or "").strip()) <= MAX_CHARS
                and (s.get("text") or "").strip()[-1:] in ".!?"):   # terminal punctuation only
            out.append(s)
    return out


def _download_24k(vid: str, url: str, cookiefile: str | None, out_wav: Path) -> bool:
    """yt-dlp bestaudio -> ffmpeg 24 kHz mono PCM at out_wav. Returns True on success."""
    with tempfile.TemporaryDirectory(prefix=f"tts_{vid}_") as tmp:
        tmp_dir = Path(tmp)
        opts = {
            "format": "bestaudio/best",
            "outtmpl": str(tmp_dir / "%(id)s.%(ext)s"),
            "noplaylist": True,
            "remote_components": list(_EJS_REMOTE_COMPONENTS),
            "ignoreerrors": True, "retries": 10, "fragment_retries": 10,
            "concurrent_fragment_downloads": 1,  # serial fragments — avoids YouTube SSL session resets
            "http_chunk_size": 10 * 1024 * 1024,  # range-chunked: a dropped TLS session retries one 10MB chunk, not the whole file
            "sleep_interval": 2, "max_sleep_interval": 6,  # dodge rate-limiting on big sequential pulls
            "socket_timeout": 30, "quiet": True, "no_warnings": True,
        }
        if cookiefile:
            opts["cookiefile"] = cookiefile
        with YoutubeDL(opts) as ydl:
            try:
                info = ydl.extract_info(url, download=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  download failed: {exc}")
                return False
        if info is None:
            return False
        cands = list(tmp_dir.glob(f"{vid}.*"))
        if not cands:
            print("  no audio file produced")
            return False
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-i", str(cands[0]),
               "-ac", "1", "-af", "aresample=resampler=soxr:precision=28",  # HQ resample (not cheap/linear)
               "-ar", str(SR), "-c:a", "pcm_s16le", str(out_wav)]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"  ffmpeg failed: {exc}")
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a 24 kHz LJSpeech TTS set from cleanest videos.")
    ap.add_argument("--list", default="data/tts/candidates.txt")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    s = load_settings(args.config)
    data_dir = s.paths.data_dir
    tts_dir = data_dir / "tts"
    wav_dir = tts_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = tts_dir / "metadata.csv"
    cookiefile = _resolve_cookiefile(s.download.cookies_file)

    vids = _read_list(Path(args.list))
    print(f"{len(vids)} videos in list | cookies: {cookiefile or 'none'} | -> {tts_dir}")

    manifest_lines: list[str] = []
    if meta_csv.exists():
        manifest_lines = meta_csv.read_text(encoding="utf-8").splitlines()
    done_vids = {ln.split("|", 1)[0].rsplit("_", 1)[0] for ln in manifest_lines if "|" in ln}

    total_clips, total_secs = len(manifest_lines), 0.0
    for vi, vid in enumerate(vids, 1):
        if vid in done_vids:
            print(f"[{vi}/{len(vids)}] {vid} — already extracted, skip")
            continue
        clean_json = data_dir / "clean" / f"{vid}.json"
        if not clean_json.exists():
            print(f"[{vi}/{len(vids)}] {vid} — no clean json, skip")
            continue
        segs = _tts_segments(clean_json)
        if not segs:
            print(f"[{vi}/{len(vids)}] {vid} — 0 TTS-grade segments, skip")
            continue

        meta_json = data_dir / "meta" / f"{vid}.json"
        url = "https://www.youtube.com/watch?v=" + vid
        if meta_json.exists():
            url = json.loads(meta_json.read_text(encoding="utf-8")).get("webpage_url") or url

        print(f"[{vi}/{len(vids)}] {vid} — downloading ({len(segs)} clips)...")
        full = wav_dir.parent / f"_full_{vid}.wav"
        ok = False
        for attempt in range(3):   # YouTube SSL session resets are intermittent — retry the whole pull
            if _download_24k(vid, url, cookiefile, full):
                ok = True
                break
            print(f"    attempt {attempt + 1}/3 failed, retrying...")
        if not ok:
            print(f"    GAVE UP on {vid} (SSL/network) — re-run later or fetch on Colab")
            continue
        audio, sr = sf.read(str(full))
        if audio.ndim > 1:
            audio = audio[:, 0]
        n = 0
        for i, seg in enumerate(segs):
            a, b = int(seg["start"] * sr), int(seg["end"] * sr)
            clip = audio[a:b]
            if len(clip) < int(MIN_DUR * sr):
                continue
            name = f"{vid}_{i:04d}"
            sf.write(str(wav_dir / f"{name}.wav"), clip, sr, subtype="PCM_16")
            text = (seg["text"] or "").strip().replace("|", " ").replace("\n", " ")
            manifest_lines.append(f"{name}|{text}|{text}")
            total_secs += len(clip) / sr
            n += 1
        full.unlink(missing_ok=True)
        total_clips += n
        # flush manifest after each video so a crash doesn't lose progress
        meta_csv.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        print(f"    -> {n} clips written ({total_clips} total)")

    print(f"\nDONE: {total_clips} clips, ~{total_secs/3600:.2f} h new audio -> {meta_csv}")
    print("Next: ear-verify (listen to a few clips per video; delete any with music/noise), then fine-tune.")


if __name__ == "__main__":
    main()
