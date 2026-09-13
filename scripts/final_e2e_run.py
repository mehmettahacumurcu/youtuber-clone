"""Final wrap-up E2E: drive the studio's own chat()+speak() headlessly over a fixed question set.

Phase 1 answers every question through one of Studio's explicit answer modes. Phase 2 loads
XTTS once (xtts_speaker_v2) and voices every answer.
Writes results/final_run/report.md + NN_*.wav for the user's review listen.

Run in the TTS env, from the repo root, with NO other studio running (ports + VRAM):
  .venv-tts\\Scripts\\python.exe scripts\\final_e2e_run.py
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import soundfile as sf

# Force UTF-8 on redirected stdout so emoji + Turkish chars don't crash on Windows cp1254
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import ui.speaker_studio as studio  # noqa: E402  (builds the Blocks tree; does not launch)

MODEL = "speaker-v5-a636:latest"
TTS_LABEL = "xtts_speaker_v2"
OUT = ROOT / "results" / "final_run"

# (question, answer-mode label, what this probes)
QUESTIONS = [
    ("Mutlak butlan meselesi hakkında ne düşünüyorsun?", "Grounded + answer anyway",
     "🎯 dedicated-video: mutlak butlan"),
    ("Dersimli KK olayı neydi, anlatsana.", "Grounded + answer anyway",
     "🎯 dedicated-video: Dersimli KK"),
    ("İmamoğlu davası hakkında ne düşünüyorsun?", "Grounded + answer anyway",
     "🎯 dedicated-video: İmamoğlu"),
    ("Adnan Menderes'in idamı hakkında ne düşünüyorsun?", "Grounded + answer anyway",
     "grounded: corpus topic, multi-video"),
    ("Menderes idam edilirken infazı yapan cellatın adı neydi?", "Grounded only",
     "strict abstention bait"),
    ("Dersim'e ilk giden gazetecinin adı neydi?", "Grounded only",
     "strict abstention bait"),
    ("Kuantum bilgisayarlar hakkında ne düşünüyorsun?", "Grounded only",
     "junk control: 🎯 must NOT fire"),
    ("Türk siyasetinde seni en çok sinirlendiren şey ne?", "Free mode",
     "open style probe"),
]


def wait_rag(timeout_s: int = 300) -> None:
    studio._ensure_retrieve_server()
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen("http://127.0.0.1:7670/health", timeout=3) as r:
                if json.loads(r.read()).get("ok"):
                    print("RAG ready", flush=True)
                    return
        except Exception:
            pass
        time.sleep(4)
    raise RuntimeError("retrieve server did not become ready")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wait_rag()

    print(f"\n=== phase 1: {len(QUESTIONS)} answers via {MODEL} ===", flush=True)
    rows = []
    for i, (q, answer_mode_label, probe) in enumerate(QUESTIONS, 1):
        t0 = time.time()
        retrieval_mode = "Clean transcripts + card hints"
        history, ctx_md, meta_html = studio.chat(
            q,
            [],
            MODEL,
            answer_mode_label,
            retrieval_mode,
            studio.DEFAULT_ANSWER_TOKEN_CAP,
        )
        answer = history[-1]["content"]
        wall = time.time() - t0
        mode = "unverified-fallback" if "unverified fallback" in (meta_html or "") else (
            "partial" if "kısmi kanıtlı" in (meta_html or "") else (
                "answerable" if "kanıtlı" in (meta_html or "") else (
                    "free" if answer_mode_label == "Free mode" else "fail-closed"
                )
            )
        )
        rows.append({
            "i": i,
            "q": q,
            "answer_mode": answer_mode_label,
            "probe": probe,
            "answer": answer,
            "mode": mode,
            "wall_s": round(wall),
            "ctx_md": ctx_md or "",
            "meta": meta_html or "",
        })
        print(f"[{i}/{len(QUESTIONS)}] {mode} {wall:5.0f}s  {q[:60]}", flush=True)

    print(f"\n=== phase 2: voicing {len(rows)} answers via {TTS_LABEL} ===", flush=True)
    assert TTS_LABEL in studio.TTS_OPTS, f"{TTS_LABEL} not in TTS_OPTS: {list(studio.TTS_OPTS)}"
    for row in rows:
        if row["answer"].startswith("[LLM error"):
            row["wav"] = ""
            continue
        t0 = time.time()
        sr, wav = studio.speak(row["answer"], TTS_LABEL)
        name = f"{row['i']:02d}_{re.sub(r'[^a-z0-9]+', '_', row['q'][:36].lower())}.wav"
        sf.write(str(OUT / name), wav, sr)
        row["wav"] = name
        print(f"[{row['i']}/{len(rows)}] tts {time.time() - t0:4.0f}s -> {name}", flush=True)

    lines = [
        f"# Final E2E run — {MODEL} + {TTS_LABEL} + evidence verifier",
        "",
        "Studio schema-v2 `chat()`/`speak()` path with explicit strict, fallback, and free policies.",
        "",
    ]
    for r in rows:
        lines += [f"## {r['i']}. {r['q']}", "",
                  f"- probe: {r['probe']}",
                  f"- answer policy: {r['answer_mode']} · result: {r['mode']}",
                  f"- LLM wall: {r['wall_s']}s · audio: `{r['wav'] or '—'}`",
                  f"- studio meta: {re.sub(r'<[^>]+>', '', r['meta']) or '—'}", "",
                  "> " + r["answer"].replace("\n", "\n> "), ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport: {OUT / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
