"""Tokenizer-efficiency report across candidate base models.

CLAUDE.md defers the base-model decision to "end of Phase 1, based on tokenizer
efficiency on his transcripts". This tool runs that comparison.

Reads ``data/transcribe/*.json``, concatenates the transcribed text per video,
runs each candidate tokenizer on it, and reports:
  - chars
  - tokens
  - chars/token (higher = more efficient encoding for this corpus)

Candidate tokenizers (Turkish text):
  - Qwen2.5 (7B / 14B share tokenizer)
  - Mistral-7B-v0.3 (with non-gated fallback)
  - Llama-3.1-8B (with non-gated fallback)
  - Gemma-2 (extra reference point)

Gated repos fall back to community mirrors automatically.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from pipeline.config import load_settings

log = logging.getLogger("pipeline.tokenizer_report")

# (display_name, [candidate HF repos to try in order])
TOKENIZER_CANDIDATES: list[tuple[str, list[str]]] = [
    ("Qwen2.5-7B/14B", ["Qwen/Qwen2.5-7B"]),
    ("Mistral-7B-v0.3", ["mistralai/Mistral-7B-v0.3", "unsloth/mistral-7b-v0.3"]),
    ("Llama-3.1-8B", ["meta-llama/Llama-3.1-8B", "NousResearch/Meta-Llama-3.1-8B"]),
    ("Gemma-2-9B", ["google/gemma-2-9b", "unsloth/gemma-2-9b"]),
]


def _try_load_tokenizer(repos: list[str]):
    """Try each repo; return (repo_loaded, tokenizer) or (None, None)."""
    from transformers import AutoTokenizer

    last_err = None
    for r in repos:
        try:
            tok = AutoTokenizer.from_pretrained(r, trust_remote_code=False)
            return r, tok
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log.debug("tokenizer %s failed: %s", r, exc)
    return None, None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()

    transcripts: dict[str, str] = {}
    for jp in sorted(settings.paths.transcribe_dir.glob("*.json")):
        d = json.loads(jp.read_text(encoding="utf-8"))
        text = "\n".join(s["text"] for s in d["segments"] if s.get("text"))
        transcripts[d["video_id"]] = text

    if not transcripts:
        print("No transcripts found in data/transcribe/", file=sys.stderr)
        return 1

    total_chars = sum(len(t) for t in transcripts.values())
    print(f"\nCorpus: {len(transcripts)} videos, {total_chars:,} chars total\n")

    print(f"{'Tokenizer':<22}  {'Repo loaded':<36}  {'tokens':>10}  {'chars/tok':>10}  {'tokens/min audio':>17}")
    print("-" * 100)

    # Per-video time totals (for tokens/min audio later)
    audio_minutes: dict[str, float] = {}
    for jp in sorted(settings.paths.transcribe_dir.glob("*.json")):
        d = json.loads(jp.read_text(encoding="utf-8"))
        audio_minutes[d["video_id"]] = sum(s["duration_s"] for s in d["segments"]) / 60.0
    total_min = sum(audio_minutes.values())

    results: list[dict] = []
    for display_name, repos in TOKENIZER_CANDIDATES:
        repo, tok = _try_load_tokenizer(repos)
        if tok is None:
            print(f"{display_name:<22}  {'(all repos failed/gated)':<36}")
            continue
        # encode each video separately to avoid cross-doc artifacts; sum tokens
        per_vid_tokens = {}
        for vid, text in transcripts.items():
            ids = tok.encode(text, add_special_tokens=False)
            per_vid_tokens[vid] = len(ids)
        total_tok = sum(per_vid_tokens.values())
        ct = total_chars / total_tok if total_tok else 0
        tpm = total_tok / total_min if total_min else 0
        print(f"{display_name:<22}  {repo:<36}  {total_tok:>10,}  {ct:>10.2f}  {tpm:>17,.0f}")
        results.append({
            "display_name": display_name,
            "repo": repo,
            "total_tokens": total_tok,
            "chars_per_token": round(ct, 4),
            "tokens_per_min_audio": round(tpm, 1),
            "per_video_tokens": per_vid_tokens,
        })

    print()
    print(f"Total transcribed audio: {total_min:.1f} min  ({total_min/60:.2f} h)")
    print(f"Total chars: {total_chars:,}")
    print()
    if results:
        best = max(results, key=lambda r: r["chars_per_token"])
        print(f"Most efficient encoder for this Turkish corpus: **{best['display_name']}**  ({best['chars_per_token']} chars/token)")
        worst = min(results, key=lambda r: r["chars_per_token"])
        if best is not worst:
            ratio = best["chars_per_token"] / worst["chars_per_token"]
            print(f"vs least efficient ({worst['display_name']} at {worst['chars_per_token']}) → {ratio:.2f}x compression advantage")

    out = settings.paths.data_dir / "tokenizer_report.json"
    out.write_text(json.dumps({
        "total_chars": total_chars,
        "total_audio_min": round(total_min, 2),
        "per_video_chars": {v: len(t) for v, t in transcripts.items()},
        "per_video_audio_min": {v: round(m, 2) for v, m in audio_minutes.items()},
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
