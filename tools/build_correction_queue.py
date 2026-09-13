"""Build a review queue from user-supplied ASR correction candidates.

Configure CANDIDATES using your own reviewed audio; no real-corpus seed list ships.
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline.config import load_settings

# (wrong_token, suggested_fix_or_empty, verify_only)
#   verify_only=True  -> probably his real word; we just want the human to confirm & whitelist.
CANDIDATES: list[tuple[str, str, bool]] = []

MAX_SEG_PER_TOKEN = 3
PAD = 0.35  # seconds of padding added by the UI when slicing audio


def _audio_path(video_id: str) -> str | None:
    """16 kHz wav primary, flac secondary."""
    for base in ("data/audio", "data/audio_flac"):
        for ext in (".wav", ".flac"):
            cand = Path(base) / f"{video_id}{ext}"
            if cand.exists():
                return str(cand)
    return None


def main() -> None:
    s = load_settings()
    clean_dir = s.paths.clean_dir

    # index every segment once
    docs = []
    for fp in sorted(clean_dir.glob("*.json")):
        if fp.name.endswith(".review.json"):
            continue
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        docs.append((doc.get("video_id") or fp.stem, doc.get("title") or "", doc.get("segments", [])))

    rows = []
    for wrong, suggest, verify in CANDIDATES:
        wl = wrong.lower()
        found = 0
        for vid, title, segs in docs:
            if found >= MAX_SEG_PER_TOKEN:
                break
            for i, seg in enumerate(segs):
                if found >= MAX_SEG_PER_TOKEN:
                    break
                text = seg.get("text") or ""
                pos = text.lower().find(wl)
                if pos < 0:
                    continue
                ap = _audio_path(vid)
                if ap is None:
                    continue  # no audio to play — skip this instance
                rows.append({
                    "token": wrong,
                    "suggest": suggest,
                    "verify": verify,
                    "video_id": vid,
                    "title": title,
                    "seg_index": i,
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "char_pos": pos,
                    "text": text,
                    "prev_text": (segs[i - 1].get("text") or "") if i > 0 else "",
                    "next_text": (segs[i + 1].get("text") or "") if i + 1 < len(segs) else "",
                    "avg_logprob": seg.get("avg_logprob"),
                    "audio": ap,
                })
                found += 1
        if found == 0:
            print(f"  (no audio-backed segment found for {wrong!r})")

    out_dir = Path("data/corrections")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "queue.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"\nqueue: {len(rows)} segments across {len({r['token'] for r in rows})} tokens -> {out}")


if __name__ == "__main__":
    main()
