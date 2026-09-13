"""Heuristic title-based speaker filtering. Calibrate format rules for your corpus.

Title labels alone do not establish speaker identity; use reference matching and
manual review before treating segments as speaker-only evidence.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


def _tnorm(title: str) -> str:
    """Turkish-safe casefold: dotted/dotless I -> i, strip combining marks, lower."""
    t = title.replace("İ", "i").replace("I", "i").replace("ı", "i")
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return t.lower()


# Multi-speaker (EXCLUDE) signals, checked first.
_RX_SPACE = re.compile(r"\bspace\b|konsey|halk mecl|altili masa")
_RX_RAMAZAN = re.compile(r"ramazan|sahur")
# guest sohbet ("X Sohbeti", "... ile sohbet"), soylesi, podcast, munazara/tartisma.
# NOTE: bare "... sohbet ediyor" (him solo) is intentionally NOT matched here.
_RX_GUESTCHAT = re.compile(r"sohbeti\b|ile sohbet|soylesi|podcast|munazara|tartisma")
_RX_DEBATE = re.compile(r"\bvs\.?\b")
_RX_INTERVIEW = re.compile(r"roportaj yap|sorulari(mizi)? cevap")
# Reaction / read-aloud (T2): plays someone else's audio and comments on it.
_RX_REACTION = re.compile(
    r"izliyor|izliyoruz|izledik|bakiyor|okuyor|inceliyor|dinliyor|roportajini|yorumluyor"
)
_RX_COSTREAM = re.compile(r"ortak yayin|konuk yayin|co-stream")


def classify_video(title: str) -> str:
    """Map a video TITLE to one of: 'T1', 'T2', 'EXCL'. Title-only, deterministic."""
    n = _tnorm(title)
    if _RX_SPACE.search(n):
        return "EXCL"
    if _RX_RAMAZAN.search(n):
        return "EXCL"
    if _RX_GUESTCHAT.search(n):
        return "EXCL"
    if re.search(r"@\w|&", title):  # collab credit in raw title
        return "EXCL"
    if _RX_DEBATE.search(n):
        return "EXCL"
    if _RX_INTERVIEW.search(n):
        return "EXCL"
    if _RX_REACTION.search(n) or _RX_COSTREAM.search(n):
        return "T2"
    return "T1"  # solo: documentary, stream-commentary, chat-stream, monologue


def select_videos(clean_dir: Path, meta_dir: Path, tiers=("T1",)) -> dict[str, str]:
    """Return {video_id: tier} for videos whose tier is in `tiers`."""
    clean_dir, meta_dir = Path(clean_dir), Path(meta_dir)
    out: dict[str, str] = {}
    for fp in sorted(clean_dir.glob("*.json")):
        if fp.name.endswith(".review.json"):
            continue
        doc = json.loads(fp.read_text(encoding="utf-8"))
        vid = doc.get("video_id") or fp.stem
        title = doc.get("title") or ""
        if not title:
            mp = meta_dir / f"{vid}.json"
            if mp.is_file():
                title = json.loads(mp.read_text(encoding="utf-8")).get("title", "")
        tier = classify_video(title)
        if tier in tiers:
            out[vid] = tier
    return out


def hisonly_segments(doc: dict, sim_floor: float = 0.0):
    """Filter one clean-video doc's segments. sim_floor default 0.0 = no extra cut.

    Raising sim_floor is NOT recommended for an opinion index: genuinely-his spans
    sit as low as 0.52 (verified on the 'mutlak butlan' takes), so a 0.60 floor
    deletes real opinions while removing no contamination. A mild 0.55 only shaves
    the bottom ~9% at near-zero benefit. Format selection does the real work.
    """
    return [s for s in doc.get("segments", []) if (s.get("similarity") or 0.0) >= sim_floor]


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify Speaker videos into his-only tiers and emit a manifest.")
    ap.add_argument("--clean-dir", default="data/clean")
    ap.add_argument("--meta-dir", default="data/meta")
    ap.add_argument("--out", default="data/dataset/hisonly_manifest.json",
                    help="Write {video_id: tier} manifest here.")
    args = ap.parse_args()
    clean_dir, meta_dir = Path(args.clean_dir), Path(args.meta_dir)

    manifest: dict[str, str] = {}
    stats = {t: {"vids": 0, "segs": 0, "chars": 0} for t in ("T1", "T2", "EXCL")}
    for fp in sorted(clean_dir.glob("*.json")):
        if fp.name.endswith(".review.json"):
            continue
        doc = json.loads(fp.read_text(encoding="utf-8"))
        vid = doc.get("video_id") or fp.stem
        title = doc.get("title") or ""
        if not title:
            mp = meta_dir / f"{vid}.json"
            if mp.is_file():
                title = json.loads(mp.read_text(encoding="utf-8")).get("title", "")
        tier = classify_video(title)
        manifest[vid] = tier
        st = stats[tier]
        segs = doc.get("segments") or []
        st["vids"] += 1
        st["segs"] += len(segs)
        st["chars"] += sum(len(s.get("text", "")) for s in segs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"manifest -> {out_path} ({len(manifest)} videos)")
    for t in ("T1", "T2", "EXCL"):
        s = stats[t]
        print(f"  {t:4s}: vids={s['vids']:4d} segs={s['segs']:6d} ~tok={int(s['chars']*0.292):,}")


if __name__ == "__main__":
    main()
