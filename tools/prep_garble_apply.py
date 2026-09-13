"""Process the extraction workflow output into (1) an AUTO-APPLY correction map with a corpus
safety-check + human-readable review list, and (2) the EAR-CHECK queue (ambiguous garbles routed
to the audio tool, prefilled with the minimal suggested fix).

AUTO = extraction flagged global_safe (wrong-form is a non-word) AND certain (no audio needed).
Safety: count each wrong-form's corpus occurrences; DEMOTE to ear-check if it appears often enough
to plausibly be a real word (could mean a blind replace would clobber legit text).

Run:  .venv\\Scripts\\python.exe -m tools.prep_garble_apply <extraction_output_file>
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

CORR = Path("data/corrections")
CONF_RANK = {"high": 0, "med": 1, "low": 2}
DEMOTE_FREQ = 8   # if a 'wrong' form occurs >= this many times, it might be a real word -> ear-check


def _load_rows(path: str) -> list[dict]:
    raw = Path(path).read_text(encoding="utf-8")
    i, j = raw.find("{"), raw.rfind("}")
    obj = json.loads(raw[i:j + 1])
    res = obj.get("result", obj)
    if isinstance(res, str):
        res = json.loads(res)
    return res["rows"]


def _norm(t: str) -> str:
    return unicodedata.normalize("NFC", t.strip().strip("'’.,!?:;()"))


def _audio_path(video_id: str) -> str | None:
    for base in ("data/audio", "data/audio_flac"):
        for ext in (".wav", ".flac"):
            c = Path(base) / f"{video_id}{ext}"
            if c.exists():
                return str(c)
    return None


def apply_fixes(text: str, fixes: list[dict]) -> str:
    for f in fixes:
        w, r = f["wrong"], f["right"]
        if w and w in text:
            text = text.replace(w, r)
    return text


def main() -> None:
    rows = _load_rows(sys.argv[1])
    index = json.loads((CORR / ".detect" / "index.json").read_text(encoding="utf-8"))
    by_cand = {c["cand_id"]: c for c in index}

    # load corpus once for occurrence counts + prev/next context
    docs: dict[str, list] = {}
    corpus_text = []
    for fp in sorted(Path("data/clean").glob("*.json")):
        if fp.name.endswith(".review.json"):
            continue
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        vid = doc.get("video_id") or fp.stem
        segs = doc.get("segments", [])
        docs[vid] = segs
        corpus_text.append(" ".join((s.get("text") or "") for s in segs))
    blob = "\n".join(corpus_text)

    def occ(wrong: str) -> int:
        return len(re.findall(rf"(?<!\w){re.escape(wrong)}(?!\w)", blob))

    decided, existing = set(), set()
    if (CORR / "decisions.jsonl").exists():
        for l in (CORR / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
            if l.strip():
                decided.add(_norm(json.loads(l).get("token", "")))
    if (CORR / "queue.jsonl").exists():
        for l in (CORR / "queue.jsonl").read_text(encoding="utf-8").splitlines():
            if l.strip():
                existing.add(_norm(json.loads(l).get("token", "")))

    auto_map: dict[str, str] = {}
    auto_review, demoted = [], []
    ear_rows = []
    for row in rows:
        fixes = row.get("fixes") or []
        if not fixes:
            continue
        cand = by_cand.get(row["cand_id"])
        if not cand:
            continue
        is_auto = row.get("global_safe") and row.get("certain")
        if is_auto:
            # safety: demote any fix whose wrong-form looks like a real (frequent) word
            row_ok, counts = True, {}
            for f in fixes:
                n = occ(f["wrong"])
                counts[f["wrong"]] = n
                if n >= DEMOTE_FREQ:
                    row_ok = False
            if row_ok:
                for f in fixes:
                    auto_map[f["wrong"]] = f["right"]
                    auto_review.append((f["wrong"], f["right"], counts[f["wrong"]], cand["text"]))
                continue
            demoted.append((row["cand_id"], fixes, counts))
        # ear-check route (ambiguous, low-conf, or demoted)
        tok = _norm(fixes[0]["wrong"])
        if not tok or tok in decided or tok in existing:
            continue
        ap = _audio_path(cand["video_id"])
        if ap is None:
            continue
        si = cand["seg_index"]
        segs = docs.get(cand["video_id"], [])
        ear_rows.append({
            "token": fixes[0]["wrong"], "suggest": apply_fixes(cand["text"], fixes),
            "suggest_text": apply_fixes(cand["text"], fixes), "fixes": fixes,
            "confidence": row.get("_confidence", "med"), "verify": False,
            "video_id": cand["video_id"], "title": cand.get("title", ""),
            "seg_index": si, "start": cand["start"], "end": cand["end"],
            "char_pos": cand["text"].find(fixes[0]["wrong"]), "text": cand["text"],
            "prev_text": (segs[si - 1].get("text") or "") if 0 < si <= len(segs) - 1 else "",
            "next_text": (segs[si + 1].get("text") or "") if 0 <= si + 1 < len(segs) else "",
            "avg_logprob": cand.get("avg_logprob"), "audio": ap,
        })

    # dedup ear queue by token
    seen, ear_dedup = set(), []
    for r in ear_rows:
        k = _norm(r["token"])
        if k in seen:
            continue
        seen.add(k)
        ear_dedup.append(r)

    (CORR / "auto_map.json").write_text(json.dumps(auto_map, ensure_ascii=False, indent=2), encoding="utf-8")
    (CORR / "broad_queue.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in ear_dedup) + "\n", encoding="utf-8")

    lines = ["AUTO-APPLY REVIEW  (wrong -> right   ·  Nx in corpus  ·  example)\n" + "=" * 70]
    for w, r, n, ex in sorted(auto_review, key=lambda x: -x[2]):
        lines.append(f"{w!r:28} -> {r!r:24}  {n:3}x   {ex[:70]}")
    (CORR / "AUTO_review.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"AUTO-APPLY fixes : {len(auto_map)}  -> data/corrections/auto_map.json")
    print(f"  (demoted to ear for safety: {len(demoted)})")
    print(f"EAR-CHECK queue  : {len(ear_dedup)} unique -> data/corrections/broad_queue.jsonl")
    print(f"review list      : data/corrections/AUTO_review.txt")


if __name__ == "__main__":
    main()
