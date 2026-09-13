"""Stage C of the broad sweep: turn the LLM-confirmed garbles into a DEDUPED audio queue.

Per the 'confirm-once' rule: group confirmed garbles by their (normalised) garble token and emit ONE
representative audio clip per UNIQUE garble — so the human ears each distinct mistake once, and the
global correction map fixes every other occurrence. Skips tokens already decided / already queued.

Reads the Stage-B workflow result + data/corrections/.detect/index.json (cand_id -> segment).
Writes data/corrections/broad_queue.jsonl and prints the deduped count by confidence.

Run:  .venv\\Scripts\\python.exe -m tools.build_broad_queue <workflow_output_file>
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

CORR = Path("data/corrections")
CONF_RANK = {"high": 0, "med": 1, "low": 2}


def _audio_path(video_id: str) -> str | None:
    for base in ("data/audio", "data/audio_flac"):
        for ext in (".wav", ".flac"):
            cand = Path(base) / f"{video_id}{ext}"
            if cand.exists():
                return str(cand)
    return None


def _norm(tok: str) -> str:
    return unicodedata.normalize("NFC", tok.strip().strip("'’.,!?:;()"))


def _load_result(path: str) -> list[dict]:
    raw = Path(path).read_text(encoding="utf-8")
    i, j = raw.find("{"), raw.rfind("}")
    obj = json.loads(raw[i:j + 1])
    res = obj.get("result", obj)
    if isinstance(res, str):
        res = json.loads(res)
    return res["garbles"]


def main() -> None:
    out_file = sys.argv[1]
    garbles = _load_result(out_file)
    index = json.loads((CORR / ".detect" / "index.json").read_text(encoding="utf-8"))
    by_cand = {c["cand_id"]: c for c in index}

    decided = set()
    if (CORR / "decisions.jsonl").exists():
        for l in (CORR / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
            if l.strip():
                decided.add(_norm(json.loads(l).get("token", "")))
    existing = set()
    if (CORR / "queue.jsonl").exists():
        for l in (CORR / "queue.jsonl").read_text(encoding="utf-8").splitlines():
            if l.strip():
                existing.add(_norm(json.loads(l).get("token", "")))

    # group by unique garble token -> keep the highest-confidence, audio-backed representative
    best: dict[str, dict] = {}
    doc_cache: dict[str, list] = {}
    for g in garbles:
        cand = by_cand.get(g["cand_id"])
        if not cand:
            continue
        ap = _audio_path(cand["video_id"])
        if ap is None:
            continue
        for tok in g.get("garble_tokens", []):
            key = _norm(tok)
            if not key or key in decided or key in existing:
                continue
            rank = CONF_RANK.get(g.get("confidence", "low"), 2)
            if key in best and CONF_RANK.get(best[key]["confidence"], 2) <= rank:
                continue
            vid = cand["video_id"]
            if vid not in doc_cache:
                p = Path("data/clean") / f"{vid}.json"
                doc_cache[vid] = json.loads(p.read_text(encoding="utf-8")).get("segments", []) if p.exists() else []
            segs = doc_cache[vid]
            si = cand["seg_index"]
            pos = cand["text"].find(tok)
            if pos < 0:
                pos = cand["text"].lower().find(key.lower())
            best[key] = {
                "token": tok, "suggest": g.get("suggestion", ""), "confidence": g.get("confidence", "low"),
                "verify": False, "video_id": vid, "title": cand.get("title", ""),
                "seg_index": si, "start": cand["start"], "end": cand["end"], "char_pos": pos,
                "text": cand["text"],
                "prev_text": (segs[si - 1].get("text") or "") if 0 < si <= len(segs) - 1 else "",
                "next_text": (segs[si + 1].get("text") or "") if 0 <= si + 1 < len(segs) else "",
                "avg_logprob": cand.get("avg_logprob"), "audio": ap,
            }

    rows = sorted(best.values(), key=lambda r: CONF_RANK.get(r["confidence"], 2))
    out = CORR / "broad_queue.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    byc = {"high": 0, "med": 0, "low": 0}
    for r in rows:
        byc[r["confidence"]] = byc.get(r["confidence"], 0) + 1
    print(f"confirmed garble segments (raw)   : {len(garbles)}")
    print(f"unique garble tokens (deduped)    : {len(best)}")
    print(f"  by confidence: high {byc['high']} / med {byc['med']} / low {byc['low']}")
    print(f"already decided/queued, skipped   : counted out")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
