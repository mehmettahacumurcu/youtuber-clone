"""v4 abstention rows — teach in-voice "I haven't covered that" instead of invention.

Why: v3 has ZERO abstention rows (weak topics were dropped, not trained), so the model's only
learned move on unknown questions is confident invention. These rows are the closed-book
counterpart: bare question (exact trained format) -> short, profane-capable, in-voice decline.

Two modes:
  verify — score candidate bait questions against speaker_clean and KEEP only the clearly-uncovered
           (top rerank < floor). This guards the #1 failure mode: training an abstention on a
           topic he DID cover would teach false declines. Run with the UI/server stopped
           (single-writer Qdrant) and Ollama idle (BGE-M3 needs the VRAM).
  parse  — read the teacher workflow output (per bait: {"question","answer"}) -> bare-question
           chat rows at data/dataset/abstain_v4.jsonl.

  .venv\\Scripts\\python.exe -m training.build_v4_abstain verify <baits.jsonl>
  .venv\\Scripts\\python.exe -m training.build_v4_abstain parse <workflow_output.json>

baits.jsonl rows: {"question": "...", "category": "..."} (categories keep the set diverse:
precise-scalar bait, out-of-domain topic, personal-unknown, fake-plausible entity, ...).
"""
from __future__ import annotations

import collections
import json
import re
import sys

from rag.prompt import SYS

DD = "E:/youtuber-clone/data/dataset"
VERIFIED = f"{DD}/abstain_baits_verified.jsonl"
OUT = f"{DD}/abstain_v4.jsonl"
UNCOVERED_MAX = -0.5   # keep a bait only if its BEST hit reranks below this (clear non-coverage
                       # margin under gate_threshold=0.0; covered topics score 0..2.5+)

PROF = re.compile(r"\b(sik|am[ıi]na|amc[ıi]k|g[öo]t|pi[çc]|o[çc]\b|orospu|yarr?a[kğ]|pezevenk|"
                  r"kahpe|ibne|gavat|anan[ıi]|avrad[ıi]|sokay[ıi]m|siktir|yavşak|şerefsiz)", re.I)


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (t or "").lower())).strip()


def verify(baits_path: str) -> None:
    from pipeline.config import load_settings
    from rag.retrieve import retrieve

    s = load_settings()
    r = s.rag
    baits = [json.loads(l) for l in open(baits_path, encoding="utf-8") if l.strip()]
    kept, dropped = [], []
    print(f"{'top':>7s}  kept  question")
    for b in baits:
        q = (b.get("question") or "").strip()
        if not q:
            continue
        hits = retrieve(q, store_path=str(r.store_path), collection=r.collection,
                        embedder_name=r.embedder, reranker_name=r.reranker,
                        top_k=r.retrieve_top_k, top_n=3)
        top = max((h.rerank_score for h in hits), default=float("-inf"))
        row = {**b, "top_score": round(top, 3)}
        ok = top < UNCOVERED_MAX
        (kept if ok else dropped).append(row)
        print(f"{top:7.2f}  {'KEEP' if ok else 'drop'}  {q[:70]}")

    with open(VERIFIED, "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(x, ensure_ascii=False) for x in kept) + "\n")
    print(f"\n{len(kept)}/{len(baits)} baits verified uncovered (top < {UNCOVERED_MAX}) -> {VERIFIED}")
    if dropped:
        print(f"dropped {len(dropped)} (he covered them; training a decline would be a FALSE abstention):")
        for d in dropped[:10]:
            print(f"  {d['top_score']:6.2f}  {d['question'][:70]}")


def parse(path: str) -> None:
    blob = json.load(open(path, encoding="utf-8"))
    res = blob["result"] if isinstance(blob, dict) else blob
    if isinstance(res, str):
        res = json.loads(res)

    verified = {norm(json.loads(l)["question"]) for l in open(VERIFIED, encoding="utf-8") if l.strip()}
    rows, seen, bad = [], set(), collections.Counter()
    for p in res:
        q = (p.get("question") or "").strip()
        a = (p.get("answer") or "").strip()
        if not q or not a:
            bad["empty_field"] += 1
            continue
        nq = norm(q)
        if nq not in verified:
            bad["not_verified"] += 1     # never train a decline that didn't pass the coverage check
            continue
        if nq in seen:
            bad["dup_question"] += 1
            continue
        seen.add(nq)
        rows.append({"messages": [{"role": "system", "content": SYS},
                                  {"role": "user", "content": q},
                                  {"role": "assistant", "content": a}]})

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    prof = sum(1 for r in rows if PROF.search(r["messages"][-1]["content"]))
    print(f"{len(rows)} abstention rows -> {OUT} | profanity {prof}/{len(rows)}")
    if bad:
        print("rejected:", dict(bad))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "verify":
        verify(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "parse":
        parse(sys.argv[2])
    else:
        raise SystemExit(__doc__)
