"""Step 1 of the opinion-training-data build: for each topic he covered, retrieve HIS real spans
from the clean index. Output feeds the Claude-teacher generation (step 2) that writes his-voice
grounded answers. Also reports coverage so weak topics (he barely covered) can be routed to
abstention instead of mined as if grounded.

Run with the UI stopped (single-writer Qdrant):
  .venv\\Scripts\\python.exe -m training.mine_topics
"""
from __future__ import annotations

import json

from pipeline.config import load_settings
from rag.retrieve import retrieve

TOPICS = "eval/probe_topics.json"
OUT = "data/dataset/opinion_spans.json"
TOP_N = 10   # mine generously — his historical takes are distributed; give the teacher enough material


def main() -> None:
    s = load_settings()
    r = s.rag
    topics = json.load(open(TOPICS, encoding="utf-8"))
    rows = []
    print(f"{'topic':28s}{'top':>7s}{'>1.0':>6s}  question")
    for t in topics:
        hits = retrieve(t["q"], store_path=str(r.store_path), collection="speaker_clean",
                        embedder_name=r.embedder, reranker_name=r.reranker,
                        top_k=r.retrieve_top_k, top_n=TOP_N)
        spans = [{"text": h.chunk.text, "video_id": h.chunk.video_id,
                  "title": h.chunk.title, "start": h.chunk.start,
                  "score": round(h.rerank_score, 3)} for h in hits]
        top = hits[0].rerank_score if hits else float("-inf")
        n_strong = sum(1 for h in hits if h.rerank_score >= r.gate_threshold)
        rows.append({"topic": t["topic"], "q": t["q"], "top_score": round(top, 3),
                     "n_strong": n_strong, "spans": spans})
        print(f"{t['topic']:28s}{top:7.2f}{n_strong:6d}  {t['q'][:46]}")

    json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    covered = sum(1 for x in rows if x["top_score"] >= r.gate_threshold)
    print(f"\n-> {OUT} | {covered}/{len(rows)} topics strongly covered (top>= {r.gate_threshold}); "
          f"the rest are weak -> abstention candidates.")


if __name__ == "__main__":
    main()
