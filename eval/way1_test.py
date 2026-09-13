"""Full Way-1 (no-retrain) test: retrieve his real spans from the clean index, feed them to the
voice model, and compare BARE vs GROUNDED. Proves whether no-retrain grounding makes the model
restate HIS actual take on a topic he covered.

  .venv\\Scripts\\python.exe -m eval.way1_test "Mutlak butlan hakkında ne düşünüyorsun?"
"""
from __future__ import annotations

import argparse

from pipeline.config import load_settings
from rag.ask_grounded import grounded_answer
from rag.retrieve import retrieve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--collection", default="speaker_clean")
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--model", default="speaker-qwen3-run2-ep2")
    ap.add_argument("--out", default="data/eval/way1_out.txt")
    args = ap.parse_args()

    s = load_settings()
    r = s.rag
    hits = retrieve(args.question, store_path=str(r.store_path), collection=args.collection,
                    embedder_name=r.embedder, reranker_name=r.reranker,
                    top_k=r.retrieve_top_k, top_n=args.top_n)
    spans = [h.chunk.text for h in hits]
    top = hits[0].rerank_score if hits else float("nan")
    bare = grounded_answer(args.question, spans=[], model=args.model)
    grounded = grounded_answer(args.question, spans=spans, model=args.model)
    out = (f"Q: {args.question}\nretrieved {len(spans)} spans from {args.collection} "
           f"(top rerank {top:.2f})\n\n=== BARE (no context) ===\n{bare}\n\n"
           f"=== GROUNDED (his real spans from speaker_clean) ===\n{grounded}\n")
    open(args.out, "w", encoding="utf-8").write(out)
    print("written", args.out)


if __name__ == "__main__":
    main()
