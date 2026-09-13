"""Build speaker_cards — distilled idea-card RAG collection from the v3 teacher pairs.

Cards fix what raw ASR chunks do badly (fragmented takes, garble the model copies verbatim):
each card is ONE clean take in his voice. v0 needs NO LLM pass — the 2,428 v3 teacher pairs are
already one-take-per-question card bodies, and they inherit the his-only gate (the video packets
they were generated from were built tiers=("T1",)). This script:

  1. embeds question+answer (BGE-M3 dense) and merges near-duplicate takes (cosine >= DUP_COS)
     — the mechanical stand-in for LLM compaction; n_takes on the card = coverage signal
  2. mines provenance per card from the T1 speaker_clean index (top spans + rerank scores, with
     the per-video diversity cap) so cards cite real video_id/start exactly like chunks do
  3. attaches upload_date of the primary source (newest-take-wins can build on this later)

Profanity is COUNTED, never filtered (hard project constraint). Weak-support cards are kept and
marked, not dropped — the pairs came from his own packets, so weak support usually means a
retrieval miss, not invention — but the count is reported so a spike is visible.

Run with the studio + retrieve server STOPPED (local Qdrant is single-writer):
  .venv\\Scripts\\python.exe -m rag.build_cards
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from pipeline.config import load_settings
from rag.embed import get_embedder, get_reranker
from rag.store import VectorStore

PAIRS = "E:/youtuber-clone/data/dataset/voice_sft_v3_train.jsonl"
META_DIR = "E:/youtuber-clone/data/meta"
OUT = "E:/youtuber-clone/data/cards/speaker_cards.json"
DUP_COS = 0.90       # q+a cosine above this = same take -> merge
SRC_TOP_K = 12       # hybrid candidates per card before rerank (provenance, not recall-critical)
SRC_TOP_N = 4        # sources kept per card
SRC_PER_VIDEO = 2    # diversity cap, mirrors rag.retrieve

# same pattern as assemble_v4_sft / the eval harness — counted for the report, never filtered
PROF = re.compile(r"\b(sik|am[ıi]na|amc[ıi]k|g[öo]t|pi[çc]|o[çc]\b|orospu|yarr?a[kğ]|pezevenk|"
                  r"kahpe|ibne|gavat|anan[ıi]|avrad[ıi]|sokay[ıi]m|siktir|yavşak|şerefsiz)", re.I)


def load_pairs() -> list[dict]:
    rows = [json.loads(l) for l in open(PAIRS, encoding="utf-8") if l.strip()]
    return [{"q": r["messages"][1]["content"].strip(),
             "a": r["messages"][2]["content"].strip()} for r in rows]


def merge_dups(pairs: list[dict], dense: np.ndarray) -> list[list[int]]:
    """Greedy union of near-duplicate takes: each pair joins the first earlier group whose
    representative it matches at >= DUP_COS. Deterministic (input order)."""
    sims = dense @ dense.T
    groups: list[list[int]] = []
    rep: list[int] = []   # representative row per group
    for i in range(len(pairs)):
        for g, r in enumerate(rep):
            if sims[i, r] >= DUP_COS:
                groups[g].append(i)
                break
        else:
            groups.append([i])
            rep.append(i)
    return groups


def upload_date(video_id: str) -> str | None:
    p = f"{META_DIR}/{video_id}.json"
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p, encoding="utf-8")).get("upload_date")
    except (json.JSONDecodeError, OSError):
        return None


def main() -> None:
    s = load_settings()
    r = s.rag
    pairs = load_pairs()
    print(f"{len(pairs)} teacher pairs from {PAIRS}")

    emb = get_embedder(r.embedder)
    texts = [f"{p['q']}\n{p['a']}" for p in pairs]
    dense_parts = []
    for i in range(0, len(texts), 64):
        dense_parts.append(np.asarray(emb.encode(texts[i:i + 64]).dense))
        if (i // 64) % 8 == 0:
            print(f"  embedded {min(i + 64, len(texts))}/{len(texts)}")
    dense = np.concatenate(dense_parts)
    dense = dense / np.linalg.norm(dense, axis=1, keepdims=True)

    groups = merge_dups(pairs, dense)
    n_merged = sum(1 for g in groups if len(g) > 1)
    print(f"{len(groups)} cards after near-dup merge (>= {DUP_COS}): "
          f"{n_merged} merged groups, largest {max(len(g) for g in groups)}")

    # ---- provenance mining against speaker_clean (one store/reranker instance, batched queries)
    store = VectorStore(path=str(r.store_path), collection="speaker_clean", dense_dim=1024)
    rr = get_reranker(r.reranker)

    cards = []
    qs = []
    for g in groups:
        body_i = max(g, key=lambda i: len(pairs[i]["a"]))   # richest take = card body
        qs.append(pairs[body_i]["q"])
    q_enc_parts = []
    q_sparse: list[dict] = []
    for i in range(0, len(qs), 64):
        out = emb.encode(qs[i:i + 64])
        q_enc_parts.append(np.asarray(out.dense))
        q_sparse.extend(out.sparse)
    q_dense = np.concatenate(q_enc_parts)

    n_weak = 0
    prof_hits = 0
    for ci, g in enumerate(groups):
        body_i = max(g, key=lambda i: len(pairs[i]["a"]))
        q, a = pairs[body_i]["q"], pairs[body_i]["a"]
        hits = store.hybrid_search(query_dense=q_dense[ci], query_sparse=q_sparse[ci],
                                   top_k=SRC_TOP_K)
        srcs = []
        if hits:
            scores = rr.score(q, [h.chunk.text for h in hits])
            for h, sc in zip(hits, scores):
                h.rerank_score = float(sc)
            hits.sort(key=lambda h: h.rerank_score, reverse=True)
            per_vid: dict[str, int] = {}
            for h in hits:
                vid = h.chunk.video_id
                if per_vid.get(vid, 0) >= SRC_PER_VIDEO:
                    continue
                per_vid[vid] = per_vid.get(vid, 0) + 1
                srcs.append({"video_id": vid, "title": h.chunk.title,
                             "start": round(h.chunk.start, 1),
                             "score": round(h.rerank_score, 3)})
                if len(srcs) >= SRC_TOP_N:
                    break
        support = srcs[0]["score"] if srcs else float("-inf")
        n_strong = sum(1 for x in srcs if x["score"] >= r.gate_threshold)
        if n_strong == 0:
            n_weak += 1
        if PROF.search(a):
            prof_hits += 1
        cards.append({
            "id": f"card::{ci:04d}",
            "q": q,
            "take": a,
            "n_takes": len(g),
            "alt_qs": [pairs[i]["q"] for i in g if i != body_i],
            "sources": srcs,
            "support": round(support, 3) if srcs else None,
            "n_strong": n_strong,
            "date": upload_date(srcs[0]["video_id"]) if srcs else None,
        })
        if ci % 200 == 0:
            print(f"  provenance {ci}/{len(groups)}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(cards, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sup = sorted(c["support"] for c in cards if c["support"] is not None)
    print(f"\n-> {OUT}")
    print(f"   {len(cards)} cards | prof rate {prof_hits / len(cards):.2f} | "
          f"weak-support (n_strong=0, kept+marked) {n_weak}")
    if sup:
        print(f"   support: min {sup[0]:.2f} / median {sup[len(sup) // 2]:.2f} / max {sup[-1]:.2f} "
              f"(gate {r.gate_threshold})")


if __name__ == "__main__":
    main()
