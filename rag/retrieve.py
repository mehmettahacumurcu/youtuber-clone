"""Query -> hybrid retrieve -> cross-encoder rerank -> top-N hits with scores."""
from __future__ import annotations

import dataclasses

import numpy as np

from rag.embed import get_embedder, get_reranker
from rag.store import VectorStore
from rag.types import Hit


def _windows(text: str, size: int, stride: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text]
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        if j < len(text):
            k = text.rfind(" ", i + size // 2, j)
            if k > 0:
                j = k
        out.append(text[i:j].strip())
        if j >= len(text):
            break
        i += stride
    return [w for w in out if w]


def refine_windows(query: str, hits: list[Hit], reranker_name: str, size: int) -> list[Hit]:
    """Replace each hit's text with its best ~size-char window w.r.t. the query.

    The model only ever sees span_char_cap chars of a chunk (training AND inference) — without
    this, that's blindly the chunk HEAD, and a take sitting mid-chunk is invisible. The reranker
    is already loaded; scoring a handful of windows per hit is cheap."""
    rr = get_reranker(reranker_name)
    out = []
    for h in hits:
        wins = _windows(h.chunk.text, size, max(size // 2, 200))
        if len(wins) > 1:
            scores = rr.score(query, wins)
            best = wins[int(np.argmax(scores))]
            h.chunk = dataclasses.replace(h.chunk, text=best)  # Chunk is frozen; Hit is mutable
        out.append(h)
    return out


def retrieve(query: str, store_path: str, collection: str, embedder_name: str,
             reranker_name: str, top_k: int, top_n: int,
             max_per_video: int | None = None, window_chars: int | None = None,
             store: VectorStore | None = None) -> list[Hit]:
    emb = get_embedder(embedder_name)
    q = emb.encode([query])
    active_store = store or VectorStore(
        path=store_path, collection=collection, dense_dim=1024
    )
    candidates = active_store.hybrid_search(
        query_dense=np.asarray(q.dense[0]),
        query_sparse=q.sparse[0],
        top_k=top_k,
    )
    if not candidates:
        return []
    rr = get_reranker(reranker_name)
    scores = rr.score(query, [h.chunk.text for h in candidates])
    for h, sc in zip(candidates, scores):
        h.rerank_score = sc
    candidates.sort(key=lambda h: h.rerank_score, reverse=True)
    def _finish(hits: list[Hit]) -> list[Hit]:
        return refine_windows(query, hits, reranker_name, window_chars) if window_chars else hits

    if not max_per_video or max_per_video <= 0:
        return _finish(candidates[:top_n])
    # Source diversity: without a cap, one video can supply every top hit (observed: 6/6 on
    # "mutlak butlan"), starving multi-video topics of their other takes.
    picked: list[Hit] = []
    per_vid: dict[str, int] = {}
    for h in candidates:
        vid = h.chunk.video_id
        if per_vid.get(vid, 0) >= max_per_video:
            continue
        per_vid[vid] = per_vid.get(vid, 0) + 1
        picked.append(h)
        if len(picked) >= top_n:
            break
    return _finish(picked)
