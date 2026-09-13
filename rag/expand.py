"""Dedicated-video span expansion — the arm-B fix from the 2026-07-04 context A/B.

When the strong hits concentrate in ONE video (he made a dedicated video about the topic),
the canonical 3x800-char injection starves the model and it parrots clipped fragments
(_tmp_ctx_ab_results.md, arm A). Widening those hits into their surrounding transcript and
merging them in document order (~14k chars ≈ 5k tokens) flips the answer to real substance
in voice (arm B). Whole-transcript injection collapses to verbatim echo (arm C) — never that.

Selection is RELATIVE to the top rerank score, not absolute: bge-reranker logits swing whole
points with query phrasing ("mutlak butlan" tops at 3.1; "mutlak butlan meselesi hakkında ne
düşünüyorsun?" tops at 0.5 for the same corpus), so any fixed threshold breaks on one regime
or the other. Calibrated on those butlan queries, 2026-07-04.

Wired into rag/retrieve_server.py (adds expanded_* fields to /retrieve responses) and consumed
by ui/speaker_studio.py (uncapped spans + num_ctx 8192 when present).
"""
from __future__ import annotations

import json
from pathlib import Path

from rag.types import Hit

# Expansion needs a logit-positive top hit — gate_threshold is currently 0.0 (everything
# "grounded"), so this is the only guard keeping 14k chars of off-topic transcript out of
# the prompt on junk questions.
TOP_MIN_SCORE = 0.0
# Hits within this of the top score count as "strong". 1.75 keeps the dedicated video's
# second chunk (top-1.63 in the keyword regime, top-1.43 in the question regime) while
# excluding the junk tail (>=2.4 below top in both).
ANCHOR_BAND = 1.75
BUDGET_CHARS = 14000            # total span budget (~5k tok) — proven scale; bigger risks echo
MIN_EXPAND, MAX_EXPAND = 1500, 7000  # per-side widening, adapted to anchor count (see expand())

_cache: dict[str, str | None] = {}


def _norm(s: str) -> str:
    return " ".join(s.split())


def _transcript(clean_dir: Path, video_id: str) -> str | None:
    """Full stitched transcript, cached. Built exactly like rag.chunk joins segments, so any
    chunk (or refined window of one) is a substring of it after whitespace normalization."""
    if video_id not in _cache:
        try:
            doc = json.loads((Path(clean_dir) / f"{video_id}.json").read_text(encoding="utf-8"))
            _cache[video_id] = _norm(" ".join((s.get("text") or "").strip()
                                              for s in doc.get("segments") or []))
        except (OSError, ValueError):
            _cache[video_id] = None
    return _cache[video_id]


def pick_video(hits: list[Hit]) -> tuple[str | None, list[Hit]]:
    """(video to expand, its anchor hits) — (None, []) when nothing is strong enough.

    Dedicated video = >=2 strong hits from one video (max_per_video caps at 2, so 2 is both
    the minimum meaningful count and the maximum observable one); falls back to the top hit's
    video when no count signal exists."""
    if not hits:
        return None, []
    top = max(hits, key=lambda h: h.rerank_score)
    if top.rerank_score <= TOP_MIN_SCORE:
        return None, []
    floor = top.rerank_score - ANCHOR_BAND
    strong = [h for h in hits if h.rerank_score >= floor]
    counts: dict[str, int] = {}
    for h in strong:
        counts[h.chunk.video_id] = counts.get(h.chunk.video_id, 0) + 1
    best_vid, best_n = max(counts.items(), key=lambda kv: kv[1])
    vid = best_vid if best_n >= 2 else top.chunk.video_id
    return vid, [h for h in strong if h.chunk.video_id == vid]


def _locate(full: str, text: str) -> int:
    t = _norm(text)
    off = full.find(t[:60])
    if off < 0 and len(t) > 260:
        off = full.find(t[200:260])
        off = off - 200 if off >= 200 else -1
    return off


def expand(hits: list[Hit], clean_dir: Path) -> tuple[str | None, list[str]]:
    """(video_id, merged long spans in document order) — (None, []) when expansion can't apply
    (no strong hits, transcript unavailable, or no anchor locatable in it)."""
    vid, anchors = pick_video(hits)
    if not vid:
        return None, []
    full = _transcript(clean_dir, vid)
    if not full:
        return None, []

    # Fewer anchors -> wider windows, so total context stays near BUDGET_CHARS either way.
    width = min(MAX_EXPAND, max(MIN_EXPAND, BUDGET_CHARS // (2 * len(anchors))))
    ivals: list[tuple[int, int]] = []
    for h in anchors:
        off = _locate(full, h.chunk.text)
        if off >= 0:
            end = off + len(_norm(h.chunk.text))
            ivals.append((max(0, off - width), min(len(full), end + width)))
    if not ivals:
        return None, []

    ivals.sort()
    merged = [list(ivals[0])]
    for s, e in ivals[1:]:
        if s <= merged[-1][1] + 200:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    spans: list[str] = []
    used = 0
    for s, e in merged:
        take = full[s:e][: max(0, BUDGET_CHARS - used)]
        if take:
            spans.append(take)
            used += len(take)
    return vid, spans
