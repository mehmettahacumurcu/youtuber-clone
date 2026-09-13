"""Turn per-video cleaned segment JSONs into token-budgeted chunks for retrieval.

Each chunk groups consecutive segments of ONE video up to ~chunk_tokens, with a
chunk_overlap-token overlap so a fact split across a boundary is still retrievable.
Citation metadata (video_id, start, end, title) rides along on every chunk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from rag.types import Chunk

# Cheap token proxy: whitespace words * 1.4 (~Turkish subword inflation). Avoids loading a
# tokenizer just to chunk; exactness does not matter — chunks only feed the embedder.
_TOK_PER_WORD = 1.4


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * _TOK_PER_WORD)


def _title_for(meta_dir: Path, video_id: str) -> str:
    mp = meta_dir / f"{video_id}.json"
    if mp.is_file():
        try:
            t = json.loads(mp.read_text(encoding="utf-8")).get("title")
            if t:
                return str(t)
        except (json.JSONDecodeError, OSError):
            pass
    return video_id


def _chunk_video(video_id: str, segments: list[dict], title: str,
                 chunk_tokens: int, overlap: int) -> Iterator[Chunk]:
    i, ordinal = 0, 0
    n = len(segments)
    while i < n:
        group: list[dict] = []
        tok = 0
        j = i
        while j < n and (not group or
                         tok + estimate_tokens(segments[j].get("text", "")) <= chunk_tokens):
            group.append(segments[j])
            tok += estimate_tokens(segments[j].get("text", ""))
            j += 1
        text = " ".join(s.get("text", "").strip() for s in group).strip()
        yield Chunk(
            id=f"{video_id}::{ordinal}",
            video_id=video_id,
            start=float(group[0].get("start", 0.0)),
            end=float(group[-1].get("end", group[0].get("start", 0.0))),
            title=title,
            text=text,
        )
        ordinal += 1
        if j >= n:
            break
        back = 0
        k = j - 1
        while k > i and back < overlap:
            back += estimate_tokens(segments[k].get("text", ""))
            k -= 1
        i = max(k + 1, i + 1)


def chunk_corpus(clean_dir: Path, meta_dir: Path,
                 chunk_tokens: int = 500, overlap: int = 50,
                 select: dict[str, str] | None = None) -> Iterator[Chunk]:
    """If `select` is given (a {video_id: tier} map from hisonly_filter.select_videos), only
    videos in it are chunked — this is how the clean his-only index excludes guest/interview
    formats wholesale (per-segment similarity can't separate speakers; format is the only axis)."""
    clean_dir, meta_dir = Path(clean_dir), Path(meta_dir)
    for fp in sorted(clean_dir.glob("*.json")):
        if fp.name.endswith(".review.json"):
            continue
        doc = json.loads(fp.read_text(encoding="utf-8"))
        video_id = doc.get("video_id") or fp.stem
        if select is not None and video_id not in select:
            continue
        segments = doc.get("segments") or []
        if not segments:
            continue
        title = _title_for(meta_dir, video_id)
        yield from _chunk_video(video_id, segments, title, chunk_tokens, overlap)
