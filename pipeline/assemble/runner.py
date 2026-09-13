"""Stage 9 — assemble the LLM training corpus from filtered segments.

Reads every ``data/filter/{video_id}.json`` and writes one JSONL document per video to
``data/dataset/llm_corpus.jsonl``. Within a video, kept segments are sorted by start
time and joined into a single text; a skip marker (``assemble.llm_skip_marker``) is
inserted between consecutive kept segments whose time gap exceeds
``assemble.skip_gap_s`` — marking non-contiguous runs per the brief.

CLI: ``python -m pipeline.assemble --config config.yaml``
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pipeline.config import Settings

log = logging.getLogger("pipeline.assemble")


# Corpus filename per source: "clean" is the canonical (ASR-corrected) build training reads;
# "filter" is the RAW build (uncorrected) kept side-by-side so raw is never lost.
_CORPUS_NAME = {"clean": "llm_corpus.jsonl", "filter": "llm_corpus.raw.jsonl"}


def dataset_path_for(settings: Settings, source: str = "clean") -> Path:
    return settings.paths.dataset_dir / _CORPUS_NAME[source]


def build_document(
    segments: list[dict[str, Any]], *, skip_marker: str, skip_gap_s: float
) -> str:
    """Stitch kept segments into one document, sorted by start time.

    Consecutive segments are joined by a space; when the gap between the previous
    segment's end and the next segment's start exceeds ``skip_gap_s``, the skip marker
    is inserted on its own line between them. Empty/blank segments are skipped.
    """
    ordered = sorted(segments, key=lambda s: float(s["start"]))
    parts: list[str] = []
    prev_end: float | None = None
    for seg in ordered:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg["start"])
        if prev_end is not None and (start - prev_end) > skip_gap_s:
            parts.append(f"\n{skip_marker}\n")
        elif parts:
            parts.append(" ")
        parts.append(text)
        prev_end = float(seg["end"])
    return "".join(parts).strip()


def _read_title(video_id: str, settings: Settings) -> str | None:
    meta = settings.paths.meta_dir / f"{video_id}.json"
    if not meta.exists():
        return None
    try:
        return json.loads(meta.read_text(encoding="utf-8")).get("title")
    except Exception:  # noqa: BLE001
        return None


def discover_video_ids(settings: Settings) -> list[str]:
    """Videos that have a filter JSON."""
    return sorted(p.stem for p in settings.paths.filter_dir.glob("*.json"))


def _source_path(video_id: str, settings: Settings, source: str) -> Path:
    """Per-video JSON to assemble from. ``filter`` always uses filter/. ``clean`` prefers the
    clean/ file (ASR-corrected) and falls back to filter/ when a clean file is absent — the
    video universe is always the filter dir, so no video is dropped if clean lags."""
    if source == "filter":
        return settings.paths.filter_dir / f"{video_id}.json"
    clean = settings.paths.clean_dir / f"{video_id}.json"
    return clean if clean.exists() else settings.paths.filter_dir / f"{video_id}.json"


def run_assemble(settings: Settings, source: str = "clean") -> Path:
    if source not in _CORPUS_NAME:
        raise ValueError(f"source must be one of {sorted(_CORPUS_NAME)}, got {source!r}")
    cfg = settings.assemble
    out = dataset_path_for(settings, source)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for vid in discover_video_ids(settings):
        doc = json.loads(_source_path(vid, settings, source).read_text(encoding="utf-8"))
        segments = doc.get("segments", [])
        text = build_document(
            segments, skip_marker=cfg.llm_skip_marker, skip_gap_s=cfg.skip_gap_s
        )
        if not text:
            log.warning("video %s has no text after assembly — skipping", vid)
            continue
        record = {
            "video_id": vid,
            "title": _read_title(vid, settings),
            "text": text,
            "n_segments": len(segments),
            "char_count": len(text),
        }
        lines.append(json.dumps(record, ensure_ascii=False))

    # newline="\n": byte-identical on Windows and Linux (no CRLF translation), so the raw/clean
    # corpora and their git blobs match across machines — same care as training/build_dataset.write.
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
    log.info("wrote %s (source=%s) — %d documents", out, source, len(lines))
    return out
