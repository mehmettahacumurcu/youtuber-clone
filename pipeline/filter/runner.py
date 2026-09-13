"""Stage 6 — quality filtering of transcribed segments.

Reads ``data/transcribe/{video_id}.json`` and writes ``data/filter/{video_id}.json``
keeping only segments that pass quality gates. Gates are quality/hallucination only —
NEVER content/profanity (profanity must survive intact; see project memory).

Drop reasons (priority order):
  empty                  text blank after strip
  low_confidence         avg_logprob < filter.min_avg_logprob
  bad_length             duration_s outside [filter.min_segment_s, filter.max_segment_s]
  word_rate              words/second > filter.max_words_per_second
  hallucination_pattern  text matches one of filter.drop_patterns

CLI: ``python -m pipeline.filter --config config.yaml (--video-id X | --all) [--force]``
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.config import FilterConfig, Settings

log = logging.getLogger("pipeline.filter")


def filter_path_for(video_id: str, settings: Settings) -> Path:
    return settings.paths.filter_dir / f"{video_id}.json"


def transcribe_path_for(video_id: str, settings: Settings) -> Path:
    return settings.paths.transcribe_dir / f"{video_id}.json"


def classify_segment(seg: dict[str, Any], cfg: FilterConfig, patterns: list[re.Pattern]) -> str | None:
    """Return a drop reason, or None if the segment passes every gate."""
    text = (seg.get("text") or "").strip()
    if not text:
        return "empty"
    if seg.get("avg_logprob") is not None and seg["avg_logprob"] < cfg.min_avg_logprob:
        return "low_confidence"
    dur = float(seg.get("duration_s") or 0.0)
    if dur < cfg.min_segment_s or dur > cfg.max_segment_s:
        return "bad_length"
    words = len(text.split())
    if dur > 0 and words / dur > cfg.max_words_per_second:
        return "word_rate"
    for pat in patterns:
        if pat.search(text):
            return "hallucination_pattern"
    return None


def scrub_text(text: str, scrub_patterns: list[re.Pattern]) -> str:
    """Strip ASR-artifact substrings (subtitle credits, thanks-for-watching) IN PLACE, then
    collapse the whitespace the deletion leaves. Returns the cleaned text — may be empty if the
    whole segment was an artifact (the caller then drops it via the ``empty`` gate). Unlike
    drop_patterns this preserves the surrounding real speech around a mid-sentence artifact.
    """
    for pat in scrub_patterns:
        text = pat.sub("", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def run_filter(video_id: str, settings: Settings, *, force: bool = False) -> Path:
    src = transcribe_path_for(video_id, settings)
    if not src.exists():
        raise FileNotFoundError(
            f"transcribe output not found: {src} (run pipeline.transcribe first)"
        )
    out = filter_path_for(video_id, settings)
    if out.exists() and not force:
        log.info("skip filter for %s (already at %s)", video_id, out)
        return out

    cfg = settings.filter
    patterns = [re.compile(p) for p in cfg.drop_patterns]
    scrub = [re.compile(p, re.IGNORECASE) for p in cfg.scrub_patterns]
    doc = json.loads(src.read_text(encoding="utf-8"))
    segments_in = doc.get("segments", [])

    kept: list[dict[str, Any]] = []
    dropped_by_reason: dict[str, int] = {}
    scrubbed = 0
    for seg in segments_in:
        if scrub:
            cleaned = scrub_text(seg.get("text") or "", scrub)
            if cleaned != (seg.get("text") or ""):
                seg = {**seg, "text": cleaned}  # copy — never mutate the transcribe input
                scrubbed += 1
        reason = classify_segment(seg, cfg, patterns)
        if reason is None:
            kept.append(seg)
        else:
            dropped_by_reason[reason] = dropped_by_reason.get(reason, 0) + 1

    payload = {
        "video_id": video_id,
        "source": "transcribe",
        "gates": {
            "min_avg_logprob": cfg.min_avg_logprob,
            "min_segment_s": cfg.min_segment_s,
            "max_segment_s": cfg.max_segment_s,
            "max_words_per_second": cfg.max_words_per_second,
            "drop_patterns": cfg.drop_patterns,
            "scrub_patterns": cfg.scrub_patterns,
        },
        "input_segments": len(segments_in),
        "kept_segments": len(kept),
        "scrubbed_segments": scrubbed,
        "dropped_by_reason": dropped_by_reason,
        "segments": kept,
        "ran_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("wrote %s — kept %d/%d segments", out, len(kept), len(segments_in))
    return out


def discover_video_ids(settings: Settings) -> list[str]:
    """Videos that have a transcribe JSON."""
    return sorted(p.stem for p in settings.paths.transcribe_dir.glob("*.json"))
