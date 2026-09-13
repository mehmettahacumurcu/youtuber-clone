"""Stage 6.5 — text-level contamination cleaning of filter output (no re-transcribe).

Reads ``data/filter/{video_id}.json`` → writes ``data/clean/{video_id}.json``, removing
read-aloud third-party prose (datelines / newswire copy / reproduced statements) that
speaker-ID correctly KEPT — it really is his voice — but which is not his style, and applying
verified ASR corrections in place.

PROFANITY IS NEVER DROPPED — ``drop_patterns`` are structural only (dateline/newswire/credit),
never content/insult words (see project memory: profanity must survive intact).

Guest/interview contamination is NOT handled here. A corpus estimate showed ~99% of
post-identify segments sit in the 0.40-0.70 ECAPA band (his authentic voice averages ~0.52,
rarely ≥0.70), so the similarity score is non-discriminating at this stage and per-segment LLM
would be ~179k calls. Instead this stage emits a VIDEO-LEVEL flag (``title_flags`` +
``drop_ratio``) marking interview/panel videos for a later, LLM-scoped guest-removal pass.

``assemble`` reads ``clean/`` and falls back to ``filter/`` when a clean file is absent.

CLI: ``python -m pipeline.clean --config config.yaml (--video-id X | --all) [--force]``
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.config import CleanConfig, Settings

log = logging.getLogger("pipeline.clean")


def clean_path_for(video_id: str, settings: Settings) -> Path:
    return settings.paths.clean_dir / f"{video_id}.json"


def filter_path_for(video_id: str, settings: Settings) -> Path:
    return settings.paths.filter_dir / f"{video_id}.json"


def meta_path_for(video_id: str, settings: Settings) -> Path:
    return settings.paths.meta_dir / f"{video_id}.json"


def apply_corrections(text: str, corrections: dict[str, str]) -> str:
    """Word-boundary replace each verified-wrong ASR form. Each key is confirmed to never
    appear legitimately (see known_transcription_corrections memory) so a blanket replace is
    safe. Profanity is never in this map."""
    for wrong, right in corrections.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text)
    return text


def structural_drop_reason(text: str, patterns: list[re.Pattern]) -> str | None:
    """Return 'read_aloud' if the text matches any structural drop pattern, else None."""
    for pat in patterns:
        if pat.search(text):
            return "read_aloud"
    return None


def _video_title(video_id: str, settings: Settings) -> str:
    mp = meta_path_for(video_id, settings)
    if not mp.exists():
        return ""
    try:
        return json.loads(mp.read_text(encoding="utf-8")).get("title") or ""
    except Exception:  # noqa: BLE001
        return ""


def run_clean(video_id: str, settings: Settings, *, force: bool = False) -> Path:
    src = filter_path_for(video_id, settings)
    if not src.exists():
        raise FileNotFoundError(
            f"filter output not found: {src} (run pipeline.filter first)"
        )
    out = clean_path_for(video_id, settings)
    if out.exists() and not force:
        log.info("skip clean for %s (already at %s)", video_id, out)
        return out

    cfg = settings.clean
    patterns = [re.compile(p) for p in cfg.drop_patterns]
    doc = json.loads(src.read_text(encoding="utf-8"))
    segments_in = doc.get("segments", [])

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []  # for the review sidecar
    dropped_by_reason: dict[str, int] = {}
    corrected = 0
    for seg in segments_in:
        text0 = seg.get("text") or ""
        text = apply_corrections(text0, cfg.corrections) if cfg.corrections else text0
        if text != text0:
            corrected += 1
            seg = {**seg, "text": text}  # copy — never mutate the filter input
        reason = structural_drop_reason(text, patterns)
        if reason is None:
            kept.append(seg)
        else:
            dropped_by_reason[reason] = dropped_by_reason.get(reason, 0) + 1
            dropped.append(
                {"start": seg.get("start"), "end": seg.get("end"), "reason": reason, "text": text}
            )

    n_in = len(segments_in)
    drop_ratio = (n_in - len(kept)) / n_in if n_in else 0.0

    title = _video_title(video_id, settings)
    tlow = title.lower()
    matched_flags = [k for k in cfg.title_flags if k.lower() in tlow]
    flag_reasons: list[str] = []
    if matched_flags:
        flag_reasons.append("title:" + ",".join(matched_flags))
    if drop_ratio > cfg.max_drop_ratio_flag:
        flag_reasons.append(f"drop_ratio:{drop_ratio:.2f}")

    payload = {
        "video_id": video_id,
        "source": "filter",
        "title": title,
        "rules": {
            "drop_patterns": cfg.drop_patterns,
            "corrections": cfg.corrections,
        },
        "input_segments": n_in,
        "kept_segments": len(kept),
        "corrected_segments": corrected,
        "dropped_by_reason": dropped_by_reason,
        "pct_dropped": round(100 * drop_ratio, 2),
        "flagged_for_review": bool(flag_reasons),
        "flag_reasons": flag_reasons,
        "segments": kept,
        "ran_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if cfg.emit_review and dropped:
        review = out.with_suffix(".review.json")
        review.write_text(
            json.dumps({"video_id": video_id, "title": title, "dropped": dropped},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    log.info(
        "wrote %s — kept %d/%d (%.1f%% dropped, %d corrected)%s",
        out, len(kept), n_in, 100 * drop_ratio, corrected,
        f" FLAGGED({';'.join(flag_reasons)})" if flag_reasons else "",
    )
    return out


def discover_video_ids(settings: Settings) -> list[str]:
    """Videos that have a filter JSON."""
    return sorted(p.stem for p in settings.paths.filter_dir.glob("*.json"))
