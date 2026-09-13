"""Stage 6.5b — LLM guest/read-aloud detection, scoped to FLAGGED interview videos only.

Why scoped: a corpus estimate showed ~99% of post-identify segments sit in the 0.40-0.70
ECAPA band, so the similarity score can't tell his voice from a guest's, and per-segment LLM
over the whole corpus would be ~179k calls. Instead, the `clean` stage flags interview/panel
videos (title + structural drop ratio); this step runs the LLM ONLY on those, dropping segments
that aren't confidently him.

Flow (decoupled so the LLM backend is pluggable — Anthropic API, a Claude-Code Workflow, or a
mock in tests):

    flagged_videos -> build_batches (segments + neighbor context) -> adjudicate (LLM) ->
    apply_verdicts (drop non-him, rewrite data/clean/{vid}.json)

PRECISION-BIASED: anything not confidently "him" is dropped (purity > volume). PROFANITY is
never a drop reason — only speaker/quote attribution is.

CLI:
    python -m pipeline.clean.guest --config config.yaml --emit            # write batch files
    python -m pipeline.clean.guest --config config.yaml --apply           # apply verdict files
    python -m pipeline.clean.guest --config config.yaml --run             # API end-to-end (needs key)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

from pipeline.config import Settings, load_settings
from pipeline.clean.runner import clean_path_for

log = logging.getLogger("pipeline.clean.guest")

LABELS = ("him", "guest", "read_aloud", "other")
KEEP_LABEL = "him"

# A batch -> list of {"idx": int, "label": str, "reason": str}
Adjudicator = Callable[[dict[str, Any]], list[dict[str, Any]]]


def flagged_videos(settings: Settings) -> list[str]:
    """Video ids whose clean output is flagged_for_review (interview/panel) and not yet
    guest-reviewed."""
    out: list[str] = []
    for p in sorted(settings.paths.clean_dir.glob("*.json")):
        if p.name.endswith(".review.json"):
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if doc.get("flagged_for_review") and not doc.get("guest_reviewed"):
            out.append(doc.get("video_id", p.stem))
    return out


def _load_profile(settings: Settings) -> str:
    """Optional: data/voice_profile.md — his recurring phrases/topics, to help the LLM judge
    'his style'. Empty string if absent (the prompt still works, just with less prior)."""
    prof = settings.paths.data_dir / "voice_profile.md"
    if prof.exists():
        return prof.read_text(encoding="utf-8")[:4000]
    return ""


def build_batches(video_id: str, settings: Settings) -> list[dict[str, Any]]:
    """Group a flagged video's clean-kept segments into review batches with neighbor context.

    review.batch_size segments per batch; review.context_neighbors segments of text on each
    side as read-only context. Each segment carries its index in the kept list (the stable key
    that apply_verdicts maps verdicts back through)."""
    doc = json.loads(clean_path_for(video_id, settings).read_text(encoding="utf-8"))
    segs = doc.get("segments", [])
    title = doc.get("title", "")
    bs = settings.review.batch_size
    nb = settings.review.context_neighbors

    batches: list[dict[str, Any]] = []
    for b0 in range(0, len(segs), bs):
        window = list(range(b0, min(b0 + bs, len(segs))))
        items = []
        for i in window:
            items.append({
                "idx": i,
                "start": segs[i].get("start"),
                "end": segs[i].get("end"),
                "text": segs[i].get("text", ""),
                "context_before": [segs[j].get("text", "") for j in range(max(0, i - nb), i)],
                "context_after": [segs[j].get("text", "") for j in range(i + 1, min(len(segs), i + 1 + nb))],
            })
        batches.append({
            "video_id": video_id,
            "title": title,
            "batch_index": len(batches),
            "segments": items,
        })
    return batches


def build_prompt(batch: dict[str, Any], profile: str = "") -> str:
    """The precision-biased adjudication prompt (Turkish-aware)."""
    head = (
        "Aşağıda bir Türk YouTuber'ın RÖPORTAJ/PANEL videosundan transkript parçaları var. "
        f"Video başlığı: \"{batch.get('title','')}\".\n"
        "Bazı parçalar KENDİSİNİN konuşmasıdır (kendi üslubu, görüşleri); bazıları bir KONUĞA "
        "aittir; bazıları okunan üçüncü-şahıs metni (haber/açıklama/alıntı) olabilir.\n\n"
        "Her parça için etiket ver: him | guest | read_aloud | other.\n"
        "ÖNEMLİ — temkinli ol: KESİN olarak kendisi değilse 'him' DEME (saflık > hacim). "
        "Küfür/argo bir düşürme sebebi DEĞİLDİR; sadece konuşmacı/alıntı kimliğine bak.\n"
    )
    if profile:
        head += f"\nÜslup profili (kendisi):\n{profile}\n"
    body = ["\nParçalar (idx | önceki bağlam || METİN || sonraki bağlam):"]
    for it in batch["segments"]:
        cb = " / ".join(it["context_before"]) or "—"
        ca = " / ".join(it["context_after"]) or "—"
        body.append(f"[{it['idx']}] {cb} || {it['text']} || {ca}")
    tail = (
        "\n\nSADECE şu JSON'u döndür: "
        '{"labels":[{"idx":<int>,"label":"him|guest|read_aloud|other","reason":"<kısa>"}]}'
    )
    return head + "\n".join(body) + tail


def apply_verdicts(video_id: str, settings: Settings, labels: dict[int, dict[str, Any]]) -> Path:
    """Drop segments not labeled 'him' and rewrite data/clean/{vid}.json.

    ``labels`` maps segment index -> {"label","reason"}. Segments without a verdict are KEPT
    (no LLM opinion -> don't drop on missing data; the structural pass already ran). Updates
    dropped_by_reason, appends to the review sidecar, sets guest_reviewed=True."""
    out = clean_path_for(video_id, settings)
    doc = json.loads(out.read_text(encoding="utf-8"))
    segs = doc.get("segments", [])

    kept: list[dict[str, Any]] = []
    newly_dropped: list[dict[str, Any]] = []
    by_reason: dict[str, int] = dict(doc.get("dropped_by_reason", {}))
    for i, seg in enumerate(segs):
        verdict = labels.get(i)
        if verdict is None or verdict.get("label") == KEEP_LABEL:
            kept.append(seg)
        else:
            label = verdict.get("label", "other")
            by_reason[label] = by_reason.get(label, 0) + 1
            newly_dropped.append({
                "start": seg.get("start"), "end": seg.get("end"),
                "reason": label, "llm_reason": verdict.get("reason", ""), "text": seg.get("text", ""),
            })

    n_in = doc.get("input_segments", len(segs))
    doc["segments"] = kept
    doc["kept_segments"] = len(kept)
    doc["dropped_by_reason"] = by_reason
    doc["pct_dropped"] = round(100 * (n_in - len(kept)) / n_in, 2) if n_in else 0.0
    doc["guest_reviewed"] = True
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    if settings.clean.emit_review and newly_dropped:
        review = out.with_suffix(".review.json")
        existing = {"video_id": video_id, "title": doc.get("title", ""), "dropped": []}
        if review.exists():
            try:
                existing = json.loads(review.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
        existing.setdefault("dropped", []).extend(newly_dropped)
        review.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("guest-review %s: dropped %d, kept %d", video_id, len(newly_dropped), len(kept))
    return out


def run_guest_review(
    settings: Settings,
    adjudicate: Adjudicator,
    video_ids: list[str] | None = None,
) -> dict[str, int]:
    """End-to-end: for each flagged video, batch -> adjudicate -> apply. ``adjudicate`` is the
    pluggable LLM backend (API, Workflow, or a test mock). Returns {video_id: dropped_count}."""
    vids = video_ids if video_ids is not None else flagged_videos(settings)
    result: dict[str, int] = {}
    for vid in vids:
        labels: dict[int, dict[str, Any]] = {}
        for batch in build_batches(vid, settings):
            for verdict in adjudicate(batch):
                if "idx" in verdict and verdict.get("label") in LABELS:
                    labels[int(verdict["idx"])] = verdict
        before = len(json.loads(clean_path_for(vid, settings).read_text(encoding="utf-8"))["segments"])
        apply_verdicts(vid, settings, labels)
        after = len(json.loads(clean_path_for(vid, settings).read_text(encoding="utf-8"))["segments"])
        result[vid] = before - after
    return result


# ---- backends ---------------------------------------------------------------------

def adjudicate_via_api(model: str = "claude-opus-4-8") -> Adjudicator:
    """Build an Anthropic-API adjudicator. Lazy-imports the SDK and reads ANTHROPIC_API_KEY,
    so importing this module never requires either. Returns a function batch -> verdicts."""
    import os
    import anthropic  # lazy

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _adj(batch: dict[str, Any]) -> list[dict[str, Any]]:
        prompt = build_prompt(batch)
        msg = client.messages.create(
            model=model, max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            log.warning("no JSON in API response for %s batch %d", batch["video_id"], batch["batch_index"])
            return []
        return json.loads(text[start:end + 1]).get("labels", [])

    return _adj


# ---- CLI --------------------------------------------------------------------------

def _emit_batches(settings: Settings) -> int:
    n = 0
    for vid in flagged_videos(settings):
        for batch in build_batches(vid, settings):
            bp = settings.paths.clean_dir / f"{vid}.guest_batch_{batch['batch_index']:04d}.json"
            bp.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
            n += 1
    print(f"emitted {n} guest-review batch files to {settings.paths.clean_dir}", file=sys.stderr)
    return 0


def _apply_verdict_files(settings: Settings) -> int:
    """Apply data/clean/{vid}.guest_verdicts.json (written by the adjudicator: {"labels":[...]})."""
    applied = 0
    for vp in sorted(settings.paths.clean_dir.glob("*.guest_verdicts.json")):
        vid = vp.name[: -len(".guest_verdicts.json")]
        labels_list = json.loads(vp.read_text(encoding="utf-8")).get("labels", [])
        labels = {int(v["idx"]): v for v in labels_list if "idx" in v and v.get("label") in LABELS}
        apply_verdicts(vid, settings, labels)
        applied += 1
    print(f"applied verdicts for {applied} videos", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pipeline.clean.guest", description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--emit", action="store_true", help="Write batch files for flagged videos")
    g.add_argument("--apply", action="store_true", help="Apply *.guest_verdicts.json files")
    g.add_argument("--run", action="store_true", help="End-to-end via Anthropic API (needs ANTHROPIC_API_KEY)")
    p.add_argument("--model", default="claude-opus-4-8")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings(args.config)
    if args.emit:
        return _emit_batches(settings)
    if args.apply:
        return _apply_verdict_files(settings)
    dropped = run_guest_review(settings, adjudicate_via_api(args.model))
    print(f"guest-review complete: {dropped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
