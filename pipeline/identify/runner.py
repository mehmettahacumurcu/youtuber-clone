"""Embed each diarization segment with ECAPA-TDNN and similarity-gate vs reference.

Output schema (``data/identify/{video_id}.json``)::

    {
      "video_id": "...",
      "embedding_model": "speechbrain/spkrec-ecapa-voxceleb",
      "reference_clip": "data/reference/speaker_reference.wav",
      "similarity_threshold": 0.65,
      "borderline_range": [0.50, 0.70],
      "segments": [
        {"start": 0.5, "end": 3.2, "speaker": "SPEAKER_00",
         "similarity": 0.81, "decision": "keep"},
        ...
      ],
      "summary": {
        "total_segments": 666,
        "kept_segments": 580,
        "borderline_segments": 30,
        "dropped_segments": 50,
        "too_short_segments": 6,
        "kept_speech_s": 1450.2,
        "kept_speech_fraction": 0.82,
        "per_speaker_avg_similarity": {"SPEAKER_00": 0.81, "SPEAKER_01": 0.32}
      },
      "device": "cuda",
      "elapsed_s": 12.4,
      "ran_at_utc": "..."
    }

ECAPA-TDNN expects 16 kHz mono. Our pipeline produces that natively (Stage 1).

Decisions are derived as follows:
  similarity >= identify.similarity_threshold      -> keep
  borderline_low <= similarity < borderline_high   -> borderline (flag for human review)
  similarity < borderline_low                      -> drop
  segment shorter than MIN_SEGMENT_S                -> too_short (no decision)

Borderline segments are NOT auto-kept; they get human review at Stage 8.

Sliding-window sub-segmentation: a Stage-3 segment longer than ``identify.window_s`` is not
judged as a whole. We slide a short ECAPA window across it, gate each window vs the
reference, bridge single-window dips, and emit one record per contiguous keep-span (with
REFINED start/end). This excises guest-only spans from the speaker-mixed segments the
clustering-free Stage 3 produces. Such records carry ``"refined": true`` plus
``parent_start``/``parent_end`` (the original Stage-3 segment bounds); one input segment can
therefore map to several output records, or collapse to a single ``drop``. Segments shorter
than ``window_s`` keep the legacy single-embedding behavior. Disable by setting ``window_s``
above ``diarize.seg_max_segment_s``.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from pipeline.config import REPO_ROOT, Settings
from pipeline.diarize import diarize_path_for
from pipeline.download import audio_path_for

log = logging.getLogger("pipeline.identify")

# Segments shorter than this are too brief for ECAPA to embed reliably.
MIN_SEGMENT_S = 0.5


def identify_path_for(video_id: str, settings: Settings) -> Path:
    return settings.paths.identify_dir / f"{video_id}.json"


def _ecapa_cache_dir() -> Path:
    d = REPO_ROOT / "data" / ".cache" / "speechbrain" / "spkrec-ecapa-voxceleb"
    d.mkdir(parents=True, exist_ok=True)
    return d


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str, device: str):
    """Load and cache the ECAPA-TDNN encoder via speechbrain."""
    # Import lazily — speechbrain pulls torch + heavy modules on import.
    from speechbrain.pretrained import EncoderClassifier  # speechbrain 0.5.x

    log.info("loading embedding model %s on %s", model_name, device)
    model = EncoderClassifier.from_hparams(
        source=model_name,
        savedir=str(_ecapa_cache_dir()),
        run_opts={"device": device},
    )
    return model


def _resolve_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _read_mono_16k(path: Path, expected_sr: int):
    """Read a wav file, return mono float32 numpy array. Assert sample rate."""
    import numpy as np
    import soundfile as sf

    samples, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != expected_sr:
        raise ValueError(f"{path} sample rate {sr} != expected {expected_sr}")
    if samples.ndim > 1:
        samples = samples.mean(axis=1).astype(np.float32, copy=False)
    return samples


def _embed(model, audio_1d, device: str):
    """Embed a 1-D numpy float32 audio array with ECAPA. Returns a 1-D torch tensor on CPU."""
    import torch

    t = torch.from_numpy(audio_1d).unsqueeze(0).to(device)  # (1, T)
    with torch.no_grad():
        # encode_batch returns (batch, 1, embedding_dim) for ECAPA
        emb = model.encode_batch(t)
    return emb.squeeze().detach().cpu()


def _embed_batch(model, clips, device, max_bs: int = 128):
    """Embed a list of 1-D float32 audio arrays with ECAPA, batching for speed.

    Returns a list of 1-D CPU tensors (one per input clip, order preserved). Clips are
    grouped by EXACT length and each same-length group is stacked and embedded in one
    call. We deliberately do NOT zero-pad + ``wav_lens``: padding is NOT bit-exact for
    ECAPA in this SpeechBrain version (short clips come back at cosine ~0.72), whereas
    same-length stacking matches the one-at-a-time path exactly (cosine 1.000000). In a
    sliding window grid almost all windows share one length (only a clamped last window
    differs), so this still amortizes ~all GPU launches → ~7-10x faster, bit-identical.
    """
    import numpy as np
    import torch
    from collections import defaultdict

    by_len: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(clips):
        by_len[len(c)].append(i)

    out: list = [None] * len(clips)
    for _length, idxs in by_len.items():
        for s in range(0, len(idxs), max_bs):
            grp = idxs[s : s + max_bs]
            batch = torch.from_numpy(np.stack([clips[i] for i in grp])).to(device)
            with torch.no_grad():
                emb = model.encode_batch(batch)  # (B, 1, D)
            emb = emb.squeeze(1).detach().cpu()
            for k, i in enumerate(grp):
                out[i] = emb[k]
    return out


def _cosine(a, b) -> float:
    import torch

    a = a.flatten().float()
    b = b.flatten().float()
    na = a.norm()
    nb = b.norm()
    if float(na) == 0.0 or float(nb) == 0.0:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))


def _load_reference_embeddings(settings: Settings, model, device: str, sr: int) -> tuple[list, list[str]]:
    """Return (list of normalized-ish reference embeddings, list of label strings).

    If ``data/reference/pool/active.json`` exists with active candidate indices,
    use those (max-similarity-across-pool semantics). Otherwise fall back to the
    single seed clip at ``settings.paths.reference_clip``.
    """
    from ui.state import active_pool_path  # local import to avoid UI dep at module load

    pool_path = active_pool_path(settings)
    candidates_path = settings.paths.data_dir / "reference" / "pool" / "candidates.json"
    if pool_path.exists() and candidates_path.exists():
        active = json.loads(pool_path.read_text(encoding="utf-8"))
        cand_doc = json.loads(candidates_path.read_text(encoding="utf-8"))
        cand_by_idx = {c["candidate_index"]: c for c in cand_doc["candidates"]}
        idxs = [int(i) for i in active.get("active_indices", []) if int(i) in cand_by_idx]
        if idxs:
            log.info("using active reference pool with %d members: %s", len(idxs), idxs)
            embs = []
            labels = []
            for idx in sorted(idxs):
                c = cand_by_idx[idx]
                p = REPO_ROOT / c["file"]
                if not p.exists():
                    log.warning("active pool member missing on disk: %s", p)
                    continue
                aud = _read_mono_16k(p, sr)
                embs.append(_embed(model, aud, device))
                labels.append(f"#{idx:02d}_{c['source_video']}")
            if embs:
                return embs, labels
            log.warning("no active pool members found on disk; falling back to seed")
    # Fallback: single seed reference
    ref_path = settings.paths.reference_clip
    if not ref_path.exists():
        raise FileNotFoundError(
            f"no reference available: neither active pool nor seed {ref_path} found"
        )
    log.info("using single seed reference %s", ref_path)
    aud = _read_mono_16k(ref_path, sr)
    return [_embed(model, aud, device)], ["seed"]


def _max_cosine(seg_emb, ref_embs) -> tuple[float, int]:
    """Return (max_cosine_sim, index_of_argmax) over a list of reference embeddings."""
    best = -2.0
    best_i = 0
    for i, r in enumerate(ref_embs):
        s = _cosine(seg_emb, r)
        if s > best:
            best = s
            best_i = i
    return best, best_i


# --- Sliding-window sub-segmentation (pure logic, unit-tested without pyannote/ECAPA) -----
# The clustering-free Stage 3 collapses speakers, so one segment can contain him + a guest.
# A whole-segment embedding is dominated by him (~0.6) and wrongly passes the gate, dragging
# the guest's words in. We instead slide a short window, embed each in isolation, and keep
# only the contiguous spans that match the reference — guest-only windows (~0.3) get excised.


def _window_grid(start: float, end: float, win: float, hop: float) -> list[tuple[float, float]]:
    """Full-length windows of length ``win`` stepping by ``hop`` across [start, end].

    Every window is exactly ``win`` long (the final one is shifted back to end at ``end``)
    so their embeddings stay comparable. Spans <= win return a single [(start, end)].
    """
    span = end - start
    if span <= win:
        return [(round(start, 6), round(end, 6))]
    out: list[tuple[float, float]] = []
    k = 0
    while True:
        w0 = start + k * hop
        w1 = w0 + win
        if w1 >= end:
            last = (round(end - win, 6), round(end, 6))
            if not out or abs(out[-1][0] - last[0]) > 1e-3:
                out.append(last)
            break
        out.append((round(w0, 6), round(w1, 6)))
        k += 1
    return out


def _bridge_dips(keep: list[bool], max_dip: int) -> list[bool]:
    """Flip runs of <= max_dip consecutive False values flanked by True on BOTH sides (a
    short him->guest->him transition window), leaving longer guest runs intact.
    """
    if max_dip <= 0:
        return list(keep)
    out = list(keep)
    n = len(out)
    i = 0
    while i < n:
        if keep[i]:
            i += 1
            continue
        j = i
        while j < n and not keep[j]:
            j += 1
        flanked = i > 0 and j < n and keep[i - 1] and keep[j]
        if flanked and (j - i) <= max_dip:
            for k in range(i, j):
                out[k] = True
        i = j
    return out


def _coalesce_keep_windows(
    grid: list[tuple[float, float]], keep: list[bool], min_span: float
) -> list[tuple[float, float, list[int]]]:
    """Merge maximal runs of keep-windows into (start, end, window_indices) spans, dropping
    spans shorter than ``min_span``.
    """
    spans: list[tuple[float, float, list[int]]] = []
    n = len(keep)
    i = 0
    while i < n:
        if not keep[i]:
            i += 1
            continue
        j = i
        while j < n and keep[j]:
            j += 1
        s = grid[i][0]
        e = grid[j - 1][1]
        if (e - s) >= min_span - 1e-9:
            spans.append((round(s, 3), round(e, 3), list(range(i, j))))
        i = j
    return spans


def _decision_for(sim: float, cfg) -> str:
    if sim >= cfg.similarity_threshold:
        return "keep"
    if sim >= cfg.borderline_low:
        return "borderline"
    return "drop"


def _embed_whole(seg, audio, start_idx, end_idx, model, ref_embs, ref_labels, cfg, dev) -> dict[str, Any]:
    """Legacy path: one embedding for the whole (short) segment."""
    rec: dict[str, Any] = {
        "start": seg["start"], "end": seg["end"], "speaker": seg.get("speaker"),
        "similarity": None, "decision": "too_short",
    }
    try:
        sim, best_i = _max_cosine(_embed(model, audio[start_idx:end_idx], dev), ref_embs)
        rec["similarity"] = round(sim, 4)
        if len(ref_embs) > 1:
            rec["matched_ref"] = ref_labels[best_i]
        rec["decision"] = _decision_for(sim, cfg)
    except Exception as exc:  # noqa: BLE001
        log.warning("embed failed for %.2f-%.2f: %s", seg["start"], seg["end"], exc)
        rec["decision"] = "error"
    return rec


def _identify_segment(seg, audio, sr, model, ref_embs, ref_labels, cfg, dev) -> list[dict[str, Any]]:
    """Classify one Stage-3 segment, returning 1+ output records.

    Long segments are sub-segmented by a sliding ECAPA window so guest-only spans are
    excised; short segments fall back to a single whole-segment embedding.
    """
    seg_start = float(seg["start"])
    seg_end = float(seg["end"])
    speaker = seg.get("speaker")
    start_idx = max(0, int(seg_start * sr))
    end_idx = min(len(audio), int(seg_end * sr))
    dur = (end_idx - start_idx) / sr

    if dur < MIN_SEGMENT_S or end_idx <= start_idx:
        return [{"start": seg_start, "end": seg_end, "speaker": speaker,
                 "similarity": None, "decision": "too_short"}]

    if dur < cfg.window_s:
        return [_embed_whole(seg, audio, start_idx, end_idx, model, ref_embs, ref_labels, cfg, dev)]

    keep_thr = cfg.win_keep_threshold if cfg.win_keep_threshold is not None else cfg.similarity_threshold
    grid = _window_grid(seg_start, seg_end, cfg.window_s, cfg.hop_s)
    clips = [audio[max(0, int(w0 * sr)) : min(len(audio), int(w1 * sr))] for w0, w1 in grid]
    valid = [i for i, c in enumerate(clips) if len(c) > 0]
    sims: list[float] = [-1.0] * len(grid)  # default -> drop (empty/failed windows)
    try:
        # Batch every window of this segment through ECAPA in one (or few) GPU call(s);
        # embeddings are identical to the per-window path. Falls back below on any error.
        embs = _embed_batch(model, [clips[i] for i in valid], dev)
        for k, i in enumerate(valid):
            sims[i] = _max_cosine(embs[k], ref_embs)[0]
    except Exception as exc:  # noqa: BLE001
        log.warning("batched window embed failed (%d windows): %s; per-window fallback", len(valid), exc)
        for i in valid:
            try:
                sims[i] = _max_cosine(_embed(model, clips[i], dev), ref_embs)[0]
            except Exception as exc2:  # noqa: BLE001
                log.warning("window embed failed (idx %d): %s", i, exc2)
                sims[i] = -1.0

    keep = _bridge_dips([s >= keep_thr for s in sims], cfg.bridge_max_dip_windows)
    spans = _coalesce_keep_windows(grid, keep, cfg.min_keep_span_s)

    if not spans:
        # whole segment is not-him: one drop record (kept for summary accounting / review)
        return [{"start": seg_start, "end": seg_end, "speaker": speaker,
                 "similarity": round(max(sims), 4) if sims else None, "decision": "drop",
                 "refined": True, "parent_start": seg_start, "parent_end": seg_end}]

    out: list[dict[str, Any]] = []
    for sp_start, sp_end, idxs in spans:
        mean_sim = sum(sims[k] for k in idxs) / len(idxs)
        out.append({"start": sp_start, "end": sp_end, "speaker": speaker,
                    "similarity": round(mean_sim, 4), "decision": _decision_for(mean_sim, cfg),
                    "refined": True, "parent_start": seg_start, "parent_end": seg_end})
    return out


def run_identify(
    video_id: str,
    settings: Settings,
    *,
    force: bool = False,
    device: str | None = None,
) -> Path:
    """Run speaker identification on a single video. Returns the JSON path."""
    diar_path = diarize_path_for(video_id, settings)
    if not diar_path.exists():
        raise FileNotFoundError(
            f"diarization output not found: {diar_path} (run pipeline.diarize first)"
        )
    audio_path = audio_path_for(video_id, settings)
    if not audio_path.exists():
        raise FileNotFoundError(f"audio not found: {audio_path}")

    out = identify_path_for(video_id, settings)
    if out.exists() and not force:
        log.info("skip identify for %s (already at %s)", video_id, out)
        return out

    cfg = settings.identify
    dev = device or _resolve_device()
    sr = settings.download.sample_rate_hz

    model = _load_embedding_model(cfg.embedding_model, dev)
    ref_embs, ref_labels = _load_reference_embeddings(settings, model, dev, sr)

    log.info("reading audio %s", audio_path)
    audio = _read_mono_16k(audio_path, sr)

    diar = json.loads(diar_path.read_text(encoding="utf-8"))
    segments_in = diar["segments"]
    log.info("identifying %d segments against %d reference(s)", len(segments_in), len(ref_embs))

    t0 = datetime.now(timezone.utc)
    out_segments: list[dict[str, Any]] = []
    per_speaker_sims: dict[str, list[float]] = {}
    for i, seg in enumerate(segments_in):
        for rec in _identify_segment(seg, audio, sr, model, ref_embs, ref_labels, cfg, dev):
            out_segments.append(rec)
            if rec.get("similarity") is not None and rec["decision"] not in ("too_short", "error"):
                per_speaker_sims.setdefault(rec.get("speaker"), []).append(rec["similarity"])

        if (i + 1) % 250 == 0:
            log.info("  ... %d / %d input segments", i + 1, len(segments_in))

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()

    # Summary
    by_decision: dict[str, int] = {}
    for s in out_segments:
        by_decision[s["decision"]] = by_decision.get(s["decision"], 0) + 1
    kept_speech_s = sum(
        s["end"] - s["start"] for s in out_segments if s["decision"] == "keep"
    )
    total_diar_speech_s = sum(s["end"] - s["start"] for s in segments_in)
    per_speaker_avg = {
        k: round(sum(v) / len(v), 4) for k, v in per_speaker_sims.items() if v
    }

    payload = {
        "video_id": video_id,
        "embedding_model": cfg.embedding_model,
        "reference_set": ref_labels,
        "similarity_threshold": cfg.similarity_threshold,
        "borderline_range": [cfg.borderline_low, cfg.borderline_high],
        "segments": out_segments,
        "summary": {
            "total_segments": len(out_segments),
            "by_decision": by_decision,
            "kept_speech_s": round(kept_speech_s, 3),
            "kept_speech_fraction_of_diarized": (
                round(kept_speech_s / total_diar_speech_s, 4) if total_diar_speech_s else 0.0
            ),
            "per_speaker_avg_similarity": per_speaker_avg,
            "speakers_in_diarization": diar.get("speakers", []),
        },
        "device": dev,
        "elapsed_s": round(elapsed, 1),
        "ran_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(
        "wrote %s — kept %d/%d segments (%.1fs of his-only speech), %.1fs elapsed",
        out,
        by_decision.get("keep", 0),
        len(out_segments),
        kept_speech_s,
        elapsed,
    )
    return out


def discover_video_ids(settings: Settings) -> list[str]:
    """Videos that have BOTH audio and a diarize JSON."""
    audio_ids = {p.stem for p in settings.paths.audio_dir.glob("*.wav")}
    diar_ids = {p.stem for p in settings.paths.diarize_dir.glob("*.json")}
    return sorted(audio_ids & diar_ids)
