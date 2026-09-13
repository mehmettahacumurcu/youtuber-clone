"""pyannote/speaker-diarization-3.1 over a stage-1 WAV.

Output schema (``data/diarize/{video_id}.json``)::

    {
      "video_id": "...",
      "model": "pyannote/speaker-diarization-3.1",
      "duration_s": 2772.2,
      "num_speakers": 2,
      "speakers": ["SPEAKER_00", "SPEAKER_01"],
      "speaker_speech_s": {"SPEAKER_00": 1700.1, "SPEAKER_01": 72.4},
      "segments": [{"start": 0.5, "end": 3.2, "speaker": "SPEAKER_00"}, ...],
      "config": {"num_speakers": null, "min_speakers": null, "max_speakers": null},
      "device": "cuda",
      "ran_at_utc": "..."
    }

Note: pyannote labels speakers anonymously (``SPEAKER_00`` etc.). Stage 4
(speaker identification) is what maps these to "him vs. not-him".
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from pipeline.config import Settings
from pipeline.download import audio_path_for

log = logging.getLogger("pipeline.diarize")


def diarize_path_for(video_id: str, settings: Settings) -> Path:
    return settings.paths.diarize_dir / f"{video_id}.json"


def _patch_torch_load_for_pyannote() -> None:
    """PyTorch 2.6+ defaults torch.load to weights_only=True; pyannote 3.x
    checkpoints embed non-tensor globals (TorchVersion, OmegaConf nodes, etc.)
    and fail under that mode. We trust HF-hosted pyannote artifacts.

    Two layers of defense:
    1. Allowlist the globals pyannote ships in checkpoints, so weights_only=True
       loads succeed (covers callers that explicitly pass weights_only=True).
    2. Patch ``torch.load`` so callers that don't pass it default to False.
    """
    import torch as _torch

    if getattr(_torch.load, "_pyannote_patched", False):
        return

    # 1) Allowlist known globals embedded in pyannote / lightning / omegaconf checkpoints.
    safe_globals: list = []
    try:
        from torch.torch_version import TorchVersion
        safe_globals.append(TorchVersion)
    except Exception:
        pass
    try:
        from omegaconf.dictconfig import DictConfig
        from omegaconf.listconfig import ListConfig
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.nodes import AnyNode
        safe_globals.extend([DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode])
    except Exception:
        pass
    try:
        import numpy as _np
        safe_globals.extend([_np.ndarray, _np.dtype, _np.core.multiarray._reconstruct])
    except Exception:
        pass
    if safe_globals:
        try:
            _torch.serialization.add_safe_globals(safe_globals)
        except Exception:
            pass

    # 2) Force-flip for callers that pass weights_only=True (pyannote 3.x does).
    _orig = _torch.load

    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig(*args, **kwargs)

    _patched_load._pyannote_patched = True  # type: ignore[attr-defined]
    _torch.load = _patched_load  # type: ignore[assignment]
    log.debug("torch.load patched to force weights_only=False for pyannote checkpoints")


@lru_cache(maxsize=1)
def _load_pipeline(model: str, device: str):
    """Cache the pyannote pipeline across multiple calls in one process.

    Loading the pipeline is multi-second; calling diarize back-to-back on N
    videos should not pay that cost N times.
    """
    import torch
    _patch_torch_load_for_pyannote()
    from pyannote.audio import Pipeline

    log.info("loading pyannote pipeline %s (this may download weights on first run)", model)
    pipe = Pipeline.from_pretrained(model)
    if pipe is None:
        raise RuntimeError(
            f"pyannote returned None for {model}. "
            "Most common cause: HF terms not accepted for "
            "pyannote/speaker-diarization-3.1 AND pyannote/segmentation-3.0 with the logged-in account. "
            "Visit those pages, click Agree, then retry."
        )
    pipe.to(torch.device(device))
    return pipe


def _resolve_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# --- Segmentation-only path (clustering-free) ---------------------------------------
# Stage 4 (identify) only consumes each segment's (start, end) boundary and computes its
# own ECAPA similarity per segment; pyannote's global speaker labels are unused metadata
# (audited: nothing downstream gates on them). So we can skip the embedding + single-
# threaded CPU AgglomerativeClustering entirely and run ONLY the GPU segmentation model.
# That removes the CPU bottleneck. Tradeoff: speakers are collapsed (no turn labels), so a
# segment can straddle a speaker change when there is no pause between turns — the
# similarity gate still drops mostly-not-him segments. Accepted per project decision.


def _split_region(start: float, end: float, max_segment_s: float | None) -> list[tuple[float, float]]:
    """Split [start, end] into contiguous equal chunks each <= max_segment_s.

    Equal-width splitting avoids leaving a tiny remainder. A region already within the cap
    is returned unchanged. Splitting matters: a long pure-monologue region would otherwise
    transcribe fine then get dropped by filter as bad_length (> filter.max_segment_s).
    """
    span = end - start
    if not max_segment_s or max_segment_s <= 0 or span <= max_segment_s:
        return [(start, end)]
    n = math.ceil(span / max_segment_s)
    step = span / n
    out: list[tuple[float, float]] = []
    for i in range(n):
        a = start + i * step
        b = end if i == n - 1 else start + (i + 1) * step
        out.append((a, b))
    return out


def _regions_to_segments(
    regions: list[tuple[float, float]],
    *,
    max_segment_s: float | None,
    label: str = "SPEECH",
) -> list[dict]:
    """Turn merged speech regions into the diarize segment schema, splitting long ones.

    Every segment gets the same placeholder ``label`` — segmentation assigns no speaker
    identity, and the label is metadata only (never gates a downstream decision).
    """
    segs: list[dict] = []
    for start, end in regions:
        for a, b in _split_region(float(start), float(end), max_segment_s):
            if b > a:
                segs.append({"start": round(a, 3), "end": round(b, 3), "speaker": label})
    return segs


@lru_cache(maxsize=1)
def _load_segmentation_pipeline(model_name: str, device: str, min_on: float, min_off: float):
    """Cache a clustering-free VoiceActivityDetection pipeline built on segmentation-3.0.

    Uses pyannote's stable high-level VAD pipeline (GPU) — no embedding, no clustering.
    segmentation-3.0 is a powerset model, so onset/offset are fixed at 0.5 internally and
    only min_duration_on/off are tunable.
    """
    import torch

    _patch_torch_load_for_pyannote()
    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection

    log.info("loading segmentation model %s on %s (clustering-free)", model_name, device)
    model = Model.from_pretrained(model_name)  # picks up HF_TOKEN from env, like the full pipeline
    if model is None:
        raise RuntimeError(
            f"pyannote returned None for {model_name}. Most common cause: HF terms not accepted "
            "for pyannote/segmentation-3.0 with the logged-in account, or HF_TOKEN unset."
        )
    pipe = VoiceActivityDetection(segmentation=model)
    pipe.instantiate({"min_duration_on": min_on, "min_duration_off": min_off})
    pipe.to(torch.device(device))
    return pipe


def _diarize_via_segmentation(video_id: str, audio: Path, dev: str, cfg) -> dict:
    """Clustering-free Stage 3: GPU segmentation -> merged speech regions -> bounded segments."""
    pipe = _load_segmentation_pipeline(
        cfg.segmentation_model, dev, cfg.seg_min_duration_on, cfg.seg_min_duration_off
    )
    # Bigger batch better saturates the GPU; attr name guarded across pyannote versions.
    if hasattr(pipe, "segmentation_batch_size"):
        pipe.segmentation_batch_size = cfg.segmentation_batch_size

    log.info("segmentation-only diarize of %s on %s", audio, dev)
    t0 = datetime.now(timezone.utc)
    annotation = pipe(str(audio))
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()

    regions = [(float(s.start), float(s.end)) for s in annotation.get_timeline().support()]
    segments = _regions_to_segments(regions, max_segment_s=cfg.seg_max_segment_s)
    total_speech = sum(e - s for s, e in regions)
    duration_s = max((seg["end"] for seg in segments), default=0.0)

    log.info(
        "segmentation diarize %s — %d regions -> %d segments, %.1fs speech, %.1fs elapsed",
        video_id, len(regions), len(segments), total_speech, elapsed,
    )
    return {
        "video_id": video_id,
        "model": cfg.segmentation_model,
        "method": "segmentation",
        "duration_s": round(duration_s, 3),
        "num_speakers": None,            # not computed without clustering
        "speakers": ["SPEECH"],
        "speaker_speech_s": {"SPEECH": round(total_speech, 3)},
        "segments": segments,
        "config": {
            "method": "segmentation",
            "seg_min_duration_on": cfg.seg_min_duration_on,
            "seg_min_duration_off": cfg.seg_min_duration_off,
            "seg_max_segment_s": cfg.seg_max_segment_s,
        },
        "device": dev,
        "elapsed_s": round(elapsed, 1),
        "ran_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _diarize_via_clustering(video_id: str, audio: Path, dev: str, cfg) -> dict:
    """Full pyannote speaker-diarization-3.1 (segmentation + embedding + CPU clustering)."""
    pipeline = _load_pipeline(cfg.model, dev)

    # Bigger batches better saturate the GPU and cut per-window CPU glue in the
    # segmentation/embedding phases. Clustering stays CPU-bound — that part is inherent
    # to pyannote. hasattr-guarded so a version without these attrs won't crash.
    for attr, val in (
        ("segmentation_batch_size", cfg.segmentation_batch_size),
        ("embedding_batch_size", cfg.embedding_batch_size),
    ):
        if hasattr(pipeline, attr):
            setattr(pipeline, attr, val)

    pipe_kwargs: dict = {}
    if cfg.num_speakers is not None:
        pipe_kwargs["num_speakers"] = cfg.num_speakers
    if cfg.min_speakers is not None:
        pipe_kwargs["min_speakers"] = cfg.min_speakers
    if cfg.max_speakers is not None:
        pipe_kwargs["max_speakers"] = cfg.max_speakers

    log.info("diarizing %s on %s (kwargs=%s)", audio, dev, pipe_kwargs)
    t0 = datetime.now(timezone.utc)
    diarization = pipeline(str(audio), **pipe_kwargs)
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()

    segments: list[dict] = []
    speaker_time: dict[str, float] = {}
    for turn, _track, speaker in diarization.itertracks(yield_label=True):
        seg = {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
        segments.append(seg)
        speaker_time[speaker] = speaker_time.get(speaker, 0.0) + (seg["end"] - seg["start"])

    speakers = sorted(speaker_time.keys())
    duration_s = max((s["end"] for s in segments), default=0.0)

    log.info(
        "clustering diarize %s — %d speakers, %d segments, %.1fs elapsed",
        video_id, len(speakers), len(segments), elapsed,
    )
    return {
        "video_id": video_id,
        "model": cfg.model,
        "method": "diarization",
        "duration_s": round(duration_s, 3),
        "num_speakers": len(speakers),
        "speakers": speakers,
        "speaker_speech_s": {k: round(v, 3) for k, v in speaker_time.items()},
        "segments": segments,
        "config": {
            "num_speakers": cfg.num_speakers,
            "min_speakers": cfg.min_speakers,
            "max_speakers": cfg.max_speakers,
        },
        "device": dev,
        "elapsed_s": round(elapsed, 1),
        "ran_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def run_diarization(
    video_id: str,
    settings: Settings,
    *,
    force: bool = False,
    device: str | None = None,
) -> Path:
    """Diarize a single video. Returns the JSON path.

    ``diarize.method`` selects the implementation: ``"segmentation"`` (clustering-free,
    GPU-only) or ``"diarization"`` (full pyannote with CPU clustering). Both produce the
    same output schema, so downstream stages are method-agnostic.
    """
    audio = audio_path_for(video_id, settings)
    if not audio.exists():
        raise FileNotFoundError(f"audio not found: {audio} (run pipeline.download first)")

    out = diarize_path_for(video_id, settings)
    if out.exists() and not force:
        log.info("skip diarization for %s (already at %s)", video_id, out)
        return out

    dev = device or _resolve_device()
    cfg = settings.diarize
    if cfg.method == "segmentation":
        payload = _diarize_via_segmentation(video_id, audio, dev, cfg)
    elif cfg.method == "diarization":
        payload = _diarize_via_clustering(video_id, audio, dev, cfg)
    else:
        raise ValueError(
            f"unknown diarize.method {cfg.method!r} (expected 'diarization' or 'segmentation')"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("wrote %s (method=%s) — %d segments", out, cfg.method, len(payload["segments"]))
    return out


def discover_video_ids(settings: Settings) -> list[str]:
    return sorted(p.stem for p in settings.paths.audio_dir.glob("*.wav"))
