"""Reference-pool candidate selector via farthest-point sampling.

Picks N diverse reference clips from segments that already scored high similarity
to the seed reference, so the active reference set covers multiple acoustic
conditions (calm/excited, clean/noisy, indoor/outdoor) instead of N near-clones
of the seed.

Produces:
    data/reference/pool/candidates/{NN}_{source_video}_{start}-{end}.wav
    data/reference/pool/candidates.json   (manifest)

The UI surfaces these candidates for human approval; the approved subset becomes
``data/reference/pool/active.json`` and Stage 4 then uses max-cosine-similarity
across the active set.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import REPO_ROOT, Settings
from pipeline.download import audio_path_for
from pipeline.identify.runner import (
    _embed,
    _load_embedding_model,
    _read_mono_16k,
    _resolve_device,
)

log = logging.getLogger("pipeline.identify.pool")


def pool_dir(settings: Settings) -> Path:
    return REPO_ROOT / "data" / "reference" / "pool"


def candidates_dir(settings: Settings) -> Path:
    return pool_dir(settings) / "candidates"


def candidates_manifest_path(settings: Settings) -> Path:
    return pool_dir(settings) / "candidates.json"


def candidate_embeddings_path(settings: Settings) -> Path:
    """Sidecar npy: per-candidate L2-normalized ECAPA embedding, in candidate_index order."""
    return pool_dir(settings) / "candidate_embeddings.npy"


def active_manifest_path(settings: Settings) -> Path:
    return pool_dir(settings) / "active.json"


def _slice_wav(src: Path, start_s: float, end_s: float, dst: Path, sr: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-i", str(src),
        "-ss", f"{start_s:.3f}",
        "-to", f"{end_s:.3f}",
        "-ac", "1", "-ar", str(sr),
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def select_pool_candidates(
    settings: Settings,
    *,
    pool_size: int = 30,
    min_sim: float = 0.65,
    per_video_cap: int = 60,
    max_per_video: int = 8,
    min_seg_s: float = 3.0,
    max_seg_s: float = 15.0,
    include_seed: bool = True,
    device: str | None = None,
) -> Path:
    """Build a pool of reference candidates via farthest-point sampling.

    Algorithm:
      1. Collect all per-video diarization segments with similarity >= min_sim
         and duration in [min_seg_s, ...] from existing identify/*.json outputs.
      2. Embed each candidate with ECAPA.
      3. Farthest-point sampling in cosine-distance space, seeded with the
         existing reference clip if ``include_seed=True``.
      4. Slice the chosen segments to WAV (16 kHz mono) and write a manifest.

    Returns the path to the manifest JSON.
    """
    import torch

    dev = device or _resolve_device()
    sr = settings.download.sample_rate_hz

    cdir = candidates_dir(settings)
    cdir.mkdir(parents=True, exist_ok=True)
    for p in cdir.glob("*.wav"):
        p.unlink()

    model = _load_embedding_model(settings.identify.embedding_model, dev)

    # Seed embedding
    seed_emb = None
    seed_manifest: dict | None = None
    if include_seed:
        ref_path = settings.paths.reference_clip
        if not ref_path.exists():
            raise FileNotFoundError(f"seed reference not found: {ref_path}")
        seed_audio = _read_mono_16k(ref_path, sr)
        seed_emb = _embed(model, seed_audio, dev)
        seed_dst = cdir / "00_seed.wav"
        shutil.copy(ref_path, seed_dst)
        seed_manifest = {
            "candidate_index": 0,
            "is_seed": True,
            "source_video": "sample00059",
            "source_speaker_label": None,
            "start": 75.0,
            "end": 135.0,
            "duration_s": 60.0,
            "sim_to_seed": 1.0,
            "file": str(seed_dst.relative_to(REPO_ROOT)).replace("\\", "/"),
        }

    # Gather candidates from identify/*.json, with per-video cap to prevent
    # the seed-source video from dominating the pool.
    log.info(
        "scanning identify/*.json for sim>=%.2f, %.1fs<=dur segments (per-video cap %d)",
        min_sim, min_seg_s, per_video_cap,
    )
    candidates: list[dict] = []
    per_video_count: dict[str, int] = {}
    for jp in sorted(settings.paths.identify_dir.glob("*.json")):
        d = json.loads(jp.read_text(encoding="utf-8"))
        vid = d["video_id"]
        # Sort that video's segments by similarity desc; take top-K above min_sim+min_seg_s
        eligible = [
            s for s in d["segments"]
            if s.get("similarity") is not None
            and s["similarity"] >= min_sim
            and (s["end"] - s["start"]) >= min_seg_s
        ]
        eligible.sort(key=lambda s: s["similarity"], reverse=True)
        eligible = eligible[:per_video_cap]
        per_video_count[vid] = len(eligible)
        for s in eligible:
            candidates.append({
                "video_id": vid,
                "start": float(s["start"]),
                "end": float(s["end"]),
                "duration_s": float(s["end"] - s["start"]),
                "sim_to_seed": float(s["similarity"]),
                "source_speaker_label": s["speaker"],
            })
    log.info("collected %d candidate segments  (per-video: %s)", len(candidates), per_video_count)
    if not candidates:
        raise RuntimeError(
            f"no segments with sim>={min_sim}; lower min_sim, or improve the seed reference."
        )

    # Embed all candidates (capping length for efficiency)
    log.info("embedding %d candidate segments (this is the slow part)", len(candidates))
    embs_list = []
    for i, c in enumerate(candidates):
        wav = _read_mono_16k(audio_path_for(c["video_id"], settings), sr)
        s_idx = max(0, int(c["start"] * sr))
        e_idx = min(len(wav), int(c["end"] * sr))
        if (e_idx - s_idx) / sr > max_seg_s:
            e_idx = s_idx + int(max_seg_s * sr)
        emb = _embed(model, wav[s_idx:e_idx], dev)
        embs_list.append(emb)
        if (i + 1) % 100 == 0:
            log.info("  embedded %d / %d", i + 1, len(candidates))

    embs = torch.stack([e.flatten().float() for e in embs_list])  # (N, D)
    embs_n = embs / (embs.norm(dim=1, keepdim=True) + 1e-8)

    # Farthest-point sampling
    if include_seed and seed_emb is not None:
        seed_n = seed_emb.flatten().float()
        seed_n = seed_n / (seed_n.norm() + 1e-8)
        min_dist = (1.0 - embs_n @ seed_n).clone()
        n_to_pick = pool_size - 1
    else:
        # Start from highest-sim candidate
        idx0 = int(max(range(len(candidates)), key=lambda i: candidates[i]["sim_to_seed"]))
        min_dist = (1.0 - embs_n @ embs_n[idx0]).clone()
        first_pick = [idx0]
        n_to_pick = pool_size - 1

    selected_idx: list[int] = [] if include_seed else first_pick  # type: ignore[unbound-local-variable]

    # Track per-source-video selection counts to enforce balance.
    SEED_VIDEO_ID = "sample00059"  # where the seed reference comes from
    src_counts: dict[str, int] = {}
    if include_seed:
        src_counts[SEED_VIDEO_ID] = 1
    else:
        for i in first_pick:  # type: ignore[possibly-undefined]
            src_counts[candidates[i]["video_id"]] = src_counts.get(candidates[i]["video_id"], 0) + 1

    for _ in range(n_to_pick):
        masked = min_dist.clone()
        for s in selected_idx:
            masked[s] = -float("inf")
        # Cap per-source-video: mask candidates whose source video is already at max_per_video.
        for i, c in enumerate(candidates):
            if src_counts.get(c["video_id"], 0) >= max_per_video:
                masked[i] = -float("inf")
        next_idx = int(masked.argmax().item())
        if not torch.isfinite(masked[next_idx]):
            log.warning(
                "only %d distinct candidates available under max_per_video=%d; pool truncated",
                len(selected_idx) + (1 if include_seed else 0), max_per_video,
            )
            break
        selected_idx.append(next_idx)
        src_counts[candidates[next_idx]["video_id"]] = src_counts.get(candidates[next_idx]["video_id"], 0) + 1
        new_dists = 1.0 - embs_n @ embs_n[next_idx]
        min_dist = torch.minimum(min_dist, new_dists)

    log.info("selected %d candidates (seed%s)", len(selected_idx), "+" if include_seed else "")

    # Compute per-candidate diversity score (avg cosine distance to other pool members)
    pool_emb_idxs = selected_idx[:]
    pool_embs = embs_n[pool_emb_idxs]
    if include_seed and seed_emb is not None:
        seed_n = seed_emb.flatten().float()
        seed_n = (seed_n / (seed_n.norm() + 1e-8)).unsqueeze(0)
        pool_embs = torch.cat([seed_n, pool_embs], dim=0)

    div_scores: list[float] = []
    if pool_embs.shape[0] > 1:
        sim_mat = pool_embs @ pool_embs.T  # (P, P)
        dist_mat = 1.0 - sim_mat
        # Each row's mean excluding diagonal
        n = pool_embs.shape[0]
        for r in range(n):
            row = dist_mat[r]
            mask = torch.ones(n, dtype=torch.bool)
            mask[r] = False
            div_scores.append(float(row[mask].mean().item()))

    # Build manifest + slice wavs
    manifest: list[dict] = []
    idx_offset = 1 if include_seed else 0
    if seed_manifest is not None:
        seed_manifest["pool_diversity_score"] = round(div_scores[0], 4) if div_scores else None
        manifest.append(seed_manifest)

    for rank, ci in enumerate(selected_idx):
        c = candidates[ci]
        end_capped = min(c["end"], c["start"] + max_seg_s)
        rec_idx = rank + idx_offset
        fname = f"{rec_idx:02d}_{c['video_id']}_{c['start']:.1f}-{end_capped:.1f}.wav"
        dst = cdir / fname
        try:
            _slice_wav(audio_path_for(c["video_id"], settings), c["start"], end_capped, dst, sr)
        except subprocess.CalledProcessError as exc:
            log.error("ffmpeg slice failed for candidate %d: %s", rec_idx, exc)
            continue
        manifest.append({
            "candidate_index": rec_idx,
            "is_seed": False,
            "source_video": c["video_id"],
            "source_speaker_label": c["source_speaker_label"],
            "start": round(c["start"], 2),
            "end": round(end_capped, 2),
            "duration_s": round(end_capped - c["start"], 2),
            "sim_to_seed": round(c["sim_to_seed"], 4),
            "pool_diversity_score": (
                round(div_scores[rank + idx_offset], 4) if rank + idx_offset < len(div_scores) else None
            ),
            "file": str(dst.relative_to(REPO_ROOT)).replace("\\", "/"),
        })

    payload = {
        "pool_size_requested": pool_size,
        "pool_size_actual": len(manifest),
        "selection_params": {
            "min_sim": min_sim,
            "per_video_cap": per_video_cap,
            "max_per_video": max_per_video,
            "min_seg_s": min_seg_s,
            "max_seg_s": max_seg_s,
            "include_seed": include_seed,
            "method": "farthest_point_sampling",
        },
        "embedding_model": settings.identify.embedding_model,
        "candidates_from_n_videos": len({c["video_id"] for c in candidates}),
        "total_high_sim_segments": len(candidates),
        "candidates": manifest,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path = candidates_manifest_path(settings)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("wrote %s with %d candidates", manifest_path, len(manifest))

    # Save embedding sidecar in candidate_index order so the UI can dynamic-rank
    # proposals by diversity-vs-active without re-embedding.
    import numpy as np

    pool_embs_list = []
    if include_seed and seed_emb is not None:
        seed_n = seed_emb.flatten().float()
        seed_n = seed_n / (seed_n.norm() + 1e-8)
        pool_embs_list.append(seed_n.cpu().numpy().astype(np.float32))
    for ci in selected_idx:
        pool_embs_list.append(embs_n[ci].cpu().numpy().astype(np.float32))
    pool_embs_arr = np.stack(pool_embs_list) if pool_embs_list else np.zeros((0, 0), dtype=np.float32)
    np.save(str(candidate_embeddings_path(settings)), pool_embs_arr)
    log.info("wrote %s shape=%s", candidate_embeddings_path(settings), pool_embs_arr.shape)
    return manifest_path


if __name__ == "__main__":
    import argparse

    from pipeline.config import load_settings

    p = argparse.ArgumentParser(prog="pipeline.identify.pool", description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--pool-size", type=int, default=30)
    p.add_argument("--min-sim", type=float, default=0.65)
    p.add_argument("--per-video-cap", type=int, default=60, help="Max raw candidates from any one video before farthest-point")
    p.add_argument("--max-per-video", type=int, default=8, help="Max candidates from any one video in the FINAL pool")
    p.add_argument("--min-seg-s", type=float, default=3.0)
    p.add_argument("--max-seg-s", type=float, default=15.0)
    p.add_argument("--no-seed", action="store_true", help="Don't auto-include the seed reference")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    s = load_settings(args.config)
    out = select_pool_candidates(
        s,
        pool_size=args.pool_size,
        min_sim=args.min_sim,
        per_video_cap=args.per_video_cap,
        max_per_video=args.max_per_video,
        min_seg_s=args.min_seg_s,
        max_seg_s=args.max_seg_s,
        include_seed=not args.no_seed,
    )
    print(f"\nManifest written to: {out}")
    print(f"Candidate audio files in: {candidates_dir(s)}")
