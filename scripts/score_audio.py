"""Score arbitrary audio file(s) against the seed + active reference pool (READ-ONLY).

Diagnostic: embed each file with ECAPA and report cosine similarity to the seed clip and
to the active pool (max = what Stage-4 identify would assign; mean = average closeness).
Useful for testing how robustly the pool recognizes augmented / held-out clips of him.
Resamples any sample-rate/channel layout to 16 kHz mono first (via ffmpeg). Does NOT modify
the pool.

    python scripts/score_audio.py seed_chorus.wav seed_distortedlimited.wav seed_stereo.wav
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import REPO_ROOT, load_settings  # noqa: E402
from pipeline.identify.runner import (  # noqa: E402
    _cosine, _embed, _load_embedding_model, _read_mono_16k, _resolve_device,
)
from pipeline.identify.pool import candidates_manifest_path  # noqa: E402
from ui.state import load_active_pool  # noqa: E402

THRESHOLD_NOTE = 0.50  # identify.similarity_threshold


def to_16k_mono(src: Path, sr: int) -> Path:
    tmp = Path(tempfile.gettempdir()) / f"score_{src.stem}_16k.wav"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(src), "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(tmp)],
                   check=True)
    return tmp


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="score_audio", description=__doc__)
    p.add_argument("files", nargs="+", help="audio files to score")
    p.add_argument("--config", default=None)
    p.add_argument("--exclude", action="append", default=[],
                   help="pool wav basename(s) to EXCLUDE from the comparison (repeatable)")
    args = p.parse_args(argv)
    exclude = set(args.exclude)

    settings = load_settings(args.config)
    sr = settings.download.sample_rate_hz
    dev = _resolve_device()
    model = _load_embedding_model(settings.identify.embedding_model, dev)

    # seed embedding
    seed_emb = _embed(model, _read_mono_16k(settings.paths.reference_clip, sr), dev)

    # active pool embeddings (label them by source so we can show the best match)
    doc = json.loads(candidates_manifest_path(settings).read_text(encoding="utf-8"))
    by_idx = {c["candidate_index"]: c for c in doc["candidates"]}
    active = load_active_pool(settings)["active_indices"]
    pool = []  # (label, emb)
    for i in active:
        c = by_idx.get(i)
        if not c:
            continue
        f = REPO_ROOT / c["file"]
        if Path(c["file"]).name in exclude:
            continue
        if f.exists():
            pool.append((Path(c["file"]).name, _embed(model, _read_mono_16k(f, sr), dev)))

    print(f"scoring {len(args.files)} file(s) vs seed + {len(pool)} active pool members "
          f"(identify keeps >= {THRESHOLD_NOTE})\n")
    print(f"{'file':<34}{'sim_seed':>10}{'max_pool':>10}{'mean_pool':>11}  best_pool_match")
    print("-" * 100)
    for fp in args.files:
        src = Path(fp)
        if not src.exists():
            print(f"{src.name:<34}  NOT FOUND")
            continue
        emb = _embed(model, _read_mono_16k(to_16k_mono(src, sr), sr), dev)
        sim_seed = _cosine(emb, seed_emb)
        sims = [(lbl, _cosine(emb, e)) for lbl, e in pool]
        sims.sort(key=lambda x: x[1], reverse=True)
        max_pool = sims[0][1] if sims else float("nan")
        mean_pool = sum(s for _, s in sims) / len(sims) if sims else float("nan")
        verdict = "KEEP" if max_pool >= THRESHOLD_NOTE else "DROP"
        print(f"{src.name:<34}{sim_seed:>10.3f}{max_pool:>10.3f}{mean_pool:>11.3f}  "
              f"{sims[0][0] if sims else '-'}  [{verdict}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
