"""Build the CLEAN his-only opinion index (a NEW collection, default `speaker_clean`) from the
his-only solo tiers only (T1 by default). The contaminated `speaker` collection is left intact for
A/B comparison.

Per rag/hisonly_filter.py: per-segment ECAPA similarity CANNOT separate him from guests, so the
only reliable his-only axis is the video FORMAT (title) — multi-speaker formats are excluded
wholesale. T1 = solo documentary/stream-commentary/monologue (~281 vids, ~2.2M tok).

Stop Ollama first (frees VRAM for BGE-M3 on the 8GB GPU), then:
  .venv\\Scripts\\python.exe -m rag.index_hisonly --tiers T1 --collection speaker_clean --chunk-tokens 600
"""
from __future__ import annotations

import argparse

from pipeline.config import load_settings
from rag.hisonly_filter import select_videos
from rag.index import build_index


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the clean his-only opinion index.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tiers", default="T1", help="comma list of tiers to index, e.g. 'T1' or 'T1,T2'")
    ap.add_argument("--collection", default="speaker_clean")
    ap.add_argument("--chunk-tokens", type=int, default=600,
                    help="opinions span multiple segments — 600 keeps a take coherent (vs 500 default)")
    ap.add_argument("--overlap", type=int, default=128)
    args = ap.parse_args()

    s = load_settings(args.config)
    tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip())
    sel = select_videos(s.paths.clean_dir, s.paths.meta_dir, tiers=tiers)
    print(f"his-only videos selected ({'+'.join(tiers)}): {len(sel)}")
    if not sel:
        raise SystemExit("No videos selected — check the manifest / clean_dir / meta_dir.")

    n = build_index(clean_dir=s.paths.clean_dir, meta_dir=s.paths.meta_dir,
                    store_path=str(s.rag.store_path), collection=args.collection,
                    embedder_name=s.rag.embedder, chunk_tokens=args.chunk_tokens,
                    overlap=args.overlap, select=sel)
    print(f"\nclean index ready -> {args.collection}: {n} chunks from {len(sel)} his-only videos "
          f"(dirty 'speaker' collection untouched for A/B)")


if __name__ == "__main__":
    main()
