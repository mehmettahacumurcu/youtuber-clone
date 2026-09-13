"""Build the RAG index: cleaned segments -> chunks -> BGE-M3 -> local Qdrant. Idempotent."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline.config import load_settings
from rag.chunk import chunk_corpus
from rag.embed import get_embedder
from rag.store import VectorStore


def build_index(clean_dir: Path, meta_dir: Path, store_path: str, collection: str,
                embedder_name: str, chunk_tokens: int, overlap: int,
                batch: int = 64, select: dict[str, str] | None = None) -> int:
    chunks = list(chunk_corpus(clean_dir, meta_dir, chunk_tokens, overlap, select=select))
    if not chunks:
        raise SystemExit(f"No chunks found under {clean_dir}. Pull the corpus or check rag.source.")
    emb = get_embedder(embedder_name)
    store = VectorStore(path=store_path, collection=collection, dense_dim=1024)
    store.ensure_collection()
    total = 0
    for i in range(0, len(chunks), batch):
        part = chunks[i:i + batch]
        out = emb.encode([c.text for c in part])
        store.upsert_chunks(part, np.asarray(out.dense), out.sparse)
        total += len(part)
        print(f"  indexed {total}/{len(chunks)}")
    print(f"Done: {total} chunks -> {store_path}::{collection}")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the Speaker RAG index.")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    s = load_settings(args.config)
    src = s.paths.clean_dir if s.rag.source == "clean" else s.paths.filter_dir
    build_index(clean_dir=src, meta_dir=s.paths.meta_dir,
                store_path=str(s.rag.store_path), collection=s.rag.collection,
                embedder_name=s.rag.embedder, chunk_tokens=s.rag.chunk_tokens,
                overlap=s.rag.chunk_overlap)


if __name__ == "__main__":
    main()
