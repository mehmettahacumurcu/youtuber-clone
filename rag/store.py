"""Local on-disk Qdrant store with BGE-M3 hybrid (dense + sparse, RRF fusion).
Spike (Task 5) confirmed local-mode hybrid works.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from rag.types import Chunk, Hit


def _stable_id(chunk_id: str) -> int:
    import hashlib
    return int(hashlib.sha1(chunk_id.encode()).hexdigest()[:15], 16)


class VectorStore:
    def __init__(self, path: str, collection: str, dense_dim: int = 1024):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=path)
        self.collection = collection
        self.dense_dim = dense_dim

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config={"dense": models.VectorParams(
                size=self.dense_dim, distance=models.Distance.COSINE)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )

    def upsert_chunks(self, chunks: list[Chunk], dense: np.ndarray,
                      sparse: list[dict[int, float]]) -> None:
        points = []
        for ch, dv, sp in zip(chunks, dense, sparse):
            points.append(models.PointStruct(
                id=_stable_id(ch.id),
                vector={
                    "dense": dv.tolist(),
                    "sparse": models.SparseVector(
                        indices=list(sp.keys()), values=list(sp.values())),
                },
                payload={"id": ch.id, "video_id": ch.video_id, "start": ch.start,
                         "end": ch.end, "title": ch.title, "text": ch.text},
            ))
        for i in range(0, len(points), 256):
            self.client.upsert(self.collection, points=points[i:i + 256])

    def hybrid_search(self, query_dense: np.ndarray, query_sparse: dict[int, float],
                      top_k: int) -> list[Hit]:
        res = self.client.query_points(
            self.collection,
            prefetch=[
                models.Prefetch(query=query_dense.tolist(), using="dense", limit=top_k),
                models.Prefetch(query=models.SparseVector(
                    indices=list(query_sparse.keys()), values=list(query_sparse.values())),
                    using="sparse", limit=top_k),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k, with_payload=True,
        )
        hits = []
        for p in res.points:
            pl = p.payload
            hits.append(Hit(chunk=Chunk(
                id=pl["id"], video_id=pl["video_id"], start=pl["start"],
                end=pl["end"], title=pl["title"], text=pl["text"]),
                dense_score=float(p.score or 0.0)))
        return hits
