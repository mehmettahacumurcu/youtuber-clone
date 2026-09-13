import numpy as np

from rag.store import VectorStore
from rag.types import Chunk


def _chunk(cid, text):
    return Chunk(id=cid, video_id="v1", start=0.0, end=1.0, title="T", text=text)


def test_upsert_then_hybrid_search_returns_relevant(tmp_path):
    store = VectorStore(path=str(tmp_path / "q"), collection="t", dense_dim=4)
    store.ensure_collection()
    chunks = [_chunk("a", "alfa"), _chunk("b", "beta"), _chunk("c", "gama")]
    dense = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    sparse = [{10: 1.0}, {20: 1.0}, {30: 1.0}]
    store.upsert_chunks(chunks, dense, sparse)
    hits = store.hybrid_search(query_dense=np.array([1, 0, 0, 0], dtype=np.float32),
                               query_sparse={10: 1.0}, top_k=3)
    assert hits[0].chunk.id == "a"
    assert {h.chunk.id for h in hits} == {"a", "b", "c"}
    assert hits[0].chunk.text == "alfa"
