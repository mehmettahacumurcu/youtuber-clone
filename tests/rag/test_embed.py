import pytest

from rag.embed import get_embedder, get_reranker


@pytest.mark.slow
def test_embedder_returns_dense_and_sparse():
    emb = get_embedder("BAAI/bge-m3")
    out = emb.encode(["merhaba dünya", "CHP ve AKP"])
    assert out.dense.shape == (2, 1024)
    assert len(out.sparse) == 2
    assert all(isinstance(d, dict) and d for d in out.sparse)


@pytest.mark.slow
def test_reranker_scores_relevant_higher():
    rr = get_reranker("BAAI/bge-reranker-v2-m3")
    scores = rr.score("Almanya'daki Türkler", [
        "Almanya'daki Türk işçiler ve göçmenler hakkında",
        "futbol maçı sonuçları",
    ])
    assert scores[0] > scores[1]
