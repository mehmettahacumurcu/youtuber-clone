import json

import pytest

from rag.index import build_index
from rag.retrieve import retrieve


@pytest.mark.slow
def test_retrieve_ranks_relevant_chunk_first(tmp_path):
    clean, meta = tmp_path / "clean", tmp_path / "meta"
    clean.mkdir(); meta.mkdir()
    docs = {
        "v1": "Almanya'daki Türk işçiler ve gurbetçiler hakkında uzun uzun konuştum.",
        "v2": "Bugün hava durumu ve yemek tarifleri üzerine sohbet edelim.",
    }
    for vid, text in docs.items():
        (clean / f"{vid}.json").write_text(json.dumps({"video_id": vid, "segments": [
            {"start": 0.0, "end": 5.0, "text": text}]}), encoding="utf-8")
        (meta / f"{vid}.json").write_text(json.dumps({"title": vid}), encoding="utf-8")
    build_index(clean, meta, str(tmp_path / "q"), "t", "BAAI/bge-m3", 500, 50)

    hits = retrieve("Almanya'daki Türkler", store_path=str(tmp_path / "q"), collection="t",
                    embedder_name="BAAI/bge-m3", reranker_name="BAAI/bge-reranker-v2-m3",
                    top_k=20, top_n=2)
    assert hits[0].chunk.video_id == "v1"
    assert hits[0].rerank_score > hits[-1].rerank_score
