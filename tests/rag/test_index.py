import json

import pytest

from rag.index import build_index


@pytest.mark.slow
def test_build_index_populates_store(tmp_path):
    clean, meta = tmp_path / "clean", tmp_path / "meta"
    clean.mkdir(); meta.mkdir()
    (clean / "v1.json").write_text(json.dumps({"video_id": "v1", "segments": [
        {"start": 0.0, "end": 4.0, "text": "Almanya'daki Türk işçiler hakkında konuşalım."}]}),
        encoding="utf-8")
    (meta / "v1.json").write_text(json.dumps({"title": "Göçmenler"}), encoding="utf-8")
    n = build_index(clean_dir=clean, meta_dir=meta, store_path=str(tmp_path / "q"),
                    collection="t", embedder_name="BAAI/bge-m3",
                    chunk_tokens=500, overlap=50)
    assert n >= 1
