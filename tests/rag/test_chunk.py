import json
from pathlib import Path

from rag.chunk import chunk_corpus, estimate_tokens


def _write_video(clean_dir: Path, meta_dir: Path, vid: str, segments, title="Başlık"):
    clean_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    (clean_dir / f"{vid}.json").write_text(
        json.dumps({"video_id": vid, "segments": segments}), encoding="utf-8")
    (meta_dir / f"{vid}.json").write_text(
        json.dumps({"title": title}), encoding="utf-8")


def test_estimate_tokens_is_wordcount_scaled():
    assert estimate_tokens("bir iki üç") == 4  # 3 words * 1.4 -> 4.2 -> int 4


def test_groups_segments_into_token_budgeted_chunks(tmp_path):
    clean, meta = tmp_path / "clean", tmp_path / "meta"
    segs = [{"start": float(i), "end": float(i) + 0.9,
             "text": " ".join(["kelime"] * 10)} for i in range(6)]
    _write_video(clean, meta, "vid1", segs)
    chunks = list(chunk_corpus(clean, meta, chunk_tokens=30, overlap=0))
    assert len(chunks) >= 3
    assert all(c.video_id == "vid1" for c in chunks)
    assert all(c.title == "Başlık" for c in chunks)
    assert chunks[0].start == 0.0
    assert chunks[0].end >= 0.9
    assert chunks[0].id == "vid1::0" and chunks[1].id == "vid1::1"


def test_title_falls_back_to_video_id_when_meta_missing(tmp_path):
    clean, meta = tmp_path / "clean", tmp_path / "meta"
    clean.mkdir(parents=True)
    (clean / "novid.json").write_text(
        json.dumps({"video_id": "novid",
                    "segments": [{"start": 0.0, "end": 1.0, "text": "selam"}]}),
        encoding="utf-8")
    chunks = list(chunk_corpus(clean, meta, chunk_tokens=500, overlap=50))
    assert chunks[0].title == "novid"


def test_profanity_is_preserved_in_chunks(tmp_path):
    clean, meta = tmp_path / "clean", tmp_path / "meta"
    _write_video(clean, meta, "vid1",
                 [{"start": 0.0, "end": 1.0, "text": "amına koyayım bu iş"}])
    chunks = list(chunk_corpus(clean, meta, chunk_tokens=500, overlap=50))
    assert "amına koyayım" in chunks[0].text


def test_overlap_repeats_tail_segments(tmp_path):
    clean, meta = tmp_path / "clean", tmp_path / "meta"
    segs = [{"start": float(i), "end": float(i) + 0.9, "text": "kelime " * 10}
            for i in range(6)]
    _write_video(clean, meta, "vid1", segs)
    chunks = list(chunk_corpus(clean, meta, chunk_tokens=30, overlap=15))
    no_ov = list(chunk_corpus(clean, meta, chunk_tokens=30, overlap=0))
    assert sum(len(c.text) for c in chunks) > sum(len(c.text) for c in no_ov)
