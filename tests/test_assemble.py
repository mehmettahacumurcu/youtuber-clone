import json

from pipeline.assemble.runner import build_document, dataset_path_for, run_assemble


def _seg(start, end, text):
    return {"start": start, "end": end, "duration_s": end - start, "text": text}


def test_build_document_orders_and_marks_gaps():
    segs = [
        _seg(10.0, 12.0, "ucuncu"),   # out of order; big gap before it
        _seg(0.0, 2.0, "birinci"),
        _seg(2.5, 4.0, "ikinci"),     # gap 0.5 <= 1.0 -> no marker
    ]
    doc = build_document(segs, skip_marker="[SKIP]", skip_gap_s=1.0)
    assert doc == "birinci ikinci\n[SKIP]\nucuncu"


def test_build_document_skips_empty_segments():
    segs = [_seg(0.0, 2.0, "  "), _seg(2.2, 4.0, "tek")]
    assert build_document(segs, skip_marker="[SKIP]", skip_gap_s=1.0) == "tek"


def _write_filter(settings, vid, segs, title=None):
    (settings.paths.filter_dir / f"{vid}.json").write_text(
        json.dumps({"video_id": vid, "segments": segs}), encoding="utf-8")
    if title is not None:
        (settings.paths.meta_dir / f"{vid}.json").write_text(
            json.dumps({"title": title}), encoding="utf-8")


def test_run_assemble_writes_jsonl_per_video(settings):
    _write_filter(settings, "b", [_seg(0.0, 2.0, "be")], title="B baslik")
    _write_filter(settings, "a", [_seg(0.0, 2.0, "a one"), _seg(5.0, 7.0, "a two")])
    out = run_assemble(settings)
    assert out == dataset_path_for(settings)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["video_id"] == "a"          # sorted by id
    assert first["title"] is None
    assert first["n_segments"] == 2
    assert first["text"] == "a one\n[SKIP]\na two"
    assert first["char_count"] == len(first["text"])
    assert json.loads(lines[1])["title"] == "B baslik"


# ---- dual-source (clean vs filter) -------------------------------------------------

def _write_clean(settings, vid, segs):
    (settings.paths.clean_dir / f"{vid}.json").write_text(
        json.dumps({"video_id": vid, "segments": segs}), encoding="utf-8")


def test_source_clean_prefers_clean_text_over_filter(settings):
    # clean/ holds the ASR-corrected text; assemble --source clean must use it, not filter/
    _write_filter(settings, "a", [_seg(0.0, 2.0, "bu işin duaylı Bolton")])
    _write_clean(settings, "a", [_seg(0.0, 2.0, "bu işin duayen Bolton")])
    out = run_assemble(settings, source="clean")
    assert out == dataset_path_for(settings, source="clean")
    assert out.name == "llm_corpus.jsonl"
    rec = json.loads(out.read_text(encoding="utf-8").strip())
    assert rec["text"] == "bu işin duayen Bolton"


def test_source_clean_falls_back_to_filter_when_clean_missing(settings):
    # a video with no clean/ file still appears, sourced from filter/ (universe = filter dir)
    _write_filter(settings, "a", [_seg(0.0, 2.0, "sadece filter")])
    out = run_assemble(settings, source="clean")
    rec = json.loads(out.read_text(encoding="utf-8").strip())
    assert rec["text"] == "sadece filter"


def test_source_filter_writes_raw_path_and_ignores_clean(settings):
    # --source filter is the RAW corpus: reads filter/, writes llm_corpus.raw.jsonl, never clean/
    _write_filter(settings, "a", [_seg(0.0, 2.0, "ham metin")])
    _write_clean(settings, "a", [_seg(0.0, 2.0, "DEĞİŞTİRİLMİŞ")])
    out = run_assemble(settings, source="filter")
    assert out == dataset_path_for(settings, source="filter")
    assert out.name == "llm_corpus.raw.jsonl"
    rec = json.loads(out.read_text(encoding="utf-8").strip())
    assert rec["text"] == "ham metin"
