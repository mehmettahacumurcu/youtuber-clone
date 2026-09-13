"""Tests for the zero-false-positive validation harness used to vet ASR corrections and
structural drop candidates before they ship in config.yaml.

The harness scans every data/filter/*.json segment for a candidate (a literal correction key,
matched at word boundaries, or a regex) and reports EVERY occurrence with video_id + context so
a human/LLM can confirm the wrong form is never legitimate."""
from __future__ import annotations

import json

from scripts.validate_clean_patterns import scan_corpus


def _write_filter(settings, vid, texts):
    p = settings.paths.filter_dir / f"{vid}.json"
    segs = [{"start": float(i), "end": float(i) + 1.0, "text": t} for i, t in enumerate(texts)]
    p.write_text(json.dumps({"video_id": vid, "segments": segs}, ensure_ascii=False),
                 encoding="utf-8")


def test_literal_word_boundary_matches_only_whole_word(settings):
    _write_filter(settings, "v1", [
        "bir duayen geldi buraya",     # match
        "duayenler toplandı",          # substring -> NO match (word-boundary)
        "bugün hava güzel",            # no match
    ])
    hits = scan_corpus(settings.paths.filter_dir, "duayen", is_regex=False)
    assert len(hits) == 1
    assert hits[0]["video_id"] == "v1"
    assert "duayen geldi" in hits[0]["text"]


def test_regex_mode(settings):
    _write_filter(settings, "v2", [
        "14 Temmuz 2019 tarihli",      # match
        "ben bugün anlatacağım",       # no match
    ])
    hits = scan_corpus(settings.paths.filter_dir, r"\d{1,2} (Ocak|Temmuz) \d{4}", is_regex=True)
    assert len(hits) == 1
    assert hits[0]["video_id"] == "v2"


def test_no_match_returns_empty(settings):
    _write_filter(settings, "v3", ["tamamen temiz bir cümle"])
    assert scan_corpus(settings.paths.filter_dir, "yokböylekelime", is_regex=False) == []
