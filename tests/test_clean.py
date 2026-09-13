"""Tests for Stage 6.5 clean: structural drops, ASR corrections, video flagging, and the
hard invariant that profanity is never content-dropped."""
from __future__ import annotations

import json
import re

from pipeline.clean.runner import (
    apply_corrections,
    clean_path_for,
    run_clean,
    structural_drop_reason,
)


# ---- pure helpers -----------------------------------------------------------------

def test_apply_corrections_word_boundary():
    # whole-word replace, never a substring
    assert apply_corrections("the duayen spoke", {"duayen": "duaylı"}) == "the duaylı spoke"
    assert apply_corrections("duayenler geldi", {"duayen": "duaylı"}) == "duayenler geldi"  # substring untouched
    assert apply_corrections("no match here", {"duayen": "duaylı"}) == "no match here"


def test_structural_drop_reason():
    pats = [re.compile(r"\d{1,2} (Ocak|Temmuz) \d{4}")]
    assert structural_drop_reason("14 Temmuz 2019 tarihli açıklama", pats) == "read_aloud"
    assert structural_drop_reason("bugün hava çok güzel", pats) is None


# ---- run_clean integration --------------------------------------------------------

def _write_filter(settings, vid, segments):
    p = settings.paths.filter_dir / f"{vid}.json"
    p.write_text(json.dumps({"video_id": vid, "segments": segments}, ensure_ascii=False),
                 encoding="utf-8")


def _write_meta(settings, vid, title):
    p = settings.paths.meta_dir / f"{vid}.json"
    p.write_text(json.dumps({"id": vid, "title": title}, ensure_ascii=False), encoding="utf-8")


def _seg(text, start=0.0, sim=0.55):
    return {"start": start, "end": start + 3.0, "similarity": sim, "duration_s": 3.0, "text": text}


def test_run_clean_drops_structural_keeps_speech_and_profanity(settings):
    settings.clean.drop_patterns = [r"\d{1,2} (Ocak|Temmuz) \d{4}"]
    vid = "vid1"
    _write_filter(settings, vid, [
        _seg("14 Temmuz 2019 tarihinde bir açıklama yapıldı"),  # read-aloud dateline -> drop
        _seg("ben bugün size bir şey anlatacağım"),             # his speech -> keep
        _seg("bu tam bir saçmalık, siktir et"),                 # profanity -> MUST keep
    ])
    out = run_clean(vid, settings, force=True)
    doc = json.loads(out.read_text(encoding="utf-8"))
    texts = [s["text"] for s in doc["segments"]]

    assert doc["kept_segments"] == 2
    assert doc["dropped_by_reason"] == {"read_aloud": 1}
    assert "14 Temmuz 2019 tarihinde bir açıklama yapıldı" not in texts  # dropped
    assert any("anlatacağım" in t for t in texts)                        # speech kept
    assert any("siktir" in t for t in texts), "profanity must never be content-dropped"


def test_run_clean_applies_corrections(settings):
    settings.clean.corrections = {"İttihatçı": "ittihatçı"}
    vid = "vid2"
    _write_filter(settings, vid, [_seg("o bir İttihatçı idi")])
    out = run_clean(vid, settings, force=True)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["corrected_segments"] == 1
    assert doc["segments"][0]["text"] == "o bir ittihatçı idi"


def test_run_clean_flags_interview_by_title(settings):
    settings.clean.title_flags = ["röportaj"]
    vid = "vid3"
    _write_filter(settings, vid, [_seg("normal konuşma")])
    _write_meta(settings, vid, "Büyük Röportaj: Konuğumuz...")  # title lowercased -> matches
    out = run_clean(vid, settings, force=True)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["flagged_for_review"] is True
    assert any(r.startswith("title:") for r in doc["flag_reasons"])


def test_run_clean_flags_by_drop_ratio(settings):
    settings.clean.drop_patterns = [r"DROP"]
    settings.clean.max_drop_ratio_flag = 0.5
    vid = "vid4"
    _write_filter(settings, vid, [_seg("DROP me"), _seg("DROP me too"), _seg("keep this")])
    out = run_clean(vid, settings, force=True)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["pct_dropped"] > 50.0
    assert any(r.startswith("drop_ratio:") for r in doc["flag_reasons"])


def test_run_clean_idempotent_and_review_sidecar(settings):
    settings.clean.drop_patterns = [r"DROP"]
    vid = "vid5"
    _write_filter(settings, vid, [_seg("DROP this"), _seg("keep this")])
    out = run_clean(vid, settings, force=True)
    # review sidecar emitted with the dropped span
    review = out.with_suffix(".review.json")
    assert review.exists()
    rdoc = json.loads(review.read_text(encoding="utf-8"))
    assert len(rdoc["dropped"]) == 1 and rdoc["dropped"][0]["reason"] == "read_aloud"
    # idempotent: a second run without force does not raise and returns the same path
    assert run_clean(vid, settings, force=False) == out
