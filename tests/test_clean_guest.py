"""Tests for the scoped LLM guest-detection step (pipeline.clean.guest).

The LLM backend is mocked, so these exercise the deterministic core: which videos are
selected, how segments are batched with context, how the prompt is shaped, and how verdicts
are applied (drop non-him, keep unverdicted, profanity-blind, stats + sidecar)."""
from __future__ import annotations

import json

from pipeline.clean import guest


def _write_clean(settings, vid, segments, title="Röportaj", flagged=True, reviewed=False):
    p = settings.paths.clean_dir / f"{vid}.json"
    p.write_text(json.dumps({
        "video_id": vid, "title": title,
        "input_segments": len(segments), "kept_segments": len(segments),
        "dropped_by_reason": {}, "flagged_for_review": flagged,
        "flag_reasons": ["title:röportaj"] if flagged else [],
        "guest_reviewed": reviewed, "segments": segments,
    }, ensure_ascii=False), encoding="utf-8")
    return p


def _seg(text, start):
    return {"start": start, "end": start + 3.0, "similarity": 0.55, "duration_s": 3.0, "text": text}


def test_flagged_videos_selects_unreviewed_flagged(settings):
    _write_clean(settings, "flag1", [_seg("a", 0)], flagged=True)
    _write_clean(settings, "noflag", [_seg("b", 0)], flagged=False)
    _write_clean(settings, "done", [_seg("c", 0)], flagged=True, reviewed=True)
    assert guest.flagged_videos(settings) == ["flag1"]


def test_build_batches_windowing_and_context(settings):
    settings.review.batch_size = 2
    settings.review.context_neighbors = 1
    segs = [_seg(f"s{i}", i * 3.0) for i in range(5)]
    _write_clean(settings, "v", segs)
    batches = guest.build_batches("v", settings)
    assert [len(b["segments"]) for b in batches] == [2, 2, 1]  # 5 segs, bs=2
    # indices preserved across batches
    assert [it["idx"] for b in batches for it in b["segments"]] == [0, 1, 2, 3, 4]
    # context: segment 2 has s1 before and s3 after
    seg2 = batches[1]["segments"][0]
    assert seg2["idx"] == 2
    assert seg2["context_before"] == ["s1"] and seg2["context_after"] == ["s3"]


def test_build_prompt_has_labels_title_and_json(settings):
    segs = [_seg("merhaba", 0), _seg("küfür siktir", 3)]
    _write_clean(settings, "v", segs, title="Büyük Röportaj")
    prompt = guest.build_prompt(guest.build_batches("v", settings)[0])
    assert "Büyük Röportaj" in prompt
    assert "him | guest | read_aloud | other" in prompt
    assert '"labels"' in prompt
    assert "küfür siktir" in prompt  # profanity present in the text shown to the LLM


def test_apply_verdicts_drops_non_him_keeps_unverdicted(settings):
    segs = [_seg("him talk", 0), _seg("guest talk", 3), _seg("him again", 6), _seg("news copy", 9)]
    _write_clean(settings, "v", segs)
    labels = {
        0: {"label": "him", "reason": "x"},
        1: {"label": "guest", "reason": "interviewee"},
        3: {"label": "read_aloud", "reason": "newswire"},
        # idx 2 intentionally has no verdict -> kept
    }
    guest.apply_verdicts("v", settings, labels)
    doc = json.loads((settings.paths.clean_dir / "v.json").read_text(encoding="utf-8"))
    kept = [s["text"] for s in doc["segments"]]
    assert kept == ["him talk", "him again"]
    assert doc["dropped_by_reason"] == {"guest": 1, "read_aloud": 1}
    assert doc["guest_reviewed"] is True
    review = json.loads((settings.paths.clean_dir / "v.review.json").read_text(encoding="utf-8"))
    assert {d["reason"] for d in review["dropped"]} == {"guest", "read_aloud"}


def test_run_guest_review_end_to_end_mock(settings):
    segs = [_seg("ben anlatıyorum", 0), _seg("GUEST cevap", 3), _seg("ben devam", 6)]
    _write_clean(settings, "v", segs)

    def mock_adjudicate(batch):
        return [
            {"idx": it["idx"], "label": "guest" if "GUEST" in it["text"] else "him", "reason": "m"}
            for it in batch["segments"]
        ]

    dropped = guest.run_guest_review(settings, mock_adjudicate, video_ids=["v"])
    assert dropped == {"v": 1}
    doc = json.loads((settings.paths.clean_dir / "v.json").read_text(encoding="utf-8"))
    assert [s["text"] for s in doc["segments"]] == ["ben anlatıyorum", "ben devam"]
