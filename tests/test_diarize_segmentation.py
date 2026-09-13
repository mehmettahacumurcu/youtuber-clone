"""Unit tests for the clustering-free segmentation path's pure logic.

Only the boundary math is tested here (no pyannote / GPU): given speech regions, the
segment builder must split long regions under the cap, keep contiguous coverage, and tag
every segment with the placeholder speaker label.
"""
from pipeline.diarize.runner import _regions_to_segments, _split_region


def test_split_region_short_unchanged():
    assert _split_region(2.0, 5.0, 28.0) == [(2.0, 5.0)]


def test_split_region_none_or_zero_cap_is_noop():
    assert _split_region(0.0, 100.0, None) == [(0.0, 100.0)]
    assert _split_region(0.0, 100.0, 0) == [(0.0, 100.0)]


def test_split_region_long_is_equal_and_contiguous():
    pieces = _split_region(0.0, 45.0, 28.0)
    # 45 / 28 -> ceil = 2 equal pieces of 22.5s, each under the cap
    assert len(pieces) == 2
    assert all((b - a) <= 28.0 + 1e-9 for a, b in pieces)
    # contiguous, no gaps or overlaps, exact coverage of [0, 45]
    assert pieces[0][0] == 0.0
    assert pieces[-1][1] == 45.0
    for (_, prev_end), (next_start, _) in zip(pieces, pieces[1:]):
        assert abs(prev_end - next_start) < 1e-9


def test_split_region_exact_multiple():
    pieces = _split_region(0.0, 56.0, 28.0)
    assert len(pieces) == 2
    assert pieces == [(0.0, 28.0), (28.0, 56.0)]


def test_regions_to_segments_schema_and_label():
    segs = _regions_to_segments([(0.0, 3.0), (10.0, 12.5)], max_segment_s=28.0)
    assert len(segs) == 2
    for s in segs:
        assert set(s) == {"start", "end", "speaker"}
        assert s["speaker"] == "SPEECH"
    assert segs[0]["start"] == 0.0 and segs[0]["end"] == 3.0


def test_regions_to_segments_splits_long_region():
    # one 60s region must split (60/28 -> 3 pieces) so filter doesn't drop it as bad_length
    segs = _regions_to_segments([(0.0, 60.0)], max_segment_s=28.0)
    assert len(segs) == 3
    assert all((s["end"] - s["start"]) <= 28.0 + 1e-9 for s in segs)
    # full coverage preserved
    assert segs[0]["start"] == 0.0
    assert segs[-1]["end"] == 60.0


def test_regions_to_segments_drops_zero_width():
    segs = _regions_to_segments([(5.0, 5.0), (6.0, 7.0)], max_segment_s=28.0)
    assert len(segs) == 1
    assert segs[0]["start"] == 6.0
