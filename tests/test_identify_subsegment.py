"""Unit tests for the sliding-window sub-segmentation logic in Stage 4 (identify).

Only the pure window/bridge/coalesce/decision math is tested here — no pyannote, no ECAPA,
no GPU. The embedding values are fed as synthetic similarity arrays, so the keep/drop/bridge
decisions are deterministic and regressions are caught locally.
"""
from pipeline.identify.runner import (
    _bridge_dips,
    _coalesce_keep_windows,
    _decision_for,
    _window_grid,
)


class _Cfg:
    similarity_threshold = 0.50
    borderline_low = 0.40


# --- _window_grid -----------------------------------------------------------------------

def test_window_grid_short_returns_single():
    assert _window_grid(2.0, 3.0, 1.5, 0.75) == [(2.0, 3.0)]


def test_window_grid_full_length_and_covers_end():
    g = _window_grid(0.0, 5.0, 1.5, 0.75)
    assert all(abs((b - a) - 1.5) < 1e-6 for a, b in g)   # every window exactly WIN long
    assert g[0] == (0.0, 1.5)
    assert g[1] == (0.75, 2.25)
    assert abs(g[-1][1] - 5.0) < 1e-6                      # last window ends exactly at end


def test_window_grid_no_exact_duplicate_windows():
    for end in (2.25, 3.0, 4.0, 5.5, 6.01):
        g = _window_grid(0.0, end, 1.5, 0.75)
        assert len(g) == len(set(g)), f"duplicate window for end={end}"
        assert abs(g[-1][1] - end) < 1e-6
        assert all(abs((b - a) - 1.5) < 1e-6 for a, b in g)


# --- _bridge_dips -----------------------------------------------------------------------

def test_bridge_single_dip_is_closed():
    assert _bridge_dips([True, False, True], 1) == [True, True, True]


def test_bridge_does_not_join_two_dips():
    assert _bridge_dips([True, False, False, True], 1) == [True, False, False, True]


def test_bridge_ignores_edge_dips():
    assert _bridge_dips([False, True, True], 1) == [False, True, True]
    assert _bridge_dips([True, True, False], 1) == [True, True, False]


def test_bridge_disabled_when_max_dip_zero():
    assert _bridge_dips([True, False, True], 0) == [True, False, True]


# --- _coalesce_keep_windows -------------------------------------------------------------

_GRID7 = [(0.0, 1.5), (0.75, 2.25), (1.5, 3.0), (2.25, 3.75), (3.0, 4.5), (3.75, 5.25), (4.5, 6.0)]


def test_coalesce_excises_guest_run():
    # middle 3-window run is the guest (sims ~0.3) -> not bridged -> excised; two him spans
    sims = [0.61, 0.62, 0.30, 0.29, 0.28, 0.60, 0.61]
    keep = _bridge_dips([s >= 0.50 for s in sims], 1)
    spans = _coalesce_keep_windows(_GRID7, keep, 2.0)
    assert [idxs for _, _, idxs in spans] == [[0, 1], [5, 6]]


def test_coalesce_drops_span_below_min():
    # only window 0 keeps -> 1.5s span < 2.0 min -> dropped
    assert _coalesce_keep_windows(_GRID7, [True, False, False, False, False, False, False], 2.0) == []


def test_coalesce_two_window_span_survives_min():
    spans = _coalesce_keep_windows(_GRID7, [True, True, False, False, False, False, False], 2.0)
    assert len(spans) == 1
    start, end, idxs = spans[0]
    assert start == 0.0 and abs(end - 2.25) < 1e-6 and idxs == [0, 1]


def test_coalesce_bridged_dip_yields_one_span():
    # a single guest-ish dip between him windows bridges into one continuous span
    keep = _bridge_dips([0.6 >= 0.5, 0.3 >= 0.5, 0.6 >= 0.5], 1)  # [T,F,T] -> [T,T,T]
    spans = _coalesce_keep_windows(_GRID7[:3], keep, 2.0)
    assert len(spans) == 1 and spans[0][2] == [0, 1, 2]


# --- _decision_for ----------------------------------------------------------------------

def test_decision_ladder():
    c = _Cfg()
    assert _decision_for(0.60, c) == "keep"
    assert _decision_for(0.45, c) == "borderline"
    assert _decision_for(0.20, c) == "drop"
