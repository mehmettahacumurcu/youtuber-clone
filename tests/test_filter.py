import json
import re

from pipeline.filter.runner import classify_segment, discover_video_ids, run_filter


def _seg(text="merhaba dunya", duration_s=5.0, avg_logprob=-0.3):
    return {"start": 0.0, "end": duration_s, "duration_s": duration_s,
            "text": text, "avg_logprob": avg_logprob}


def _write_transcribe(settings, vid, segments):
    payload = {"video_id": vid, "segments": segments}
    (settings.paths.transcribe_dir / f"{vid}.json").write_text(
        json.dumps(payload), encoding="utf-8")


def test_classify_each_gate(settings):
    cfg = settings.filter
    patterns = [__import__("re").compile(p) for p in cfg.drop_patterns]
    assert classify_segment(_seg(), cfg, patterns) is None
    assert classify_segment(_seg(text="  "), cfg, patterns) == "empty"
    assert classify_segment(_seg(avg_logprob=-2.0), cfg, patterns) == "low_confidence"
    assert classify_segment(_seg(duration_s=0.5), cfg, patterns) == "bad_length"
    assert classify_segment(_seg(duration_s=40.0), cfg, patterns) == "bad_length"
    # 60 words in 5 s = 12 wps > 6.0
    fast = _seg(text=" ".join(["kelime"] * 60))
    assert classify_segment(fast, cfg, patterns) == "word_rate"


def test_profanity_is_kept(settings):
    # Hard project constraint: profanity must never be filtered. Prove that the REAL
    # production drop patterns (from config.yaml) do not match profane-but-clean text.
    real_patterns = [re.compile(p) for p in (
        r"(?i)thank(s)? for watching",
        r"(?i)subscribe to",
        r"(?i)like and subscribe",
        r"(?i)\bcopyright\b.{0,40}\b(youtube|all rights)\b",
    )]
    cfg = settings.filter
    assert classify_segment(_seg(text="bu tam bir bok"), cfg, real_patterns) is None
    assert classify_segment(_seg(text="siktir git"), cfg, real_patterns) is None


def test_classify_flags_hallucination_pattern(settings):
    cfg = settings.filter
    patterns = [re.compile(r"(?i)thank(s)? for watching")]
    seg = _seg(text="thanks for watching this video")
    assert classify_segment(seg, cfg, patterns) == "hallucination_pattern"


def test_run_filter_writes_kept_and_counts(settings):
    vid = "vidf"
    _write_transcribe(settings, vid, [
        _seg(text="iyi bir cumle"),                 # keep
        _seg(text="", duration_s=4.0),              # empty
        _seg(avg_logprob=-3.0),                     # low_confidence
        _seg(duration_s=0.4),                       # bad_length
    ])
    out = run_filter(vid, settings)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["input_segments"] == 4
    assert doc["kept_segments"] == 1
    assert doc["dropped_by_reason"] == {"empty": 1, "low_confidence": 1, "bad_length": 1}
    assert doc["segments"][0]["text"] == "iyi bir cumle"


def test_run_filter_idempotent_skip(settings):
    vid = "vidf2"
    _write_transcribe(settings, vid, [_seg(text="bir")])
    out = run_filter(vid, settings)
    out.write_text('{"sentinel": true}', encoding="utf-8")  # mutate
    run_filter(vid, settings)  # should skip, not overwrite
    assert json.loads(out.read_text(encoding="utf-8")) == {"sentinel": True}


def test_discover_video_ids(settings):
    _write_transcribe(settings, "b", [_seg()])
    _write_transcribe(settings, "a", [_seg()])
    assert discover_video_ids(settings) == ["a", "b"]
