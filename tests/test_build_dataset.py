"""Tests for the dataset formatter (clean_text, split_videos, check_outputs).

Uses the REAL config.yaml scrub_patterns (not a hardcoded subset) so a pattern change in
config is exercised here — that is how the subtitle-credit sibling-variant leak slipped past
the original 2-pattern test fixture.
"""
import re

import pytest

import json

from pipeline.config import load_settings
from training.build_dataset import PILOT_EXCLUDE, build, check_outputs, clean_text, split_videos

# Full-corpus (495-video) deterministic val holdout: seed=7, val_frac=0.1, AFTER dropping the
# PILOT_EXCLUDE drop-list from both splits. Refreshed 2026-06-23 when the corpus grew 8x from the
# ~60-video pilot (the old 6-video anchor was pilot-era). 47 videos.
_ASSESSED_VAL = {
    "-0cMi16Uoxg", "-OzO1RK0YcI", "082eWdfpCLo", "0TOSeq5aj2U", "0htCsgAHACM", "114xvlYQVC0",
    "56oOaeBs8Tk", "sample00007", "Ar2WOHRM0ds", "B0aNrY_pEhY", "C_djrDjYkSE", "DOC9VJ9agaA",
    "FW1YupykF6A", "sample00013", "HU_BG8RgJoQ", "sample00014", "sample00016", "sample00017",
    "K_eeoADiDZA", "LDpDuQOan8w", "Ls4mP0XEu0w", "sample00021", "RciPDw-DfuA", "sample00029",
    "X0iS8xhbt8M", "sample00033", "sample00034", "ZnM6x89COuU", "aeFz0VpNE8I", "bN6GPNRPqJY",
    "ccQzYx7yOD8", "d8clssZyjEY", "eYZNEZ_Tbb4", "iewbgImBBAI", "kR9B_ZMMUNQ", "sample00049",
    "lVFoBu07dLg", "sample00051", "mqKVpsw4KbM", "n8eb_aOMt1A", "scusg3SPkxg", "t1Bh-Xh-Jvg",
    "uAE2JmBrgYo", "wfwIlivq9cg", "xCrFKyMC-3A", "yVKwHkYu4-Y", "yjjuEIp6ANw",
}

# The actual production patterns, including the broadened subtitle-credit stem.
SCRUB = [re.compile(p, re.IGNORECASE) for p in load_settings(None).filter.scrub_patterns]


def test_config_ships_three_scrub_patterns():
    # guard against silent config drift
    assert len(SCRUB) == 3


def test_skip_becomes_paragraph_break():
    out = clean_text("birinci cumle\n[SKIP]\nikinci cumle", [])
    assert "[SKIP]" not in out
    assert out == "birinci cumle\n\nikinci cumle"


def test_multiple_skips_collapse_to_single_breaks():
    out = clean_text("a\n[SKIP]\n[SKIP]\nb", [])
    assert out == "a\n\nb"
    assert "\n\n\n" not in out


def test_scrub_applied_in_place_keeps_surrounding_speech():
    out = clean_text("merhaba Altyazı M.K. dünya", SCRUB)
    assert "Altyaz" not in out
    assert "merhaba" in out and "dünya" in out


def test_subtitle_credit_tail_variant_scrubbed():
    out = clean_text("söz Altyazı ekleyen ve yorumladığı için teşekkürler. devam", SCRUB)
    assert "Altyaz" not in out
    assert "söz" in out and "devam" in out


def test_subtitle_credit_cizgi_film_variant_scrubbed():
    # the sibling variant that leaked into train.jsonl (FCDy5sobdgM), fused mid-paragraph
    out = clean_text("Pentagon'daki pizzacılar gibi. Altyazı ekleyen ve çizgi film müziği", SCRUB)
    assert "Altyaz" not in out
    assert "pizzacılar gibi." in out


def test_scrub_preserves_paragraph_structure():
    out = clean_text("ilk paragraf\n[SKIP]\nAltyazı M.K.\n[SKIP]\nson paragraf", SCRUB)
    parts = out.split("\n\n")
    assert parts[0] == "ilk paragraf"
    assert parts[-1] == "son paragraf"


def test_real_speech_with_tesekkur_survives():
    # broad keyword traps: he really says these — must NOT be scrubbed
    txt = "çok teşekkür ederim canlar, müzik açalım"
    assert clean_text(txt, SCRUB) == txt


def test_empty_after_clean():
    assert clean_text("[SKIP]\n[SKIP]", []) == ""


def test_nfc_normalization():
    # decomposed 'ç' (c + combining cedilla) -> precomposed U+00E7
    decomposed = "soru çok"
    out = clean_text(decomposed, [])
    assert "ç" in out and "̧" not in out


def test_lone_internal_newline_becomes_space():
    # defensive: a stray single newline inside a segment must not survive as a hard break
    assert clean_text("line one\nline two", []) == "line one line two"
    # but real paragraph breaks are preserved
    assert clean_text("a\n[SKIP]\nb", []) == "a\n\nb"


def test_whitespace_normalization_is_unconditional():
    # internal multi-space collapses whether or not scrub patterns are configured
    assert clean_text("ali   veli", []) == "ali veli"
    assert clean_text("ali   veli", SCRUB) == "ali veli"


def test_split_holds_out_whole_videos_no_overlap():
    ids = [f"v{i}" for i in range(56)]
    train, val = split_videos(ids, val_frac=0.1, seed=7)
    assert not (train & val)
    assert train | val == set(ids)
    assert val  # ~10% of 56 ≈ 6


def test_split_is_deterministic():
    ids = [f"v{i}" for i in range(56)]
    assert split_videos(ids, 0.1, 7) == split_videos(ids, 0.1, 7)
    # different seed -> (very likely) different holdout
    assert split_videos(ids, 0.1, 7)[1] != split_videos(ids, 0.1, 99)[1]


def test_split_val_frac_zero_gives_empty_val():
    ids = [f"v{i}" for i in range(56)]
    train, val = split_videos(ids, 0.0, 7)
    assert val == set()
    assert train == set(ids)


def test_split_rejects_out_of_range_val_frac():
    ids = [f"v{i}" for i in range(10)]
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            split_videos(ids, bad, 7)


def test_split_dedupes_so_dup_id_cannot_straddle():
    # a duplicate video_id must never land in both splits
    train, val = split_videos(["a", "a", "b", "c"], 0.5, 1)
    assert not (train & val)


def test_check_outputs_passes_clean_split():
    train = [{"video_id": "a", "text": "temiz prose"}]
    val = [{"video_id": "b", "text": "başka prose"}]
    assert check_outputs(train, val) == {}


def test_check_outputs_raises_on_skip_leak():
    train = [{"video_id": "a", "text": "kötü [SKIP] metin"}]
    with pytest.raises(AssertionError):
        check_outputs(train, [])


def test_check_outputs_raises_on_video_id_leak():
    train = [{"video_id": "a", "text": "x"}]
    val = [{"video_id": "a", "text": "y"}]
    with pytest.raises(AssertionError):
        check_outputs(train, val)


def test_check_outputs_warns_on_artifact_signature():
    train = [{"video_id": "a", "text": "Altyazı bir şey"}]
    warnings = check_outputs(train, [])
    assert warnings  # reported, not raised


def _corpus_line(vid, text):
    return json.dumps({"video_id": vid, "title": None, "text": text}, ensure_ascii=False)


def test_build_source_clean_reads_canonical_and_writes_train_val(settings):
    dd = settings.paths.dataset_dir
    cfg = settings.paths.data_dir.parent / "config.yaml"
    (dd / "llm_corpus.jsonl").write_text(
        "\n".join(_corpus_line(f"v{i}", f"cümle {i} duayen") for i in range(10)) + "\n",
        encoding="utf-8")
    stats = build(str(cfg), val_frac=0.1, seed=7, exclude_ids=set(), source="clean")
    assert (dd / "train.jsonl").exists() and (dd / "val.jsonl").exists()
    assert stats["train_path"].endswith("train.jsonl")
    train = (dd / "train.jsonl").read_text(encoding="utf-8")
    assert "duayen" in train  # canonical clean text preserved


def test_build_source_filter_reads_raw_and_writes_raw_names(settings):
    dd = settings.paths.dataset_dir
    cfg = settings.paths.data_dir.parent / "config.yaml"
    (dd / "llm_corpus.raw.jsonl").write_text(
        "\n".join(_corpus_line(f"v{i}", f"ham {i} duaylı") for i in range(10)) + "\n",
        encoding="utf-8")
    stats = build(str(cfg), val_frac=0.1, seed=7, exclude_ids=set(), source="filter")
    assert (dd / "train.raw.jsonl").exists() and (dd / "val.raw.jsonl").exists()
    assert not (dd / "train.jsonl").exists()  # raw build must not clobber the canonical
    assert stats["train_path"].endswith("train.raw.jsonl")
    assert "duaylı" in (dd / "train.raw.jsonl").read_text(encoding="utf-8")  # raw uncorrected


def test_pilot_exclude_list_present():
    # the 2026-06-18 content-review drop-list (not-him / mixed / legal-hazard docs)
    assert {"9AejUUYv35k", "sample00038", "9CDdXsqqNME", "AUr4uLeBi1k"} <= PILOT_EXCLUDE


def test_produced_dataset_honors_exclude_and_val_anchor():
    # integration guard against the actual committed train/val: no excluded doc in train,
    # and the human-assessed val anchor is preserved exactly.
    dd = load_settings(None).paths.dataset_dir
    tp, vp = dd / "train.jsonl", dd / "val.jsonl"
    if not (tp.exists() and vp.exists()):
        import pytest
        pytest.skip("dataset not built")
    train_ids = {json.loads(l)["video_id"] for l in tp.read_text(encoding="utf-8").splitlines() if l.strip()}
    val_ids = {json.loads(l)["video_id"] for l in vp.read_text(encoding="utf-8").splitlines() if l.strip()}
    assert not (PILOT_EXCLUDE & train_ids), "an excluded doc leaked into train.jsonl"
    assert not (PILOT_EXCLUDE & val_ids)
    assert val_ids == _ASSESSED_VAL, "val anchor drifted from the assessed holdout"
