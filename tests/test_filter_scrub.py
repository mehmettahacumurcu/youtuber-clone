"""Tests for the in-place ASR-artifact scrub (filter Stage 6).

These pin the PRECISION guarantee from the Stage-7 artifact review: the validated patterns
excise subtitle credits / thanks-for-watching while leaving his real speech (including the
real words "teşekkür" and "müzik") completely untouched.
"""
import re

from pipeline.config import load_settings
from pipeline.filter.runner import scrub_text

# The shipped patterns (kept in sync with config.yaml filter.scrub_patterns).
PATTERNS = [
    r"Altyaz[ıi]\s+M\.?\s*K\.?",
    r"Altyazı ekleyen[^.\n]*için teşekkürler\.?",
    r"^\s*İzledi[ğg]iniz i[çc]in te[şs]ekk[üu]r ederim\.?\s*$",
]


def _scrub():
    return [re.compile(p, re.IGNORECASE) for p in PATTERNS]


def test_altyazi_mk_embedded_strips_and_keeps_speech():
    # 6 of 8 real occurrences are spliced mid-sentence — must keep the surrounding words
    out = scrub_text("ültimatom koyduğu zaman Altyazı M.K. sık somut eylemlerle", _scrub())
    assert out == "ültimatom koyduğu zaman sık somut eylemlerle"


def test_altyazi_mk_standalone_becomes_empty():
    assert scrub_text("Altyazı M.K.", _scrub()) == ""


def test_translator_credit_becomes_empty():
    assert scrub_text("Altyazı ekleyen ve yorumladığı için teşekkürler.", _scrub()) == ""


def test_thanks_for_watching_standalone_becomes_empty():
    assert scrub_text("İzlediğiniz için teşekkür ederim.", _scrub()) == ""


def test_thanks_for_watching_with_trailing_real_speech_is_preserved():
    # full-line ^...$ anchor: must NOT fire when real speech follows in the same segment
    txt = "İzlediğiniz için teşekkür ederim. Yerlilerle anlaşma meselesinde böyle püf noktaları"
    assert scrub_text(txt, _scrub()) == txt


def test_real_thanks_to_a_guest_untouched():
    txt = "Doğan Şutucuğum teşekkür ederim canım."
    assert scrub_text(txt, _scrub()) == txt


def test_real_music_word_untouched():
    txt = "Bu işler dansla müzikle olmaz."
    assert scrub_text(txt, _scrub()) == txt


def test_no_scrub_patterns_is_noop():
    txt = "Altyazı M.K. burada kalmalı"
    assert scrub_text(txt, []) == txt


def test_config_ships_the_three_validated_patterns():
    cfg = load_settings().filter
    assert len(cfg.scrub_patterns) == 3
    assert any("Altyaz" in p and "K" in p for p in cfg.scrub_patterns)
    assert any("İzledi" in p for p in cfg.scrub_patterns)
