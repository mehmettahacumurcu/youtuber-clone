from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from eval.card_eval_prompt import (
    BASE_RULES,
    STYLE_ADDENDUM,
    PromptConstructionError,
    build_messages,
)
from eval.card_eval_schema import CardSnapshot, load_suite


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_PATH = REPO_ROOT / "eval" / "fixtures" / "verified_card_eval_v1.json"


@pytest.fixture(scope="module")
def suite():
    return load_suite(SUITE_PATH)


@pytest.fixture(scope="module")
def g01(suite):
    return next(case for case in suite.cases if case.id == "G01")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_prompt_rules_are_byte_frozen_and_real_utf8():
    assert _sha256(BASE_RULES) == (
        "978e1a028c86ad7bd9f5532b42cafca56e3978373eb337cf307f5b36bc12c189"
    )
    assert _sha256(STYLE_ADDENDUM) == (
        "1719c745c410c13d4e5b2d396565d1b8dfd30d6e12bb362d9a9b4297b8e1e49b"
    )
    assert _sha256(BASE_RULES + STYLE_ADDENDUM) == (
        "78138a46cf41b5f845144a139806bc76318ea24fe235912579a3a8f474ca8383"
    )
    assert "güya" in BASE_RULES
    assert "Üslup örnekleri yalnızca ton içindir, bilgi kaynağı değildir" in (
        STYLE_ADDENDUM
    )
    assert "Ã" not in BASE_RULES + STYLE_ADDENDUM


def test_fixed_and_reordered_cards_build_identical_messages(g01):
    fixed = build_messages(g01, g01.cards, "semantic_plus_style")
    reordered = build_messages(
        g01, list(reversed(g01.cards)), "semantic_plus_style"
    )

    assert fixed == reordered
    assert fixed == [
        {"role": "system", "content": BASE_RULES + STYLE_ADDENDUM},
        {"role": "user", "content": fixed[1]["content"]},
    ]
    assert fixed[1]["content"].startswith("KANIT KARTLARI\n\n")
    assert fixed[1]["content"].endswith(f"SORU\n{g01.question}")
    assert "support=" not in fixed[1]["content"]
    assert "Üslup örnekleri" not in fixed[1]["content"]

    positions = [fixed[1]["content"].index(card_id) for card_id in g01.card_ids]
    assert positions == sorted(positions)
    assert "video=sample00005 | t=2391.4s" in fixed[1]["content"]


def test_semantic_arm_has_no_style_addendum(g01):
    messages = build_messages(g01, g01.cards, "semantic_only")
    assert messages[0] == {"role": "system", "content": BASE_RULES}
    assert "amına koyayım" not in messages[0]["content"]


def test_required_cards_precede_live_extras_without_mutating_provider_order(
    suite, g01
):
    extra = next(case.cards[0] for case in suite.cases if case.id == "F02")
    provider = [extra, *reversed(g01.cards)]
    before = tuple(card.id for card in provider)

    user = build_messages(g01, provider, "semantic_only")[1]["content"]

    assert tuple(card.id for card in provider) == before
    positions = [user.index(card_id) for card_id in (*g01.card_ids, extra.id)]
    assert positions == sorted(positions)


@pytest.mark.parametrize("arm", ["semantic_plus_speaker_style", "", "SEMANTIC_ONLY"])
def test_unknown_arm_is_rejected(g01, arm):
    with pytest.raises(PromptConstructionError, match="arm"):
        build_messages(g01, g01.cards, arm)


def test_duplicate_missing_and_sourceless_cards_are_rejected(g01):
    with pytest.raises(PromptConstructionError, match="duplicate"):
        build_messages(g01, [*g01.cards, g01.cards[0]], "semantic_only")

    with pytest.raises(PromptConstructionError, match="missing"):
        build_messages(g01, g01.cards[1:], "semantic_only")

    broken: CardSnapshot = replace(g01.cards[0], sources=())
    with pytest.raises(PromptConstructionError, match="source"):
        build_messages(
            g01, [broken, *g01.cards[1:]], "semantic_only"
        )


def test_required_card_id_cannot_mask_a_changed_snapshot(g01):
    changed = replace(g01.cards[0], take=g01.cards[0].take + " Uydurma ek.")
    with pytest.raises(PromptConstructionError, match="snapshot"):
        build_messages(
            g01, [changed, *g01.cards[1:]], "semantic_only"
        )
