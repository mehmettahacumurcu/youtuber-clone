"""Shared, deterministic prompts for the verified-card evaluation harness."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence

from eval.card_eval_schema import CardSnapshot, EvalCase


BASE_RULES = """Sen kanıta bağlı çalışan, dikkatli bir Türkçe cevap üreticisisin.

Öncelik sırası: doğruluk, doğru atıf, açıklık. Yalnızca verilen kanıt kartlarındaki bilgiyi
kullan. Yeni kişi, olay, tarih, sayı veya neden ekleme. Kanıtta bir görüş başkasına "iddia",
"hikâye", "güya" veya "ona göre" diye atfediliyorsa bunu gerçekleşmiş bir olgu gibi yazma.
Negasyonu ve kimin ne söylediğini koru. Kanıtlar yetmiyorsa açıkça söyle. Soruyu doğrudan,
tutarlı ve 120-220 kelimelik Türkçe bir yanıtla cevapla."""

STYLE_ADDENDUM = """

Doğruluk kurallarını bozmadan yanıtı Speaker'in yayın ağzına yaklaştır: senli benli, doğrudan,
rahat ve gerektiği yerde en fazla bir-iki doğal küfür kullan. Önce tezi açıkça söyle, sonra
gerekçeyi kur. Parça parça konuşma, kelime salatası yapma ve kanıtta olmayan ayrıntı ekleme.

Üslup örnekleri yalnızca ton içindir, bilgi kaynağı değildir:
- "Bak olay şu kardeşim..."
- "O kadar kolay değil abi."
- "Bunu kimseye yediremezsin amına koyayım."
"""

ARM_IDS = ("semantic_only", "semantic_plus_style")
_MOJIBAKE_MARKERS = ("Ã", "Â", "Å", "Ä", "â€", "ðŸ", "ï»¿")


class PromptConstructionError(ValueError):
    """Raised when evidence cannot produce one canonical prompt."""


def build_messages(
    case: EvalCase,
    evidence_cards: Sequence[CardSnapshot],
    arm_id: str,
) -> list[dict[str, str]]:
    """Build the exact native-chat messages used by fixed and live providers.

    Required fixture cards are always ordered by ``case.card_ids``. A live provider may
    append other retrieved cards; those retain provider order after the required packet.
    Support scores are deliberately excluded because they are discovery metadata, not
    evidence of a proposition.
    """

    if not isinstance(case, EvalCase):
        raise TypeError("case must be an EvalCase")
    if arm_id not in ARM_IDS:
        raise PromptConstructionError(
            f"unknown prompt arm {arm_id!r}; expected one of {ARM_IDS}"
        )

    cards = tuple(evidence_cards)
    if not cards:
        raise PromptConstructionError("evidence cards must not be empty")
    if any(not isinstance(card, CardSnapshot) for card in cards):
        raise TypeError("every evidence card must be a CardSnapshot")

    by_id: dict[str, CardSnapshot] = {}
    provider_order: list[str] = []
    for card in cards:
        if card.id in by_id:
            raise PromptConstructionError(f"duplicate evidence card ID {card.id!r}")
        if not card.sources:
            raise PromptConstructionError(
                f"evidence card {card.id!r} has no canonical source"
            )
        source = card.sources[0]
        for label, value in (
            ("ID", card.id),
            ("question", card.q),
            ("take", card.take),
            ("source video", source.video_id),
            ("source title", source.title),
        ):
            _validate_prompt_text(value, f"card {card.id!r} {label}")
        if not source.video_id.strip() or not math.isfinite(source.start) or source.start < 0:
            raise PromptConstructionError(
                f"evidence card {card.id!r} has an invalid canonical source"
            )
        by_id[card.id] = card
        provider_order.append(card.id)

    missing = tuple(card_id for card_id in case.card_ids if card_id not in by_id)
    if missing:
        raise PromptConstructionError(
            f"missing required evidence cards: {', '.join(missing)}"
        )

    canonical_by_id = {card.id: card for card in case.cards}
    if tuple(canonical_by_id) != case.card_ids:
        raise PromptConstructionError(
            "case card snapshots do not match the declared required card IDs"
        )
    changed = tuple(
        card_id
        for card_id in case.card_ids
        if by_id[card_id] != canonical_by_id[card_id]
    )
    if changed:
        raise PromptConstructionError(
            "required card snapshot differs from the frozen fixture: "
            + ", ".join(changed)
        )

    required = set(case.card_ids)
    ordered_ids = (*case.card_ids, *(item for item in provider_order if item not in required))
    blocks = [_format_card(by_id[card_id]) for card_id in ordered_ids]
    user = "KANIT KARTLARI\n\n" + "\n\n".join(blocks)
    user += f"\n\nSORU\n{case.question}"

    system = BASE_RULES
    if arm_id == "semantic_plus_style":
        system += STYLE_ADDENDUM
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _format_card(card: CardSnapshot) -> str:
    source = card.sources[0]
    timestamp = format(source.start, ".15g")
    return (
        f"[{card.id} | video={source.video_id} | t={timestamp}s]\n"
        f"{card.take}"
    )


def _validate_prompt_text(value: str, label: str) -> None:
    if type(value) is not str or not value.strip():
        raise PromptConstructionError(f"{label} must be non-empty text")
    if (
        not unicodedata.is_normalized("NFC", value)
        or "\ufffd" in value
        or any(marker in value for marker in _MOJIBAKE_MARKERS)
        or any(0xD800 <= ord(character) <= 0xDFFF for character in value)
    ):
        raise PromptConstructionError(f"{label} contains corrupt Unicode")
