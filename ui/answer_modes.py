"""Stable answer-policy values shared by Studio and chat runtime."""
from __future__ import annotations

from typing import Literal, cast


AnswerMode = Literal["grounded_strict", "grounded_fallback", "free"]
ANSWER_MODE_LABELS: dict[str, AnswerMode] = {
    "Grounded only": "grounded_strict",
    "Grounded + answer anyway": "grounded_fallback",
    "Free mode": "free",
}
DEFAULT_ANSWER_MODE_LABEL = "Grounded only"
_ANSWER_MODES = frozenset(ANSWER_MODE_LABELS.values())


def answer_mode_from_label(label: str) -> AnswerMode:
    try:
        return ANSWER_MODE_LABELS[label]
    except KeyError as exc:
        raise ValueError(f"unknown answer mode {label!r}") from exc


def validate_answer_mode(mode: object) -> AnswerMode:
    if type(mode) is not str or mode not in _ANSWER_MODES:
        raise ValueError(f"unknown answer mode {mode!r}")
    return cast(AnswerMode, mode)


def uses_retrieval(mode: AnswerMode) -> bool:
    return validate_answer_mode(mode) != "free"
