import pytest

from ui.answer_modes import (
    ANSWER_MODE_LABELS,
    DEFAULT_ANSWER_MODE_LABEL,
    answer_mode_from_label,
    uses_retrieval,
    validate_answer_mode,
)


def test_answer_mode_labels_are_ordered_and_strict_is_default():
    assert list(ANSWER_MODE_LABELS.items()) == [
        ("Grounded only", "grounded_strict"),
        ("Grounded + answer anyway", "grounded_fallback"),
        ("Free mode", "free"),
    ]
    assert DEFAULT_ANSWER_MODE_LABEL == "Grounded only"


@pytest.mark.parametrize("mode", ["grounded_strict", "grounded_fallback"])
def test_grounded_modes_use_retrieval(mode):
    assert uses_retrieval(mode) is True


def test_free_mode_disables_retrieval_and_unknown_labels_fail():
    assert uses_retrieval("free") is False
    with pytest.raises(ValueError, match="unknown answer mode"):
        answer_mode_from_label("Anything")


@pytest.mark.parametrize("mode", [True, False, "anything"])
def test_invalid_answer_mode_values_fail(mode):
    with pytest.raises(ValueError, match="unknown answer mode"):
        validate_answer_mode(mode)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Grounded only", True),
        ("Grounded + answer anyway", True),
        ("Free mode", False),
    ],
)
def test_retrieval_control_interactivity_follows_answer_mode(label, expected):
    assert uses_retrieval(answer_mode_from_label(label)) is expected
