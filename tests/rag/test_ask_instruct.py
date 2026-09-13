from unittest.mock import Mock

from rag.ask_instruct import answer
from rag.evidence_pipeline import EvidenceDecision


def _decision():
    return EvidenceDecision("answerable", (), (), (), "safe question", (), None, {})


def test_instruct_unsupported_never_generates():
    generate = Mock(return_value="must not run")
    decision = EvidenceDecision("unsupported", (), (), (), None, (), None, {})
    result = answer(
        "question",
        assess_fn=lambda *args, **kwargs: decision,
        verifier_factory=lambda config: object(),
        generate_fn=generate,
        ensure_absent_fn=lambda model: None,
    )
    generate.assert_not_called()
    assert result


def test_rag_on_hands_off_answer_model_then_verifier_before_generation():
    events = []

    result = answer(
        "question", model="speaker-custom",
        assess_fn=lambda *args, **kwargs: events.append("verify") or _decision(),
        verifier_factory=lambda config: object(),
        generate_fn=lambda *args: events.append("generate") or "answer",
        ensure_absent_fn=lambda model: events.append(("unload", model)),
    )

    assert result == "answer"
    assert events == [
        ("unload", "speaker-custom"),
        "verify",
        ("unload", "qwen3:4b"),
        "generate",
    ]


def test_rag_on_residency_error_never_generates():
    events = []
    generate = Mock(return_value="must not run")

    def ensure_absent(model):
        events.append(("unload", model))
        if model == "qwen3:4b":
            raise RuntimeError("verifier residency failed")

    result = answer(
        "question", model="speaker-custom",
        assess_fn=lambda *args, **kwargs: events.append("verify") or _decision(),
        verifier_factory=lambda config: object(), generate_fn=generate,
        ensure_absent_fn=ensure_absent,
    )

    generate.assert_not_called()
    assert "verifier residency failed" in result
    assert events == [("unload", "speaker-custom"), "verify", ("unload", "qwen3:4b")]


def test_ungrounded_does_not_unload_before_direct_generation():
    generate = Mock(return_value="free")

    result = answer(
        "free question", ungrounded=True, generate_fn=generate,
        ensure_absent_fn=lambda model: (_ for _ in ()).throw(AssertionError("must not unload")),
    )

    assert result == "free"
