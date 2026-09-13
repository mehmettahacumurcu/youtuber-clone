from unittest.mock import Mock

from rag.ask_finetuned import answer
from rag.evidence_pipeline import EvidenceDecision
from rag.types import EvidenceSpan


def _decision(status):
    span = EvidenceSpan(
        "span::v::1000::2000", "v", "T", 1.0, 2.0, "clean evidence", 2.0,
        ("v::0",), ("original",),
    )
    return EvidenceDecision(
        status=status, claims=(), selected_spans=(span,) if status in {"answerable", "partial"} else (),
        hint_hits=(), answer_question="safe claim" if status in {"answerable", "partial"} else None,
        unsupported_claims=("fabricated claim",) if status == "partial" else (),
        failure_reason="verifier_invalid_output" if status == "error" else None,
        diagnostics={},
    )


def test_rag_answerable_uses_verified_question_and_span():
    generate = Mock(return_value="answer")
    result = answer(
        "original", model="m", use_rag=True,
        assess_fn=lambda *args, **kwargs: _decision("answerable"),
        verifier_factory=lambda config: object(), generate_fn=generate,
        ensure_absent_fn=lambda model: None,
    )
    messages = generate.call_args.args[0]
    assert "safe claim" in messages[1]["content"]
    assert "clean evidence" in messages[1]["content"]
    assert "watch?v=v&t=1s" in result


def test_partial_excludes_fabricated_claim_from_model_but_displays_notice():
    generate = Mock(return_value="answer")
    result = answer(
        "original fabricated claim", model="m", use_rag=True,
        assess_fn=lambda *args, **kwargs: _decision("partial"),
        verifier_factory=lambda config: object(), generate_fn=generate,
        ensure_absent_fn=lambda model: None,
    )
    serialized = str(generate.call_args.args[0])
    assert "safe claim" in serialized
    assert "fabricated claim" not in serialized
    assert "fabricated claim" in result


def test_unsupported_and_error_never_generate():
    for status in ("unsupported", "error"):
        generate = Mock(return_value="must not run")
        result = answer(
            "question", model="m", use_rag=True,
            assess_fn=lambda *args, status=status, **kwargs: _decision(status),
            verifier_factory=lambda config: object(), generate_fn=generate,
            ensure_absent_fn=lambda model: None,
        )
        generate.assert_not_called()
        assert result


def test_rag_off_generates_bare_question_without_assessment():
    generate = Mock(return_value="free")
    result = answer(
        "bare question", model="m", use_rag=False,
        assess_fn=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no assessment")),
        verifier_factory=lambda config: object(), generate_fn=generate,
    )
    assert result == "free"
    assert generate.call_args.args[0][1]["content"] == "bare question"


def test_rag_on_hands_off_answer_model_then_verifier_before_generation():
    events = []

    result = answer(
        "question", model="speaker-custom", use_rag=True,
        assess_fn=lambda *args, **kwargs: events.append("verify") or _decision("answerable"),
        verifier_factory=lambda config: object(),
        generate_fn=lambda *args: events.append("generate") or "answer",
        ensure_absent_fn=lambda model: events.append(("unload", model)),
    )

    assert result.startswith("answer")
    assert events == [
        ("unload", "speaker-custom"),
        "verify",
        ("unload", "qwen3:4b"),
        "generate",
    ]


def test_rag_on_residency_error_never_generates():
    for failing_model, expected_events in (
        ("speaker-custom", [("unload", "speaker-custom")]),
        ("qwen3:4b", [("unload", "speaker-custom"), "verify", ("unload", "qwen3:4b")]),
    ):
        events = []
        generate = Mock(return_value="must not run")

        def ensure_absent(model):
            events.append(("unload", model))
            if model == failing_model:
                raise RuntimeError("residency failed")

        result = answer(
            "question", model="speaker-custom", use_rag=True,
            assess_fn=lambda *args, **kwargs: events.append("verify") or _decision("answerable"),
            verifier_factory=lambda config: object(), generate_fn=generate,
            ensure_absent_fn=ensure_absent,
        )

        generate.assert_not_called()
        assert "residency failed" in result
        assert events == expected_events
