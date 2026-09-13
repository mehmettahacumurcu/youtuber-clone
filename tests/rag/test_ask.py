import rag.ask as ask


def test_default_cli_delegates_to_verified_rag(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ask,
        "answer_finetuned",
        lambda *args, **kwargs: calls.append(kwargs) or "verified",
    )

    assert ask.answer("question", ungrounded=False) == "verified"
    assert calls == [{"model": "speaker-v5-a636", "use_rag": True, "config": None, "ensure_absent_fn": None}]


def test_ungrounded_flag_disables_rag_before_generation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ask,
        "answer_finetuned",
        lambda *args, **kwargs: calls.append(kwargs) or "free",
    )

    assert ask.answer("question", ungrounded=True) == "free"
    assert calls == [{"model": "speaker-v5-a636", "use_rag": False, "config": None, "ensure_absent_fn": None}]


def test_canonical_cli_forwards_rag_mode_and_residency_hook(monkeypatch):
    calls = []
    hook = object()

    def fine_tuned(question, *, model, use_rag, config, ensure_absent_fn):
        calls.append((question, model, use_rag, config, ensure_absent_fn))
        return "answer"

    monkeypatch.setattr(ask, "answer_finetuned", fine_tuned)

    assert ask.answer("question", ungrounded=False, ensure_absent_fn=hook) == "answer"
    assert calls == [("question", "speaker-v5-a636", True, None, hook)]
