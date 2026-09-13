"""Inference path for the fine-tuned voice model with verified RAG as an explicit mode."""
from __future__ import annotations

import argparse

from pipeline.config import load_settings
from rag.ollama_residency import ensure_absent
from rag.prompt import SYS
from runtime.ollama_http import ollama_json
from runtime.settings import RuntimeSettings

_DEFAULT_MODEL = "speaker-qwen3-ep2"

__all__ = ["SYS", "answer"]


def _generate(messages: list[dict], model: str, r) -> str:
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": r.gen_temperature,
            "top_p": r.gen_top_p,
            "repeat_penalty": r.gen_repeat_penalty,
            "num_predict": 512,
        },
    }
    url = RuntimeSettings.from_environment().ollama_origin + "/api/chat"
    payload = ollama_json("POST", url, payload=body, timeout_s=300.0)
    return payload["message"]["content"].strip()


def answer(
    question: str,
    model: str = _DEFAULT_MODEL,
    use_rag: bool = False,
    config: str | None = None,
    *,
    assess_fn=None,
    verifier_factory=None,
    generate_fn=None,
    ensure_absent_fn=None,
) -> str:
    from rag.evidence_pipeline import assess_question
    from rag.prompt import REFUSAL_TR, build_grounded_user, format_evidence_citations
    from rag.verifier import EvidenceVerifier

    settings = load_settings(config)
    rag = settings.rag
    active_assess = assess_fn or assess_question
    active_verifier_factory = verifier_factory or EvidenceVerifier.from_config
    active_generate = generate_fn or _generate
    active_ensure_absent = ensure_absent_fn or ensure_absent
    if not use_rag:
        messages = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": question.strip()},
        ]
        return active_generate(messages, model, rag)

    try:
        active_ensure_absent(model)
        decision = active_assess(question, settings, verifier=active_verifier_factory(rag))
    except Exception as exc:
        return f"RAG kan\u0131t denetimi i\u00e7in model ge\u00e7i\u015fi ba\u015far\u0131s\u0131z oldu. Hata: {exc}."
    if decision.status == "unsupported":
        return REFUSAL_TR
    if decision.status == "error":
        return (
            "RAG kan\u0131t denetimi ba\u015far\u0131s\u0131z oldu\u011fu i\u00e7in cevap \u00fcretmedim. "
            f"Hata: {decision.failure_reason}."
        )
    user = build_grounded_user(
        decision.answer_question,
        [span.text for span in decision.selected_spans],
        max_spans=5,
        span_char_cap=1800,
    )
    messages = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": user},
    ]
    try:
        active_ensure_absent(rag.verifier_model)
    except Exception as exc:
        return f"RAG kan\u0131t denetimi i\u00e7in model ge\u00e7i\u015fi ba\u015far\u0131s\u0131z oldu. Hata: {exc}."
    generated = active_generate(messages, model, rag)
    notice = ""
    if decision.status == "partial":
        notice = "Kan\u0131t\u0131 bulunamayan k\u0131s\u0131m: " + "; ".join(decision.unsupported_claims) + "\n\n"
    citations = format_evidence_citations(decision.selected_spans)
    return notice + generated + (f"\n\nKaynaklar:\n{citations}" if citations else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the fine-tuned Speaker voice model.")
    parser.add_argument("question")
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument(
        "--rag",
        action="store_true",
        help=(
            "use verified direct local-Qdrant RAG; stop the persistent retrieval server first "
            "because both processes cannot own the local store simultaneously"
        ),
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    print(answer(args.question, model=args.model, use_rag=args.rag, config=args.config))


if __name__ == "__main__":
    main()
