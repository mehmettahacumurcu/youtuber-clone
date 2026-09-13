"""EXPERIMENTAL instruct-generation path (spike).

Coherent, grounded answers in his voice via an UNCENSORED instruct model — few-shot his
real style + RAG grounding — instead of the base-completion voice model in `rag.ask`.
The instruct model supplies coherence; few-shot supplies the voice; RAG supplies substance.

  .venv\\Scripts\\python.exe -m rag.ask_instruct "soru"
  .venv\\Scripts\\python.exe -m rag.ask_instruct "soru" --model huihui_ai/qwen2.5-abliterate:14b-instruct

Compare its output head-to-head with `rag.ask` (the base-completion voice model).
"""
from __future__ import annotations

import argparse

from pipeline.config import load_settings
from rag.ollama_residency import ensure_absent
from rag.prompt import REFUSAL_TR
from runtime.ollama_http import ollama_json
from runtime.settings import RuntimeSettings

_DEFAULT_MODEL = "huihui_ai/qwen2.5-abliterate:7b-instruct"

_STYLE = []  # Supply consented style examples for your own dataset.

_SYSTEM = (
    "İzinli verilerle uyarlanmış bir dil modelisin. Verilen kaynaklara dayanarak "
    "soruyu açık ve tutarlı Türkçe ile yanıtla. Kanıtta olmayan bilgi ekleme. "
    "Öğrendiğin anlatım üslubunu koru; gerçek bir kişi olduğunu iddia etme."
)


def build_messages(question: str, spans) -> list[dict]:
    context = "\n\n".join(f"- {span.text.strip()}" for span in spans)
    user = (
        f"Senin videolarından alıntılar:\n\n{context}\n\n"
        f"Soru: {question}\n\n"
        "Yukarıdaki alıntılardaki bilgiyle, kendi üslubunla yanıtla:"
    )
    return [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}]


def generate_chat(
    messages: list[dict], model: str, temperature: float, num_predict: int, timeout: int = 300,
) -> str:
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    url = RuntimeSettings.from_environment().ollama_origin + "/api/chat"
    payload = ollama_json("POST", url, payload=body, timeout_s=float(timeout))
    return payload["message"]["content"].strip()


def answer(
    question: str,
    model: str = _DEFAULT_MODEL,
    ungrounded: bool = False,
    config: str | None = None,
    *,
    assess_fn=None,
    verifier_factory=None,
    generate_fn=None,
    ensure_absent_fn=None,
) -> str:
    from rag.evidence_pipeline import assess_question
    from rag.prompt import format_evidence_citations
    from rag.verifier import EvidenceVerifier

    settings = load_settings(config)
    rag = settings.rag
    active_generate = generate_fn or generate_chat
    active_ensure_absent = ensure_absent_fn or ensure_absent
    if ungrounded:
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": question.strip()},
        ]
        return active_generate(messages, model, rag.gen_temperature, 512)
    try:
        active_ensure_absent(model)
        decision = (assess_fn or assess_question)(
            question,
            settings,
            verifier=(verifier_factory or EvidenceVerifier.from_config)(rag),
        )
    except Exception as exc:
        return f"RAG kan\u0131t denetimi i\u00e7in model ge\u00e7i\u015fi ba\u015far\u0131s\u0131z oldu. Hata: {exc}."
    if decision.status == "unsupported":
        return REFUSAL_TR
    if decision.status == "error":
        return f"RAG kan\u0131t denetimi ba\u015far\u0131s\u0131z: {decision.failure_reason}."
    try:
        active_ensure_absent(rag.verifier_model)
    except Exception as exc:
        return f"RAG kan\u0131t denetimi i\u00e7in model ge\u00e7i\u015fi ba\u015far\u0131s\u0131z oldu. Hata: {exc}."
    text = active_generate(
        build_messages(decision.answer_question, decision.selected_spans),
        model,
        rag.gen_temperature,
        512,
    )
    notice = (
        "Kan\u0131t\u0131 bulunamayan k\u0131s\u0131m: " + "; ".join(decision.unsupported_claims) + "\n\n"
        if decision.status == "partial" else ""
    )
    citations = format_evidence_citations(decision.selected_spans)
    return notice + text + (f"\n\nKaynaklar:\n{citations}" if citations else "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask Speaker (instruct + verified RAG).")
    parser.add_argument("question")
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument(
        "--ungrounded",
        action="store_true",
        help="disable RAG before generation; no automatic fallback",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    print(answer(args.question, model=args.model, ungrounded=args.ungrounded, config=args.config))


if __name__ == "__main__":
    main()
