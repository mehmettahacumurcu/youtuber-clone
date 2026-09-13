"""Way-1 grounded answer path: answer from HIS OWN opinion spans, in his voice.

Branch A (spans present) → restate his take, grounded on the spans, no invention.
Branch B (no spans)      → abstain in his voice ("bu konuya pek girmemişim..."), do NOT invent.

Uses the CANONICAL grounded template from rag.prompt (system stays plain SYS; the grounding
instruction lives in the user turn) — the same format ui/speaker_studio.py sends and the v4 grounded
training rows use, so post-v4 this path is in-distribution. ABSTAIN_CLAUSE is a pre-v4 crutch for
Branch B: once v4's abstention rows are trained in, drop it and send the bare question.

Reused by eval/stage0_probe.py (inject hand-clean gold spans) and the UI/eval (retrieve from
speaker_clean). Does NOT touch the validated ask_finetuned.py default path.
"""
from __future__ import annotations

import argparse

from rag.prompt import SYS, build_grounded_user
from runtime.ollama_http import ollama_json
from runtime.settings import RuntimeSettings

_DEFAULT_MODEL = "speaker-qwen3-run2-ep2"

ABSTAIN_CLAUSE = (" Bu soruyla ilgili kendi videolarindan alinti yok; bu konuya girmemissem kendi "
                  "uslubunla kisaca 'bu konuya pek girmemisim, elimde bir sey yok' de, sakin uydurma.")


def _gen(messages: list[dict], model: str, num_predict: int = 512) -> str:
    body = {"model": model, "messages": messages, "stream": False, "keep_alive": 0,
            "options": {"temperature": 0.7, "top_p": 0.85, "repeat_penalty": 1.2,
                        "num_predict": num_predict}}
    url = RuntimeSettings.from_environment().ollama_origin + "/api/chat"
    payload = ollama_json("POST", url, payload=body, timeout_s=600.0)
    return payload["message"]["content"].strip()


def grounded_answer(question: str, spans: list[str] | None, model: str = _DEFAULT_MODEL,
                    grounding: bool = True) -> str:
    spans = [s.strip() for s in (spans or []) if s and s.strip()]
    if spans:
        sys = SYS
        if grounding:
            user = build_grounded_user(question, spans)
        else:
            # probe control arm: raw span injection with NO grounding instruction
            ctx = "\n\n".join(spans)
            user = f"Alintilar (senin kendi videolarindan):\n{ctx}\n\nSoru: {question}"
    else:
        sys = SYS + ABSTAIN_CLAUSE
        user = question
    return _gen([{"role": "system", "content": sys}, {"role": "user", "content": user}], model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--model", default=_DEFAULT_MODEL)
    ap.add_argument("--span", action="append", default=[], help="a his-span to ground on (repeatable)")
    args = ap.parse_args()
    print(grounded_answer(args.question, args.span, model=args.model))


if __name__ == "__main__":
    main()
