"""Stage-0 go/no-go: can the context-free speaker-qwen3-run2-ep2 actually USE his injected opinion
spans? For each topic, compare BARE (no context → what it invents) vs GROUNDED (his real spans).

Decision: if GROUNDED restates his take in his voice (vs ignoring it / verbatim-copying / drifting)
→ Way-1 (no retrain) is viable. If it ignores even clean gold context → escalate to Way-2 (RAFT).

Spans below are HAND-PICKED clean stance sentences from data/clean (not retrieved) so this isolates
the model's in-context-use ability, independent of retrieval quality.

  .venv\\Scripts\\python.exe -m eval.stage0_probe
  .venv\\Scripts\\python.exe -m eval.stage0_probe --model huihui_ai/qwen3-abliterated:14b-v2
"""
from __future__ import annotations

import argparse

from rag.ask_grounded import grounded_answer

PROBES = [
    {
        "topic": "mutlak butlan",
        "q": "Mutlak butlan meselesi hakkında ne düşünüyorsun?",
        "spans": [
            "Böyle bu şekilde mutlak butlanla geri gelip bunun yedirilmesi, kabul ettirilmesi mümkün değil.",
            "Dersimli KK mutlak butlanla birlikte Cumhuriyet Halk Partisi'ne geri döndü.",
            "Butlan kararının çıkması, bunun siyasi olması; özgürlük hiç umurlarında değil, adamlar başka şey analiz ediyorlar.",
        ],
    },
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="speaker-qwen3-run2-ep2")
    args = ap.parse_args()
    for p in PROBES:
        print("=" * 72)
        print(f"TOPIC: {p['topic']}   MODEL: {args.model}")
        print(f"Q: {p['q']}")
        print("=" * 72)
        print("\n----- BARE (no context — what it invents on its own) -----")
        print(grounded_answer(p["q"], spans=[], model=args.model))
        print("\n----- GROUNDED (his real spans injected) -----")
        print(grounded_answer(p["q"], spans=p["spans"], model=args.model))
        print("\n(his actual stance in the spans: 'mutlak butlanla CHP'ye geri dönüş yedirilemez/"
              "kabul ettirilemez; karar siyasi')\n")


if __name__ == "__main__":
    main()
