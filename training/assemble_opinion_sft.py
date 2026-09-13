"""Assemble the Claude-teacher opinion pairs (from the mining workflow) into a voice-SFT training
set: merge the new grounded his-take pairs with the existing 281 context-free voice pairs.

  .venv\\Scripts\\python.exe training\\assemble_opinion_sft.py <workflow_output.json>
"""
from __future__ import annotations

import json
import re
import sys

SYS = ("Sen izinli egitim verisindeki anlatim uslubuna uyarlanmis bir "
       "dil modelisin. Sana bir soru sorulur; ogrendigin anlatim uslubuyla, "
       "KONUYA SADIK kalarak akici ve net Turkce yanit ver. Lafi dagitma, sorulani cevapla.")

DD = "E:/youtuber-clone/data/dataset"
PROF = re.compile(r"\b(sik|am[ıi]na|amc[ıi]k|g[öo]t|pi[çc]|o[çc]\b|orospu|yarr?a[kğ]|pezevenk|"
                  r"kahpe|ibne|gavat|anan[ıi]|avrad[ıi]|sokay[ıi]m|siktir|yavşak|şerefsiz)", re.I)


def has_prof(t: str) -> bool:
    return bool(PROF.search(t or ""))


def main() -> None:
    out_path = sys.argv[1] if len(sys.argv) > 1 else \
        "C:/Users/Developer/AppData/Local/Temp/claude/E--youtuber-clone/2dc2d129-b91c-45ef-a38f-373a8ee422a2/tasks/w10vhe3i8.output"
    blob = json.load(open(out_path, encoding="utf-8"))
    results = blob["result"] if isinstance(blob, dict) else blob
    if isinstance(results, str):
        results = json.loads(results)

    rows, prof_n, covered, abst = [], 0, 0, 0
    by_topic = {}
    for r in results:
        n = 0
        for p in r.get("pairs", []):
            q, a = p.get("question", "").strip(), p.get("answer", "").strip()
            if not q or not a:
                continue
            rows.append({"messages": [{"role": "system", "content": SYS},
                                      {"role": "user", "content": q},
                                      {"role": "assistant", "content": a}]})
            if has_prof(a):
                prof_n += 1
            n += 1
        by_topic[r["topic"]] = (r.get("covered"), n)
        covered += 1 if r.get("covered") else 0
        abst += 0 if r.get("covered") else 1

    # write the new opinion pairs alone, and a combined train (existing 281 + new)
    open(f"{DD}/opinion_sft.jsonl", "w", encoding="utf-8").write(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n")

    existing = [json.loads(l) for l in open(f"{DD}/voice_sft_train.jsonl", encoding="utf-8") if l.strip()]
    combined = existing + rows
    open(f"{DD}/voice_sft_v2_train.jsonl", "w", encoding="utf-8").write(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in combined) + "\n")

    print(f"NEW opinion pairs: {len(rows)} ({covered} topics covered, {abst} abstention)")
    print(f"profanity in new answers: {prof_n}/{len(rows)} = {100*prof_n/max(len(rows),1):.0f}%")
    print(f"existing voice pairs: {len(existing)}")
    print(f"-> {DD}/opinion_sft.jsonl ({len(rows)})")
    print(f"-> {DD}/voice_sft_v2_train.jsonl (COMBINED {len(combined)})")
    print("\nper-topic pairs (covered?):")
    for t, (cov, n) in by_topic.items():
        print(f"  {t:24s} {'OK ' if cov else 'ABS'} {n}")


if __name__ == "__main__":
    main()
