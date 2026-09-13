"""Assemble the big per-video opinion pairs (+ the 24-topic pairs) into the scaled training set.
Dedups near-duplicate questions/answers (his stream clips repeat takes). Keeps the 374 test set
(voice_sft_train.jsonl) untouched; writes the big set + a combined v3 separately.

  .venv\\Scripts\\python.exe training\\assemble_big_sft.py <big_workflow_output.json>
"""
from __future__ import annotations

import json
import os
import re
import sys

SYS = ("Sen izinli egitim verisindeki anlatim uslubuna uyarlanmis bir "
       "dil modelisin. Sana bir soru sorulur; ogrendigin anlatim uslubuyla, "
       "KONUYA SADIK kalarak akici ve net Turkce yanit ver. Lafi dagitma, sorulani cevapla.")
DD = "E:/youtuber-clone/data/dataset"
PROF = re.compile(r"\b(sik|am[ıi]na|amc[ıi]k|g[öo]t|pi[çc]|o[çc]\b|orospu|yarr?a[kğ]|pezevenk|"
                  r"kahpe|ibne|gavat|anan[ıi]|avrad[ıi]|sokay[ıi]m|siktir|yavşak|şerefsiz)", re.I)


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (t or "").lower())).strip()


def has_prof(t: str) -> bool:
    return bool(PROF.search(t or ""))


def to_row(q: str, a: str) -> dict:
    return {"messages": [{"role": "system", "content": SYS},
                         {"role": "user", "content": q}, {"role": "assistant", "content": a}]}


def pairs_from_output(path: str) -> list[tuple[str, str]]:
    blob = json.load(open(path, encoding="utf-8"))
    res = blob["result"] if isinstance(blob, dict) else blob
    if isinstance(res, str):
        res = json.loads(res)
    out = []
    for r in res:
        for p in (r.get("pairs") or []):
            q, a = (p.get("question") or "").strip(), (p.get("answer") or "").strip()
            if q and a:
                out.append((q, a))
    return out


def main() -> None:
    big_out = sys.argv[1] if len(sys.argv) > 1 else \
        "C:/Users/Developer/AppData/Local/Temp/claude/E--youtuber-clone/2dc2d129-b91c-45ef-a38f-373a8ee422a2/tasks/woei90y0o.output"
    big = pairs_from_output(big_out)

    # also fold in the 24-topic opinion pairs (already rows)
    topic = []
    op = f"{DD}/opinion_sft.jsonl"
    if os.path.exists(op):
        for l in open(op, encoding="utf-8"):
            if l.strip():
                m = json.loads(l)["messages"]
                topic.append((m[1]["content"], m[2]["content"]))

    seen_q, seen_a, new = set(), set(), []
    for q, a in big + topic:
        nq, na = norm(q), norm(a)
        if not nq or not na or nq in seen_q or na in seen_a:
            continue
        seen_q.add(nq)
        seen_a.add(na)
        new.append(to_row(q, a))

    open(f"{DD}/voice_sft_big.jsonl", "w", encoding="utf-8").write(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in new) + "\n")

    orig = [json.loads(l) for l in open(f"{DD}/voice_sft_train.281bak", encoding="utf-8") if l.strip()]
    combined = orig + new
    open(f"{DD}/voice_sft_v3_train.jsonl", "w", encoding="utf-8").write(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in combined) + "\n")

    prof = sum(1 for r in new if has_prof(r["messages"][-1]["content"]))
    chars = sum(len(m["content"]) for r in combined for m in r["messages"])
    print(f"big video pairs: {len(big)} | topic pairs: {len(topic)} | after dedup: {len(new)}")
    print(f"profanity in new: {prof}/{len(new)} = {100*prof//max(len(new),1)}%")
    print(f"-> {DD}/voice_sft_big.jsonl ({len(new)} new pairs)")
    print(f"-> {DD}/voice_sft_v3_train.jsonl (281 orig + {len(new)} = {len(combined)} rows, ~{int(chars*0.292)//1000}k tok)")
    print("(voice_sft_train.jsonl = the 374 test set, left untouched)")


if __name__ == "__main__":
    main()
