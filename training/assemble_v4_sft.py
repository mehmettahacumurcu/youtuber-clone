"""Assemble v4 = v3 backbone (2,428 closed-book) + grounded_v4 (RAFT span rows) + abstain_v4
(in-voice declines). The three behaviors the final model needs: his takes closed-book, faithful
use of retrieved spans, and honest declines instead of invention.

Rules encoded here:
- backbone rows pass through untouched (style/opinion anchor; also the recipe's replay mass)
- grounded questions MAY repeat backbone questions (same Q closed-book AND grounded = good
  contrast); dedup is only within-bucket
- abstention questions must NOT collide with any answered question (a decline + a confident
  answer to the same Q is contradictory supervision) -> colliding abstentions are dropped
- profanity floor checked per bucket and overall (swearing is a hard constraint of the project)
- val = the stable 12 (eval_loss continuity) + held-out grounded/abstain rows so val loss sees
  the new behaviors too

  .venv\\Scripts\\python.exe -m training.assemble_v4_sft
"""
from __future__ import annotations

import json
import os
import random
import re

DD = "E:/youtuber-clone/data/dataset"
BACKBONE = f"{DD}/voice_sft_v3_train.jsonl"
GROUNDED = f"{DD}/grounded_v4.jsonl"
ABSTAIN = f"{DD}/abstain_v4.jsonl"
OLD_VAL = f"{DD}/voice_sft_val.jsonl"
OUT_TRAIN = f"{DD}/voice_sft_v4_train.jsonl"
OUT_VAL = f"{DD}/voice_sft_v4_val.jsonl"
VAL_HOLDOUT = 6      # per new bucket
PROFANITY_FLOOR = 0.35
SEED = 7

PROF = re.compile(r"\b(sik|am[ıi]na|amc[ıi]k|g[öo]t|pi[çc]|o[çc]\b|orospu|yarr?a[kğ]|pezevenk|"
                  r"kahpe|ibne|gavat|anan[ıi]|avrad[ıi]|sokay[ıi]m|siktir|yavşak|şerefsiz)", re.I)


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (t or "").lower())).strip()


def load(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"  (missing: {path} — treating as empty)")
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def q_of(row: dict) -> str:
    u = row["messages"][1]["content"]
    # grounded rows embed the question after the [Soru] marker; bare rows ARE the question
    return u.rsplit("[Soru]", 1)[-1].strip() if "[Soru]" in u else u.strip()


def a_of(row: dict) -> str:
    return row["messages"][-1]["content"]


def prof_rate(rows: list[dict]) -> float:
    return sum(1 for r in rows if PROF.search(a_of(r))) / max(len(rows), 1)


def main() -> None:
    backbone = load(BACKBONE)
    grounded = load(GROUNDED)
    abstain = load(ABSTAIN)
    old_val = load(OLD_VAL)

    answered_qs = {norm(q_of(r)) for r in backbone} | {norm(q_of(r)) for r in grounded}
    ab_kept = [r for r in abstain if norm(q_of(r)) not in answered_qs]
    if len(ab_kept) < len(abstain):
        print(f"dropped {len(abstain) - len(ab_kept)} abstentions colliding with answered questions")

    rng = random.Random(SEED)
    g_val = rng.sample(grounded, min(VAL_HOLDOUT, len(grounded))) if grounded else []
    a_val = rng.sample(ab_kept, min(VAL_HOLDOUT, len(ab_kept))) if ab_kept else []
    g_ids, a_ids = {id(r) for r in g_val}, {id(r) for r in a_val}
    g_train = [r for r in grounded if id(r) not in g_ids]
    a_train = [r for r in ab_kept if id(r) not in a_ids]

    train = backbone + g_train + a_train
    rng.shuffle(train)          # interleave modes so no tail-of-epoch mode bias
    val = old_val + g_val + a_val

    with open(OUT_TRAIN, "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n")
    with open(OUT_VAL, "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r, ensure_ascii=False) for r in val) + "\n")

    chars = sum(len(m["content"]) for r in train for m in r["messages"])
    print(f"\nv4 TRAIN = {len(train)} rows -> {OUT_TRAIN}  (~{int(chars * 0.292) // 1000}k tok)")
    print(f"  backbone {len(backbone)} | grounded {len(g_train)} | abstain {len(a_train)}")
    print(f"v4 VAL   = {len(val)} rows -> {OUT_VAL}  (12 stable + {len(g_val)} grounded + {len(a_val)} abstain)")
    for name, rows in (("backbone", backbone), ("grounded", g_train), ("abstain", a_train),
                       ("OVERALL", train)):
        if rows:
            rate = prof_rate(rows)
            flag = "  <-- BELOW FLOOR" if name == "OVERALL" and rate < PROFANITY_FLOOR else ""
            print(f"  profanity {name:9s}: {rate:.0%}{flag}")
    years = sum(1 for r in train if re.search(r"\b(19|20)\d\d\b", a_of(r)))
    print(f"  rows asserting a 4-digit year: {years} ({years / max(len(train), 1):.0%})")


if __name__ == "__main__":
    main()
