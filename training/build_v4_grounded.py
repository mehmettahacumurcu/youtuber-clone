"""v4 grounded (RAFT-style) rows — teach the model to USE his retrieved words at inference.

Why: run2/v3 models were trained 100% context-free, so the grounded prompt is OOD — given gold
spans they half-ignore them and invent (stage0_ep2 / way1_out evidence). Fix = training rows whose
user turn is EXACTLY the inference grounded format (rag.prompt.build_grounded_user: same header,
same [Senin sözlerin] block, same MAX_SPANS/SPAN_CHAR_CAP), with answers faithful to the spans.

Two modes:
  prep   — derive per-T1-video span menus (same chunking as the video packets: 600-tok chunks,
           evenly sampled, CAP 12, PRE-TRUNCATED to SPAN_CHAR_CAP so the teacher only ever sees
           what the training row will contain) -> data/dataset/grounded_tasks_v4.json
  parse  — read the teacher workflow output (per task: {"rows":[{"question","span_idx":[..],
           "answer"}]}) -> data/dataset/grounded_v4.jsonl chat rows. ~40%% of rows get ONE
           distractor span from another video spliced in at a random slot (seeded), teaching
           span-selection under real retrieval noise (the per-video diversity cap admits junk).

  .venv\\Scripts\\python.exe -m training.build_v4_grounded prep
  .venv\\Scripts\\python.exe -m training.build_v4_grounded parse <workflow_output.json>
"""
from __future__ import annotations

import collections
import json
import random
import re
import sys

from rag.prompt import MAX_SPANS, SPAN_CHAR_CAP, SYS, build_grounded_user

DD = "E:/youtuber-clone/data/dataset"
TASKS = f"{DD}/grounded_tasks_v4.json"
OUT = f"{DD}/grounded_v4.jsonl"
CAP = 12                 # span-menu size per video (mirrors build_video_packets)
DISTRACTOR_RATE = 0.4    # fraction of rows that get one off-video distractor span
SEED = 7


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (t or "").lower())).strip()


def prep() -> None:
    from pipeline.config import load_settings
    from rag.chunk import chunk_corpus
    from rag.hisonly_filter import select_videos

    s = load_settings()
    sel = select_videos(s.paths.clean_dir, s.paths.meta_dir, tiers=("T1",))
    chunks = list(chunk_corpus(s.paths.clean_dir, s.paths.meta_dir, 600, 128, select=sel))
    byvid: dict[str, list] = collections.defaultdict(list)
    for c in chunks:
        byvid[c.video_id].append(c)

    tasks = []
    for vid, cs in byvid.items():
        cs.sort(key=lambda c: c.start)
        if len(cs) > CAP:
            idx = sorted({round(i * (len(cs) - 1) / (CAP - 1)) for i in range(CAP)})
            cs = [cs[i] for i in idx]
        # pre-truncate: the teacher must ground the answer ONLY in text the training row will show
        spans = [c.text.strip()[:SPAN_CHAR_CAP] for c in cs]
        tasks.append({"video_id": vid, "title": cs[0].title, "spans": spans})

    json.dump(tasks, open(TASKS, "w", encoding="utf-8"), ensure_ascii=False)
    n_spans = sum(len(t["spans"]) for t in tasks)
    print(f"{len(tasks)} video tasks, {n_spans} spans (<= {SPAN_CHAR_CAP} chars each) -> {TASKS}")


def parse(path: str) -> None:
    blob = json.load(open(path, encoding="utf-8"))
    res = blob["result"] if isinstance(blob, dict) else blob
    if isinstance(res, str):
        res = json.loads(res)

    tasks = json.load(open(TASKS, encoding="utf-8"))
    by_vid = {t["video_id"]: t for t in tasks}
    rng = random.Random(SEED)
    vids = [t["video_id"] for t in tasks]

    rows, seen_q, bad = [], set(), collections.Counter()
    for r in res:
        vid = r.get("video_id")
        task = by_vid.get(vid)
        if not task:
            bad["unknown_video"] += 1
            continue
        for p in (r.get("rows") or []):
            q = (p.get("question") or "").strip()
            a = (p.get("answer") or "").strip()
            idxs = p.get("span_idx") or []
            inline = [s.strip() for s in (p.get("spans") or []) if s and s.strip()]
            if not q or not a or not (idxs or inline):
                bad["empty_field"] += 1
                continue
            if inline:                       # retrofit path: window texts carried directly
                spans = inline[:MAX_SPANS]
            else:
                try:
                    spans = [task["spans"][i] for i in idxs][:MAX_SPANS]
                except (IndexError, TypeError):
                    bad["bad_span_idx"] += 1
                    continue
            nq = norm(q)
            if nq in seen_q:
                bad["dup_question"] += 1
                continue
            # reject verbatim span-copy answers (we want synthesis in voice, not parroting)
            na = norm(a)
            if any(na and na in norm(sp) for sp in spans):
                bad["verbatim_copy"] += 1
                continue
            # distractor may only fill a FREE slot — replacing a chosen span would strip support
            # the coverage/number/proper gates were computed on (audit 2026-07-02 caught this)
            if len(spans) < MAX_SPANS and rng.random() < DISTRACTOR_RATE and len(vids) > 1:
                other = rng.choice([v for v in vids if v != vid])
                distractor = by_vid[other]["spans"][0]
                spans.insert(rng.randrange(len(spans) + 1), distractor)
            seen_q.add(nq)
            rows.append({"messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": build_grounded_user(q, spans)},
                {"role": "assistant", "content": a}]})

    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"{len(rows)} grounded rows -> {OUT}")
    if bad:
        print("rejected:", dict(bad))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "prep":
        prep()
    elif len(sys.argv) >= 3 and sys.argv[1] == "parse":
        parse(sys.argv[2])
    else:
        raise SystemExit(__doc__)
