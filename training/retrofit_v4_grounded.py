"""Retrofit the EXISTING Opus teacher pairs into v4 grounded rows — no new generation.

The big-run raw output (272 entries, video_id + pairs) tells us which VIDEO each answer came
from but not which chunks. This script recovers that mechanically. The teacher read FULL
~600-token chunks, but a training/inference row only shows <=3 spans of <=SPAN_CHAR_CAP chars —
so the menu here is SLIDING WINDOWS (cap-sized, 50% overlap) over the same evenly-sampled chunks
the packet had, and the matcher picks the windows that actually contain the support (mirrors
inference, where rag.retrieve.refine_windows injects each chunk's best window, not its head).

Keep a pair only if the chosen visible windows support the answer:
  - every number/year asserted in the answer appears in the chosen windows (numbers are the
    confabulation signature — zero tolerance)
  - >=70% of proper nouns covered
  - >=60% rarity-weighted content coverage overall
Kept pairs are emitted with INLINE span texts and pushed through training.build_v4_grounded.parse
(same dedup/distractor/row-building as generated rows). Cap 3 rows/video (best-covered first).

  .venv\\Scripts\\python.exe -m training.retrofit_v4_grounded [raw_output_path]
"""
from __future__ import annotations

import collections
import json
import re
import sys

from rag.prompt import SPAN_CHAR_CAP
from training.build_v4_grounded import CAP, parse

DD = "E:/youtuber-clone/data/dataset"
RAW_DEFAULT = (r"C:\Users\Developer\AppData\Local\Temp\claude\E--youtuber-clone"
               r"\2dc2d129-b91c-45ef-a38f-373a8ee422a2\tasks\w9xetjc1e.output")
OUT_TASKS = f"{DD}/grounded_retrofit_out.json"

MAX_SPANS = 3
PER_VID_CAP = 3
COV_MIN = 0.45          # rarity-weighted content coverage by the chosen windows. Calibrated by
                        # eyeball 2026-07-01: 0.45-0.60 pairs passing the HARD gates below were
                        # consistently faithful (synthesis paraphrases, so lexical coverage
                        # under-scores); the number/proper gates carry the anti-confab load.
PROPER_MIN = 0.70       # share of proper nouns that must appear in the windows
MIN_GAIN = 0.08         # marginal-coverage floor per added window (0.03 let stray irrelevant
                        # windows scrape in on a handful of shared stems)
WIN_STRIDE = 400        # 50% overlap at SPAN_CHAR_CAP=800

_STOP = {"ama", "ancak", "aslında", "bence", "bile", "bunu", "bunun", "böyle", "çünkü", "daha",
         "değil", "diye", "falan", "filan", "gibi", "için", "kadar", "lazım", "mesela", "yani",
         "zaten", "şey", "şimdi", "sonra", "önce", "bütün", "kendi", "olarak", "olan", "oluyor",
         "böylece", "burada", "orada", "hani", "işte"}


def _toks(t: str) -> list[str]:
    return re.findall(r"[a-zçğıöşü0-9]+", (t or "").lower().replace("i̇", "i"))


def _stem(t: str) -> str:
    return t[:5] if len(t) >= 6 else t


def _content_stems(t: str) -> set[str]:
    return {_stem(x) for x in _toks(t) if len(x) >= 4 and x not in _STOP and not x.isdigit()}


def _numbers(t: str) -> set[str]:
    return set(re.findall(r"\d+", t or ""))


def _propers(t: str) -> set[str]:
    out = set()
    for sent in re.split(r"[.!?…]\s+", t or ""):
        words = sent.split()
        for w in words[1:]:                              # skip sentence-initial capitals
            w = w.split("'")[0].split("’")[0]
            if len(w) >= 3 and w[0].isupper() and not w.isupper():
                out.add(_stem(w.lower()))
    return out


def _windows(text: str, size: int = SPAN_CHAR_CAP, stride: int = WIN_STRIDE) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        if j < len(text):
            k = text.rfind(" ", i + size // 2, j)
            if k > 0:
                j = k
        out.append(text[i:j].strip())
        if j >= len(text):
            break
        i += stride
    return [w for w in out if w]


def _full_chunk_menus() -> dict[str, list[str]]:
    """Per-video window menus over the SAME evenly-sampled chunks the teacher packets used."""
    from pipeline.config import load_settings
    from rag.chunk import chunk_corpus
    from rag.hisonly_filter import select_videos

    s = load_settings()
    sel = select_videos(s.paths.clean_dir, s.paths.meta_dir, tiers=("T1",))
    byvid: dict[str, list] = collections.defaultdict(list)
    for c in chunk_corpus(s.paths.clean_dir, s.paths.meta_dir, 600, 128, select=sel):
        byvid[c.video_id].append(c)
    menus: dict[str, list[str]] = {}
    for vid, cs in byvid.items():
        cs.sort(key=lambda c: c.start)
        if len(cs) > CAP:
            idx = sorted({round(i * (len(cs) - 1) / (CAP - 1)) for i in range(CAP)})
            cs = [cs[i] for i in idx]
        menus[vid] = [w for c in cs for w in _windows(c.text)]
    return menus


def retrofit(raw_path: str) -> None:
    blob = json.load(open(raw_path, encoding="utf-8"))
    res = blob["result"] if isinstance(blob, dict) else blob
    if isinstance(res, str):
        res = json.loads(res)
    menus = _full_chunk_menus()

    out, rej = [], collections.Counter()
    n_pairs, cov_kept = 0, []
    for entry in res:
        vid = entry.get("video_id")
        menu = menus.get(vid)
        if not menu:
            rej["no_menu_for_video"] += len(entry.get("pairs") or [])
            continue
        menu_stems = [_content_stems(sp) for sp in menu]
        # rarity weight: a stem appearing in many windows identifies the video, not the take
        df = collections.Counter(s for st in menu_stems for s in st)
        wt = {s: 1.0 / (1 + df[s]) for s in df}

        scored_rows = []
        for p in (entry.get("pairs") or []):
            n_pairs += 1
            q, a = (p.get("question") or "").strip(), (p.get("answer") or "").strip()
            if not q or not a:
                rej["empty"] += 1
                continue
            a_stems = _content_stems(a)
            if not a_stems:
                rej["no_content"] += 1
                continue
            total_w = sum(wt.get(s, 1.0) for s in a_stems)

            # greedy set-cover: pick windows by marginal rarity-weighted coverage
            chosen, covered = [], set()
            for _ in range(MAX_SPANS):
                best, best_gain = None, 0.0
                for i, st in enumerate(menu_stems):
                    if i in chosen:
                        continue
                    gain = sum(wt.get(s, 1.0) for s in (a_stems & st) - covered)
                    if gain > best_gain:
                        best, best_gain = i, gain
                if best is None or best_gain < MIN_GAIN * total_w:
                    break
                chosen.append(best)
                covered |= a_stems & menu_stems[best]

            cov = sum(wt.get(s, 1.0) for s in covered) / total_w if total_w else 0.0
            if not chosen or cov < COV_MIN:
                rej["low_coverage"] += 1
                continue
            union_txt = " ".join(menu[i] for i in chosen)
            union_stems = set().union(*(menu_stems[i] for i in chosen))
            if _numbers(a) - _numbers(union_txt):
                rej["number_unsupported"] += 1
                continue
            props = _propers(a)
            if props and len(props & union_stems) / len(props) < PROPER_MIN:
                rej["proper_unsupported"] += 1
                continue
            scored_rows.append((cov, {"question": q, "spans": [menu[i] for i in chosen],
                                      "answer": a}))

        scored_rows.sort(key=lambda x: -x[0])
        kept = scored_rows[:PER_VID_CAP]
        rej["over_video_cap"] += max(0, len(scored_rows) - PER_VID_CAP)
        cov_kept.extend(c for c, _ in kept)
        if kept:
            out.append({"video_id": vid, "rows": [r for _, r in kept]})

    json.dump({"result": out}, open(OUT_TASKS, "w", encoding="utf-8"), ensure_ascii=False)
    n_kept = sum(len(e["rows"]) for e in out)
    print(f"retrofit: {n_kept}/{n_pairs} pairs span-supported "
          f"({len(out)}/{len(res)} videos) -> {OUT_TASKS}")
    print("rejected:", dict(rej))
    if cov_kept:
        cov_kept.sort()
        print(f"kept coverage: min {cov_kept[0]:.2f} / median {cov_kept[len(cov_kept)//2]:.2f} / "
              f"max {cov_kept[-1]:.2f}")
    print("\n--- building rows via build_v4_grounded.parse ---")
    parse(OUT_TASKS)


if __name__ == "__main__":
    retrofit(sys.argv[1] if len(sys.argv) > 1 else RAW_DEFAULT)
