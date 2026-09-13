"""Diagnostic: does his documentary/historical content exist in data/clean, and which tier is it in?
Tells us if the clean (T1-only) index is missing his main documentary corpus.

  .venv\\Scripts\\python.exe -m eval.coverage_check
"""
from __future__ import annotations

import collections
import glob
import json
import os

from rag.hisonly_filter import classify_video

KWS = ["atatürk", "mustafa kemal", "hitler", "musaddık", "cumhuriyet", "osmanlı", "vesayet", "pkk", "öcalan"]


def main() -> None:
    tier_of = {}
    for fp in glob.glob("data/clean/*.json"):
        if fp.endswith(".review.json"):
            continue
        d = json.load(open(fp, encoding="utf-8"))
        vid = d.get("video_id") or os.path.basename(fp)[:-5]
        tier_of[vid] = (classify_video(d.get("title", "")), d.get("title", ""), d.get("segments", []))

    print(f"{'keyword':14s}{'total':>8s}{'T1':>8s}{'T2':>8s}{'EXCL':>8s}")
    for kw in KWS:
        c = collections.Counter()
        for tier, _title, segs in tier_of.values():
            for s in segs:
                if kw in (s.get("text") or "").lower():
                    c[tier] += 1
                    c["total"] += 1
        print(f"{kw:14s}{c['total']:8d}{c['T1']:8d}{c['T2']:8d}{c['EXCL']:8d}")

    tc = collections.Counter(t for t, _, _ in tier_of.values())
    segc = collections.Counter()
    for t, _, segs in tier_of.values():
        segc[t] += len(segs)
    print("\nvideos per tier:", dict(tc))
    print("segments per tier:", dict(segc))


if __name__ == "__main__":
    main()
