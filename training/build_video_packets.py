"""Build one content packet per his-only (T1) video for the big opinion-SFT generation. Each packet
= the video's title + up to CAP coherent chunks (evenly sampled across long videos), so a teacher
agent can read a single video's dense content and synthesize his takes from it (no retrieval, no
cross-video dilution). Reads data/clean directly — does NOT touch Qdrant (UI server stays up).

  .venv\\Scripts\\python.exe -m training.build_video_packets
"""
from __future__ import annotations

import collections
import json
import statistics

from pipeline.config import load_settings
from rag.chunk import chunk_corpus
from rag.hisonly_filter import select_videos

OUT = "data/dataset/video_packets.json"
CAP = 12          # max chunks per video packet (~7k tokens); long videos are evenly sampled


def main() -> None:
    s = load_settings()
    sel = select_videos(s.paths.clean_dir, s.paths.meta_dir, tiers=("T1",))
    chunks = list(chunk_corpus(s.paths.clean_dir, s.paths.meta_dir, 600, 128, select=sel))
    byvid: dict[str, list] = collections.defaultdict(list)
    for c in chunks:
        byvid[c.video_id].append(c)

    packets = []
    for vid, cs in byvid.items():
        cs.sort(key=lambda c: c.start)
        if len(cs) > CAP:
            idx = sorted({round(i * (len(cs) - 1) / (CAP - 1)) for i in range(CAP)})
            cs = [cs[i] for i in idx]
        packets.append({"video_id": vid, "title": cs[0].title, "n_chunks": len(cs),
                        "text": "\n\n".join(c.text for c in cs)})

    json.dump(packets, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    chars = [len(p["text"]) for p in packets]
    print(f"{len(packets)} video packets -> {OUT}")
    print(f"chars/packet: median {int(statistics.median(chars))}, max {max(chars)}, "
          f"total ~{sum(chars)//1000}k chars (~{int(sum(chars)*0.292)//1000}k tok)")
    print("N =", len(packets))


if __name__ == "__main__":
    main()
