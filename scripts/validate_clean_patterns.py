r"""Zero-false-positive validation harness for Stage-6.5 cleaning rules.

Before any ASR correction (``clean.corrections``) or structural drop (``clean.drop_patterns``)
goes into config.yaml, run it through here against the REAL corpus. The harness prints EVERY
occurrence of the candidate — literal (word-boundary) or regex — with its video_id and the full
segment text, plus a total count. You then eyeball the matches: if the "wrong" form is ALWAYS an
ASR error (never legitimate Turkish), it is safe to add. His speech contains real dates, quotes,
and formal documentary narration, so this manual zero-FP gate is mandatory.

    # how many times does the (wrong) form 'duayen' appear, and is it ever legitimate?
    python -m scripts.validate_clean_patterns --config config.yaml "duayen"

    # vet a structural regex candidate (won't ship unless zero false positives)
    python -m scripts.validate_clean_patterns --config config.yaml --regex "\d{1,2} (Ocak|Temmuz) \d{4}"
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import load_settings  # noqa: E402


def scan_corpus(filter_dir: Path, candidate: str, *, is_regex: bool) -> list[dict[str, Any]]:
    """Return every segment whose text matches ``candidate``.

    Literal mode matches the candidate at WORD BOUNDARIES (mirrors apply_corrections' ``\\b``);
    regex mode compiles the candidate as-is. Each hit: {video_id, start, text, match}.
    """
    if is_regex:
        pat = re.compile(candidate)
    else:
        pat = re.compile(rf"\b{re.escape(candidate)}\b")

    hits: list[dict[str, Any]] = []
    for fp in sorted(filter_dir.glob("*.json")):
        if fp.name.endswith(".review.json"):
            continue
        try:
            doc = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        vid = doc.get("video_id") or fp.stem
        for seg in doc.get("segments", []):
            text = seg.get("text") or ""
            m = pat.search(text)
            if m:
                hits.append({
                    "video_id": vid,
                    "start": seg.get("start"),
                    "text": text.strip(),
                    "match": m.group(0),
                })
    return hits


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="validate_clean_patterns", description=__doc__)
    p.add_argument("candidate", help="literal wrong-form (word-boundary) or regex with --regex")
    p.add_argument("--config", default=None)
    p.add_argument("--regex", action="store_true", help="treat candidate as a regex")
    p.add_argument("--dir", default=None, help="override the filter dir to scan")
    p.add_argument("--limit", type=int, default=0, help="print at most N matches (0 = all)")
    args = p.parse_args(argv)

    settings = load_settings(args.config)
    scan_dir = Path(args.dir) if args.dir else settings.paths.filter_dir
    hits = scan_corpus(scan_dir, args.candidate, is_regex=args.regex)

    vids = sorted({h["video_id"] for h in hits})
    print(f"candidate={args.candidate!r}  mode={'regex' if args.regex else 'literal'}  "
          f"matches={len(hits)}  across {len(vids)} video(s)\n")
    shown = hits if args.limit <= 0 else hits[: args.limit]
    for h in shown:
        st = h["start"]
        st_s = f"{st:.0f}s" if isinstance(st, (int, float)) else str(st)
        print(f"[{h['video_id']} {st_s}] …{h['text']}")
    if args.limit and len(hits) > args.limit:
        print(f"\n(+{len(hits) - args.limit} more; re-run with --limit 0 to see all)")
    if not hits:
        print("ZERO matches — candidate not present in corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
