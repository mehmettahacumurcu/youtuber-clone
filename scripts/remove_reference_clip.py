"""Remove contaminated reference clips from the speaker pool, cleanly.

Pair to add_reference_clip.py — for pruning pool members that sound contaminated on
review. Removal = delete the wav + drop the entry from candidates.json + drop the index
from active.json. We do NOT renumber the remaining candidate_index values, so the
embedding sidecar (candidate_embeddings.npy, indexed by candidate_index) stays aligned for
every surviving member; the removed rows simply become unused (identify re-embeds the active
wavs at runtime and never reads the npy anyway, so identification is unaffected).

    python scripts/remove_reference_clip.py --file 05_9CDd..._10243.2-10246.9.wav --file ...
    python scripts/remove_reference_clip.py --index 5 10 12 14
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import REPO_ROOT, load_settings  # noqa: E402
from pipeline.identify.pool import candidates_manifest_path  # noqa: E402
from ui.state import active_pool_path, load_active_pool, save_active_pool  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="remove_reference_clip", description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--index", type=int, nargs="*", default=[], help="candidate_index values to remove")
    p.add_argument("--file", action="append", default=[], help="wav basename to remove (repeatable)")
    args = p.parse_args(argv)
    if not args.index and not args.file:
        print("ERROR: pass --index and/or --file", file=sys.stderr)
        return 2

    settings = load_settings(args.config)
    manifest_path = candidates_manifest_path(settings)
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    cands = doc["candidates"]
    by_idx = {c["candidate_index"]: c for c in cands}

    # Resolve targets (by index and/or by filename) to a set of candidate_index values.
    targets: set[int] = set(args.index)
    fnames = set(args.file)
    for c in cands:
        if Path(c["file"]).name in fnames:
            targets.add(c["candidate_index"])
    missing_files = fnames - {Path(c["file"]).name for c in cands}
    for mf in sorted(missing_files):
        print(f"  WARN: filename not found in pool: {mf}")
    targets = {t for t in targets if t in by_idx}
    if not targets:
        print("nothing to remove (no matching candidates)")
        return 1

    # 1) delete wav files
    for t in sorted(targets):
        wav = REPO_ROOT / by_idx[t]["file"]
        if wav.exists():
            wav.unlink()
        print(f"removed #{t}: {Path(by_idx[t]['file']).name}")

    # 2) drop entries from candidates.json (keep all other candidate_index values UNCHANGED)
    doc["candidates"] = [c for c in cands if c["candidate_index"] not in targets]
    doc["pool_size_actual"] = len(doc["candidates"])
    doc["generated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) drop from active.json (no-op if already inactive)
    active = load_active_pool(settings)["active_indices"]
    new_active = [i for i in active if i not in targets]
    if new_active != active:
        save_active_pool(settings, new_active)

    print(f"\npool now: {len(doc['candidates'])} candidates, {len(new_active)} active")
    print("(candidate_embeddings.npy left intact — surviving rows stay index-aligned; "
          "identify re-embeds active wavs at runtime so identification is unaffected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
