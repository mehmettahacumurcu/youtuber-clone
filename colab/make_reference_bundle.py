"""Zip the private reference assets (seed clip + pool) for per-session upload to Colab.

The active reference pool — not just the seed clip — drives identify quality, so it must
travel to Colab or identification won't match local calibration. Output is gitignored.

Run locally: python colab/make_reference_bundle.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import load_settings  # noqa: E402


def main() -> int:
    s = load_settings()
    ref_dir = s.paths.data_dir / "reference"
    if not (s.paths.reference_clip).exists():
        print(f"ERROR: seed clip missing: {s.paths.reference_clip}", flush=True)
        return 2
    out = s.paths.data_dir / "reference_bundle.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in ref_dir.rglob("*"):
            if p.is_file():
                # .as_posix() forces forward-slash arcnames so the bundle (authored on
                # Windows) extracts to reference/... on the Linux Colab runtime.
                z.write(p, p.relative_to(s.paths.data_dir).as_posix())  # store as reference/...
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
