"""Turn the assembled LLM corpus into train/val JSONL for the QLoRA pilot.

Input  : data/dataset/llm_corpus.jsonl       (--source clean, default; ASR-corrected canonical)
      or data/dataset/llm_corpus.raw.jsonl   (--source filter; the RAW uncorrected build)
Output : data/dataset/{train,val}.jsonl       (clean) or {train,val}.raw.jsonl (filter)
         one {"text": ...} per video. Raw is kept side-by-side so it is never lost/overwritten.

What it does (see finetune_plan / colab_run_status memory for the why):
  1. [SKIP] markers -> paragraph break. The corpus stitches non-contiguous kept segments
     with a literal "[SKIP]" line; a BASE model trained on that learns to EMIT "[SKIP]".
     A blank-line paragraph break gives the same discontinuity signal in plain prose.
  2. NFC-normalize, then re-apply filter.scrub_patterns (subtitle credits / thanks-for-watching)
     in place. The corpus was already scrubbed at the filter stage; this is the last line of
     defense (it once caught a subtitle-credit variant the filter stage missed mid-paragraph).
  3. Collapse runs of 3+ newlines to a paragraph break, trim. NO aggressive min-length
     filter (his style includes short interjections) — only drop a doc if it is empty.
  4. Hold out ~10% of WHOLE videos for validation (never split a video across train/val
     -> that would leak the same speaker/topic into both). Deterministic, seeded.
  5. Structural guard before writing: abort if any [SKIP] / 3+ newline run survives, or if a
     video_id appears in both splits. Artifact-signature residue (e.g. "Altyazı") is reported
     as a warning, not an abort, since those CAN be legitimate speech at full-corpus scale.

One document per video; EOS-between-docs and packing are handled at TRAIN time, NOT here.
SFTTrainer reads only the "text" field; video_id/title ride along for traceability.
CONTRACT for the train script (the single most consequential shaping line): it MUST append an
EOS per doc, e.g. `formatting_func=lambda ex: ex["text"] + tokenizer.eos_token`
(Llama-3.1 -> "<|end_of_text|>"), and assert `tokenizer.eos_token is not None`. Do NOT bake a
literal EOS string into "text" here — a base model would learn to emit the literal token.

CLI: python -m training.build_dataset [--config config.yaml] [--val-frac 0.1] [--seed 7]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

from pipeline.config import load_settings
from pipeline.filter.runner import scrub_text

# A "[SKIP]" line (optionally with surrounding blank lines) -> one paragraph break.
_SKIP_RE = re.compile(r"\s*\n?\s*\[SKIP\]\s*\n?\s*")
# A single newline NOT part of a "\n\n" paragraph break (defensive: segments are single-line today,
# but this makes the no-internal-newline invariant explicit instead of relying on the segmenter).
_LONE_NL_RE = re.compile(r"(?<!\n)\n(?!\n)")
_MULTI_NL_RE = re.compile(r"\n{3,}")

# Structural invariants — these are NEVER legitimate in training text; abort the build if seen.
_FORBIDDEN = {
    "[SKIP] marker": re.compile(r"\[SKIP\]"),
    "3+ newline run": re.compile(r"\n{3,}"),
}
# Content-heuristic artifact signatures — report as a warning (could be real speech at scale).
_ARTIFACT_SIGNATURES = {
    "subtitle-credit (Altyazı…)": re.compile(r"Altyaz[ıi]", re.IGNORECASE),
}

# Pilot exclude-list from the 2026-06-18 content review (data_quality_review memory). These docs are
# not-him / mixed-identity / legally hazardous and would poison the first-person style signal. Excluded
# from TRAIN only, AFTER the split, so the human-assessed val holdout (seed=7) stays byte-identical
# ("do not reshuffle the val seed"). All seven are train docs; none are in val. Pass exclude_ids=set()
# (CLI --keep-all) to disable. NOTE: this is a PILOT shortcut — before the full ~500-video run, fix the
# root cause (stronger speaker-ID + read-aloud-quotation detector) rather than hand-dropping.
PILOT_EXCLUDE = {
    "9AejUUYv35k",  # scripted comedy sketch — not him, zero style signal
    "sample00038",  # ~50% read-aloud Öcalan/Karayılan text + targeted threats — legal/ethical hazard
    "9CDdXsqqNME",  # ~12-guest moderated panel — majority guest speech
    "50U1bC8joHY",  # folklore guest interview — guest autobiography leaked past speaker-ID
    "AUr4uLeBi1k",  # Oğuzhan Uğur narrates ~half the doc
    "8URgE3ylK-c",  # guest-dominated interview
    "dTRQlPj75Ck",  # guest speech dominates
}


def clean_text(text: str, scrub_patterns: list[re.Pattern]) -> str:
    """Corpus text -> clean prose: NFC, [SKIP] -> paragraph break, scrub artifacts, tidy newlines."""
    text = unicodedata.normalize("NFC", text)
    text = _SKIP_RE.sub("\n\n", text)
    # Only deliberate "\n\n" paragraph breaks should remain; collapse any stray single newline so
    # per-paragraph scrubbing can't half-eat it (scrub_text treats \n as whitespace).
    text = _LONE_NL_RE.sub(" ", text)
    # scrub_text also collapses intra-line whitespace + strips, so run it per paragraph
    # UNCONDITIONALLY (an empty scrub list still normalizes whitespace consistently).
    text = "\n\n".join(scrub_text(p, scrub_patterns) for p in text.split("\n\n"))
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def split_videos(video_ids: list[str], val_frac: float, seed: int) -> tuple[set[str], set[str]]:
    """Deterministically hold out ~val_frac of WHOLE videos for validation."""
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0.0, 1.0), got {val_frac}")
    ordered = sorted(set(video_ids))  # dedupe + stable order so no id can straddle the slice
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_val = round(len(ordered) * val_frac)
    if val_frac > 0 and len(ordered) > 1:
        n_val = max(1, n_val)  # at least one val video when val_frac>0 and there's more than one
    val = set(ordered[:n_val])
    train = set(ordered[n_val:])
    return train, val


def check_outputs(train: list[dict], val: list[dict]) -> dict[str, int]:
    """Abort on structural violations; return a count of artifact-signature residue (warnings)."""
    leak = {r["video_id"] for r in train} & {r["video_id"] for r in val}
    if leak:
        raise AssertionError(f"train/val video_id leak: {sorted(leak)}")
    for label, pat in _FORBIDDEN.items():
        for split_name, rows in (("train", train), ("val", val)):
            bad = [r["video_id"] for r in rows if pat.search(r["text"])]
            if bad:
                raise AssertionError(f"{label} survived into {split_name}: {bad}")
    warnings: dict[str, int] = {}
    for label, pat in _ARTIFACT_SIGNATURES.items():
        n = sum(len(pat.findall(r["text"])) for r in (*train, *val))
        if n:
            warnings[label] = n
    return warnings


# Input corpus + output train/val names per source. "clean" = ASR-corrected canonical (what
# training consumes); "filter" = RAW (uncorrected) build, kept side-by-side so raw is never lost.
_SOURCE_FILES = {
    "clean": ("llm_corpus.jsonl", "train.jsonl", "val.jsonl"),
    "filter": ("llm_corpus.raw.jsonl", "train.raw.jsonl", "val.raw.jsonl"),
}


def build(config: str | None, val_frac: float, seed: int, exclude_ids: set[str] | None = None,
          source: str = "clean") -> dict:
    if source not in _SOURCE_FILES:
        raise ValueError(f"source must be one of {sorted(_SOURCE_FILES)}, got {source!r}")
    corpus_name, train_name, val_name = _SOURCE_FILES[source]
    settings = load_settings(config)
    exclude = PILOT_EXCLUDE if exclude_ids is None else exclude_ids
    scrub = [re.compile(p, re.IGNORECASE) for p in settings.filter.scrub_patterns]

    corpus_path = settings.paths.dataset_dir / corpus_name
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"{corpus_path} not found; run 'python -m pipeline.assemble --source {source}' first")
    records = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    cleaned: list[dict] = []
    dropped_empty = 0
    dropped_no_id = 0
    for rec in records:
        vid = rec.get("video_id")
        if not vid:
            dropped_no_id += 1
            continue
        text = clean_text(rec.get("text") or "", scrub)
        if not text:
            dropped_empty += 1
            continue
        cleaned.append({"video_id": vid, "title": rec.get("title"), "text": text})

    train_ids, val_ids = split_videos([r["video_id"] for r in cleaned], val_frac, seed)
    # Excluded (not-him / mixed / legal-hazard) videos are dropped from BOTH splits: they must not
    # train the style model NOR pollute the eval holdout. (The pilot excluded from train only, to
    # keep a hand-assessed val anchor byte-stable; at full-corpus scale the seeded split is fresh,
    # so we simply remove excludes everywhere. Raw build passes exclude=set() via --keep-all.)
    train = [r for r in cleaned if r["video_id"] in train_ids and r["video_id"] not in exclude]
    val = [r for r in cleaned if r["video_id"] in val_ids and r["video_id"] not in exclude]

    artifact_warnings = check_outputs(train, val)  # raises on structural violations, before any write

    def write(rows: list[dict], name: str) -> Path:
        path = settings.paths.dataset_dir / name
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
            encoding="utf-8",
            newline="\n",  # byte-identical on Windows and Linux (no CRLF translation)
        )
        return path

    train_path = write(train, train_name)
    val_path = write(val, val_name)

    def chars(rows: list[dict]) -> int:
        return sum(len(r["text"]) for r in rows)

    # ~3.42 chars/token for Llama-3.1 on his Turkish (base_model_tokenizer_finding memory). This is a
    # 5-video CALIBRATION estimate, not a real tokenization — re-run pipeline.assemble.tokenizer_report
    # on the full corpus before committing the final model.
    stats = {
        "source": source,
        "input_docs": len(records),
        "dropped_empty": dropped_empty,
        "dropped_no_id": dropped_no_id,
        "excluded_from_train": sorted(exclude),
        "train_docs": len(train),
        "val_docs": len(val),
        "train_chars": chars(train),
        "val_chars": chars(val),
        "train_tokens_est_at_3.42cpt": round(chars(train) / 3.42),
        "val_tokens_est_at_3.42cpt": round(chars(val) / 3.42),
        "artifact_signature_warnings": artifact_warnings,
        "val_ids": sorted(r["video_id"] for r in val),  # actual val docs (post-exclude)
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Build train/val JSONL from llm_corpus.jsonl")
    ap.add_argument("--config", default=None)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--keep-all", action="store_true", help="disable the PILOT_EXCLUDE content drop-list")
    ap.add_argument("--source", choices=["clean", "filter"], default="clean",
                    help="clean = ASR-corrected canonical (train.jsonl/val.jsonl); "
                         "filter = RAW build (train.raw.jsonl/val.raw.jsonl), pair with --keep-all")
    args = ap.parse_args()

    stats = build(args.config, args.val_frac, args.seed,
                  exclude_ids=set() if args.keep_all else None, source=args.source)
    if stats["artifact_signature_warnings"]:
        print(f"WARNING: artifact-signature residue (verify these are real speech, not artifacts): "
              f"{stats['artifact_signature_warnings']}", file=sys.stderr)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
