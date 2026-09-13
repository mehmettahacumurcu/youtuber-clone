"""Build a deterministic, manifest-verified audio bundle for the RVC Colab POC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import soundfile as sf

SEED = 20260715
TARGET_SECONDS = 1440.0
EXPECTED_SAMPLE_RATE = 22_050
MIN_DURATION = 3.0
MAX_DURATION = 11.1


@dataclass(frozen=True)
class Clip:
    name: str
    source_id: str
    path: Path
    text: str
    duration_s: float
    sample_rate: int
    channels: int


def source_id_from_stem(stem: str) -> str:
    """Return the video/source prefix from a ``source_NNNN`` WAV stem."""
    source, separator, suffix = stem.rpartition("_")
    if not separator or not source or not suffix.isdigit():
        raise ValueError(f"WAV stem must end in a numeric suffix: {stem}")
    return source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_clips(
    metadata_path: Path,
    wavs_dir: Path,
    *,
    expected_sample_rate: int = EXPECTED_SAMPLE_RATE,
) -> list[Clip]:
    """Load metadata rows and reject audio outside the frozen POC contract."""
    metadata_path = Path(metadata_path)
    wavs_dir = Path(wavs_dir)
    rows: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        metadata_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        fields = raw_line.split("|", 2)
        if len(fields) != 3 or not fields[0].strip():
            raise ValueError(f"invalid metadata row {line_number}: expected name|text|text")
        name, text, _ = fields
        if name in rows:
            raise ValueError(f"duplicate metadata name: {name}")
        rows[name] = text

    clips: list[Clip] = []
    for name in sorted(rows):
        path = wavs_dir / f"{name}.wav"
        if not path.is_file():
            raise FileNotFoundError(f"metadata WAV is missing: {name} ({path})")
        info = sf.info(path)
        duration_s = info.frames / info.samplerate
        if info.samplerate != expected_sample_rate:
            raise ValueError(
                f"{name} must be {expected_sample_rate} Hz, got {info.samplerate}"
            )
        if info.channels != 1:
            raise ValueError(f"{name} must be mono, got {info.channels} channels")
        if not MIN_DURATION <= duration_s <= MAX_DURATION:
            raise ValueError(
                f"{name} duration {duration_s:.3f}s is outside "
                f"{MIN_DURATION:.1f}-{MAX_DURATION:.1f}s"
            )
        clips.append(
            Clip(
                name=name,
                source_id=source_id_from_stem(name),
                path=path,
                text=rows[name],
                duration_s=duration_s,
                sample_rate=info.samplerate,
                channels=info.channels,
            )
        )
    if not clips:
        raise ValueError("metadata contains no audio rows")
    return clips


def split_holdouts(clips: Sequence[Clip]) -> tuple[list[Clip], list[Clip]]:
    """Reserve one median-duration reference from every source group."""
    grouped: dict[str, list[Clip]] = defaultdict(list)
    for clip in clips:
        grouped[clip.source_id].append(clip)
    refs: list[Clip] = []
    for source_id in sorted(grouped):
        group = grouped[source_id]
        median = statistics.median(clip.duration_s for clip in group)
        refs.append(min(group, key=lambda clip: (abs(clip.duration_s - median), clip.name)))
    ref_names = {clip.name for clip in refs}
    return [clip for clip in clips if clip.name not in ref_names], refs


def round_robin_select(
    clips: Sequence[Clip],
    target_seconds: float = TARGET_SECONDS,
    seed: int = SEED,
) -> list[Clip]:
    """Select reproducibly across all sources until the duration target is reached."""
    if target_seconds <= 0:
        raise ValueError("target duration must be positive")
    grouped: dict[str, list[Clip]] = defaultdict(list)
    for clip in clips:
        grouped[clip.source_id].append(clip)
    queues: dict[str, deque[Clip]] = {}
    for source_id in sorted(grouped):
        values = sorted(grouped[source_id], key=lambda clip: clip.name)
        random.Random(f"{seed}:{source_id}").shuffle(values)
        queues[source_id] = deque(values)

    selected: list[Clip] = []
    duration = 0.0
    while duration < target_seconds and any(queues.values()):
        for source_id in sorted(queues):
            if queues[source_id] and duration < target_seconds:
                clip = queues[source_id].popleft()
                selected.append(clip)
                duration += clip.duration_s
    if duration < target_seconds:
        raise ValueError(
            f"target duration {target_seconds:.1f}s exceeds available {duration:.1f}s"
        )
    return selected


def _audio_entry(clip: Clip, archive_path: str) -> dict[str, object]:
    return {
        "name": clip.name,
        "source_id": clip.source_id,
        "path": archive_path,
        "text": clip.text,
        "duration_s": round(clip.duration_s, 6),
        "sample_rate": clip.sample_rate,
        "channels": clip.channels,
        "sha256": _sha256(clip.path),
    }


def _load_probes(probes_dir: Path, manifest_path: Path) -> tuple[dict, list[dict]]:
    probe_manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    rows = probe_manifest.get("probes")
    if probe_manifest.get("schema") != "rvc_xtts_probes.v1" or not isinstance(rows, list):
        raise ValueError("probe manifest must use schema rvc_xtts_probes.v1")
    if len(rows) != 5:
        raise ValueError(f"probe manifest must contain exactly 5 probes, got {len(rows)}")
    seen: set[str] = set()
    output: list[dict] = []
    for row in rows:
        relative = str(row.get("path", ""))
        if not relative or relative in seen:
            raise ValueError(f"invalid or duplicate probe path: {relative!r}")
        seen.add(relative)
        source = Path(probes_dir) / relative
        if not source.is_file():
            raise FileNotFoundError(f"probe WAV is missing: {relative} ({source})")
        info = sf.info(source)
        if info.samplerate != 24_000 or info.channels != 1:
            raise ValueError(f"probe {relative} must be mono 24000 Hz")
        output.append(
            {
                "id": row.get("id") or Path(relative).stem,
                "text": str(row.get("text", "")),
                "path": f"probes/raw_xtts/{Path(relative).name}",
                "duration_s": round(info.frames / info.samplerate, 6),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "sha256": _sha256(source),
                "_source": source,
            }
        )
    return probe_manifest, output


def _verify_archive(path: Path, expected_hashes: dict[str, str]) -> dict:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"corrupt ZIP member: {bad}")
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        for archive_path, expected in expected_hashes.items():
            actual = hashlib.sha256(archive.read(archive_path)).hexdigest()
            if actual != expected:
                raise ValueError(f"ZIP hash mismatch for {archive_path}")
    return manifest


def build_bundle(
    *,
    metadata_path: Path,
    wavs_dir: Path,
    probes_dir: Path,
    probe_manifest_path: Path,
    output_path: Path,
    target_seconds: float = TARGET_SECONDS,
    seed: int = SEED,
) -> dict:
    """Create the final ZIP atomically and verify every stored audio hash."""
    clips = collect_clips(Path(metadata_path), Path(wavs_dir))
    pool, refs = split_holdouts(clips)
    training = round_robin_select(pool, target_seconds=target_seconds, seed=seed)
    probe_manifest, probes = _load_probes(Path(probes_dir), Path(probe_manifest_path))

    training_rows = [
        _audio_entry(clip, f"dataset/{clip.path.name}") for clip in training
    ]
    ref_rows = [
        _audio_entry(clip, f"eval_refs/{clip.path.name}") for clip in refs
    ]
    public_probes = [{key: value for key, value in row.items() if key != "_source"} for row in probes]
    source_counts = Counter(clip.source_id for clip in training)
    manifest = {
        "schema": "rvc_poc_bundle.v1",
        "seed": seed,
        "target_duration_s": target_seconds,
        "training_duration_s": round(sum(clip.duration_s for clip in training), 6),
        "training_source_counts": dict(sorted(source_counts.items())),
        "training": training_rows,
        "eval_refs": ref_rows,
        "probes": public_probes,
        "probe_synthesis": probe_manifest,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    expected_hashes: dict[str, str] = {}
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for clip, row in zip(training, training_rows, strict=True):
                archive.write(clip.path, row["path"])
                expected_hashes[str(row["path"])] = str(row["sha256"])
            for clip, row in zip(refs, ref_rows, strict=True):
                archive.write(clip.path, row["path"])
                expected_hashes[str(row["path"])] = str(row["sha256"])
            for row in probes:
                archive.write(Path(row["_source"]), str(row["path"]))
                expected_hashes[str(row["path"])] = str(row["sha256"])
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
            )
        verified = _verify_archive(temporary, expected_hashes)
        os.replace(temporary, output_path)
        return verified
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--wavs", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-seconds", type=float, default=TARGET_SECONDS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_bundle(
        metadata_path=args.metadata,
        wavs_dir=args.wavs,
        probes_dir=args.probes,
        probe_manifest_path=args.probe_manifest,
        output_path=args.output,
        target_seconds=args.target_seconds,
    )
    print(
        f"bundle: {len(manifest['training'])} training clips / "
        f"{manifest['training_duration_s'] / 60:.2f} min / "
        f"{len(manifest['eval_refs'])} refs / {len(manifest['probes'])} probes "
        f"-> {args.output}"
    )


if __name__ == "__main__":
    main()
