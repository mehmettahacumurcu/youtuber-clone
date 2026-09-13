from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from training.rvc_poc_bundle import (
    build_bundle,
    collect_clips,
    round_robin_select,
    source_id_from_stem,
    split_holdouts,
)


def _write_wav(
    path: Path,
    duration_s: float,
    *,
    sample_rate: int = 22_050,
    channels: int = 1,
) -> None:
    frames = round(duration_s * sample_rate)
    audio = np.zeros((frames, channels), dtype=np.float32)
    if channels == 1:
        audio = audio[:, 0]
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)


def _make_corpus(root: Path) -> tuple[Path, Path]:
    wavs = root / "wavs"
    lines: list[str] = []
    for source in ("a", "b", "c"):
        for index, duration in enumerate((3.0, 3.1, 3.2, 3.3, 3.4)):
            name = f"{source}_{index:04d}"
            _write_wav(wavs / f"{name}.wav", duration)
            text = f"{source} Türkçe metin {index}"
            lines.append(f"{name}|{text}|{text}\n")
    metadata = root / "metadata.csv"
    metadata.write_text("".join(lines), encoding="utf-8")
    return metadata, wavs


def _make_probes(root: Path) -> tuple[Path, Path]:
    probes = root / "probes"
    rows = []
    texts = [
        "Bak şimdi sana açık açık söyleyeyim, bu işin aslı tamamen başka.",
        "Türkiye'nin başkenti Ankara'dır.",
        "Sence bu ülkede gerçekten adalet var mı?",
        "Şu çocuğun yaptığı şey yanlıştı.",
        "Geçen hafta uzun uzun konuştuk.",
    ]
    for index, text in enumerate(texts, start=1):
        name = f"probe_{index:02d}.wav"
        _write_wav(probes / name, 0.2, sample_rate=24_000)
        rows.append({"id": f"probe_{index:02d}", "path": name, "text": text})
    manifest = probes / "manifest.json"
    manifest.write_text(
        json.dumps({"schema": "rvc_xtts_probes.v1", "probes": rows}, ensure_ascii=False),
        encoding="utf-8",
    )
    return probes, manifest


def test_source_id_requires_numeric_suffix() -> None:
    assert source_id_from_stem("video_a_0012") == "video_a"
    with pytest.raises(ValueError, match="numeric suffix"):
        source_id_from_stem("video_a_final")


def test_split_holdouts_reserves_one_median_clip_per_source(tmp_path: Path) -> None:
    metadata, wavs = _make_corpus(tmp_path)
    clips = collect_clips(metadata, wavs)

    training_pool, refs = split_holdouts(clips)

    assert {clip.source_id for clip in refs} == {"a", "b", "c"}
    assert [clip.name for clip in refs] == ["a_0002", "b_0002", "c_0002"]
    assert not ({clip.name for clip in refs} & {clip.name for clip in training_pool})


def test_round_robin_selection_is_reproducible_and_reaches_target(tmp_path: Path) -> None:
    metadata, wavs = _make_corpus(tmp_path)
    clips, _ = split_holdouts(collect_clips(metadata, wavs))

    first = round_robin_select(clips, target_seconds=20.0, seed=20260715)
    second = round_robin_select(clips, target_seconds=20.0, seed=20260715)

    assert [clip.name for clip in first] == [clip.name for clip in second]
    assert sum(clip.duration_s for clip in first) >= 20.0
    assert {clip.source_id for clip in first} == {"a", "b", "c"}


def test_collect_clips_rejects_audio_outside_contract(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    wavs = tmp_path / "wavs"
    metadata.write_text("bad_0001|metin|metin\n", encoding="utf-8")
    _write_wav(wavs / "bad_0001.wav", 3.0, sample_rate=16_000)
    with pytest.raises(ValueError, match="22050"):
        collect_clips(metadata, wavs)

    _write_wav(wavs / "bad_0001.wav", 3.0, channels=2)
    with pytest.raises(ValueError, match="mono"):
        collect_clips(metadata, wavs)


def test_collect_clips_rejects_duplicate_and_missing_rows(tmp_path: Path) -> None:
    wavs = tmp_path / "wavs"
    _write_wav(wavs / "a_0001.wav", 3.0)
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("a_0001|x|x\na_0001|x|x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        collect_clips(duplicate, wavs)

    missing = tmp_path / "missing.csv"
    missing.write_text("missing_0001|x|x\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing_0001"):
        collect_clips(missing, wavs)


def test_round_robin_selection_rejects_unreachable_target(tmp_path: Path) -> None:
    metadata, wavs = _make_corpus(tmp_path)
    clips, _ = split_holdouts(collect_clips(metadata, wavs))
    with pytest.raises(ValueError, match="exceeds available"):
        round_robin_select(clips, target_seconds=10_000.0)


def test_build_bundle_writes_verified_utf8_contract(tmp_path: Path) -> None:
    metadata, wavs = _make_corpus(tmp_path / "corpus")
    probes, probe_manifest = _make_probes(tmp_path / "probe_data")
    output = tmp_path / "nested" / "bundle.zip"

    manifest = build_bundle(
        metadata_path=metadata,
        wavs_dir=wavs,
        probes_dir=probes,
        probe_manifest_path=probe_manifest,
        output_path=output,
        target_seconds=20.0,
    )

    assert output.is_file()
    assert manifest["schema"] == "rvc_poc_bundle.v1"
    assert manifest["seed"] == 20260715
    assert manifest["training_duration_s"] >= 20.0
    assert len(manifest["eval_refs"]) == 3
    assert len(manifest["probes"]) == 5
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        names = set(archive.namelist())
        assert "manifest.json" in names
        assert sum(name.startswith("eval_refs/") for name in names) == 3
        assert sum(name.startswith("probes/raw_xtts/") for name in names) == 5
        decoded = json.loads(archive.read("manifest.json").decode("utf-8"))
        assert decoded["probes"][0]["text"].startswith("Bak şimdi")


def test_build_bundle_rejects_missing_probe(tmp_path: Path) -> None:
    metadata, wavs = _make_corpus(tmp_path / "corpus")
    probes, probe_manifest = _make_probes(tmp_path / "probe_data")
    (probes / "probe_05.wav").unlink()

    with pytest.raises(FileNotFoundError, match="probe_05"):
        build_bundle(
            metadata_path=metadata,
            wavs_dir=wavs,
            probes_dir=probes,
            probe_manifest_path=probe_manifest,
            output_path=tmp_path / "bundle.zip",
            target_seconds=20.0,
        )
