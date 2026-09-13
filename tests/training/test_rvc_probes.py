from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from training.rvc_probes import PROBE_TEXTS, write_probes


def test_probe_texts_are_exact_utf8_turkish() -> None:
    assert len(PROBE_TEXTS) == 5
    assert PROBE_TEXTS[0] == "Bak şimdi sana açık açık söyleyeyim, bu işin aslı tamamen başka."
    assert "Türkiye'nin" in PROBE_TEXTS[1]
    assert "Şu çocuğun" in PROBE_TEXTS[3]
    assert "uzun uzun konuştuk" in PROBE_TEXTS[4]


def test_write_probes_uses_one_synthesizer_and_records_audio(tmp_path: Path) -> None:
    calls: list[str] = []

    def synth(text: str) -> tuple[np.ndarray, int]:
        calls.append(text)
        return np.zeros(2_400, dtype=np.float32), 24_000

    manifest_path = write_probes(
        output_dir=tmp_path,
        synthesizer=synth,
        checkpoint_path=tmp_path / "fake.pth",
        checkpoint_sha256="abc123",
        reference_names=("ref_a.wav", "ref_b.wav"),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert calls == list(PROBE_TEXTS)
    assert manifest["schema"] == "rvc_xtts_probes.v1"
    assert manifest["checkpoint"]["sha256"] == "abc123"
    assert manifest["references"] == ["ref_a.wav", "ref_b.wav"]
    assert [row["path"] for row in manifest["probes"]] == [
        f"probe_{index:02d}.wav" for index in range(1, 6)
    ]
    assert all(
        sf.info(tmp_path / row["path"]).samplerate == 24_000
        for row in manifest["probes"]
    )
    assert all(row["duration_s"] == pytest.approx(0.1) for row in manifest["probes"])


@pytest.mark.parametrize(
    "waveform",
    [
        np.array([], dtype=np.float32),
        np.array([0.0, np.nan], dtype=np.float32),
        np.array([0.0, np.inf], dtype=np.float32),
        np.zeros((10, 2), dtype=np.float32),
    ],
)
def test_write_probes_rejects_invalid_synthesis_without_manifest(
    tmp_path: Path, waveform: np.ndarray
) -> None:
    def synth(text: str) -> tuple[np.ndarray, int]:
        return waveform, 24_000

    with pytest.raises(ValueError, match="finite mono samples"):
        write_probes(
            output_dir=tmp_path,
            synthesizer=synth,
            checkpoint_path=tmp_path / "fake.pth",
            checkpoint_sha256="abc123",
        )

    assert not (tmp_path / "manifest.json").exists()


def test_write_probes_rejects_unexpected_sample_rate(tmp_path: Path) -> None:
    def synth(text: str) -> tuple[np.ndarray, int]:
        return np.zeros(100, dtype=np.float32), 22_050

    with pytest.raises(ValueError, match="24000 Hz"):
        write_probes(
            output_dir=tmp_path,
            synthesizer=synth,
            checkpoint_path=tmp_path / "fake.pth",
            checkpoint_sha256="abc123",
        )
