from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from training.rvc_eval import (
    audio_metrics,
    evaluate_rows,
    normalise_turkish,
    summarize_gates,
)


def _write(path: Path, values: np.ndarray, sample_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, values.astype(np.float32), sample_rate)


def _make_eval_fixture(root: Path) -> tuple[list[dict], Path, dict[str, torch.Tensor]]:
    raw = root / "probes" / "raw_xtts" / "probe_01.wav"
    converted = root / "comparisons" / "probe_01_e150_i050.wav"
    ref_a = root / "eval_refs" / "a_0001.wav"
    ref_b = root / "eval_refs" / "b_0001.wav"
    for path in (raw, converted, ref_a, ref_b):
        _write(path, np.linspace(-0.2, 0.2, 1_600, dtype=np.float32))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "eval_refs": [
                    {"path": "eval_refs/a_0001.wav"},
                    {"path": "eval_refs/b_0001.wav"},
                ]
            }
        ),
        encoding="utf-8",
    )
    text = "Şu çocuğun yaptığı şey gerçekten çok yanlıştı."
    rows = [
        {
            "kind": "raw",
            "probe_id": "probe_01",
            "text": text,
            "path": str(raw),
            "epoch": None,
            "index_rate": None,
        },
        {
            "kind": "converted",
            "probe_id": "probe_01",
            "text": text,
            "path": str(converted),
            "epoch": 150,
            "index_rate": 0.5,
        },
    ]
    embeddings = {
        raw.name: torch.tensor([0.0, 1.0]),
        converted.name: torch.tensor([0.95, 0.05]),
        ref_a.name: torch.tensor([1.0, 0.0]),
        ref_b.name: torch.tensor([0.9, 0.1]),
    }
    return rows, root, embeddings


def test_audio_metrics_reports_duration_peak_and_clipping(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    output = tmp_path / "output.wav"
    _write(source, np.array([0.0, 0.5, -0.5]), sample_rate=3)
    _write(output, np.array([0.0, 1.0, -1.0]), sample_rate=3)

    result = audio_metrics(source, output)

    assert result["duration_ratio"] == pytest.approx(1.0)
    assert result["peak"] == pytest.approx(1.0, abs=1e-4)
    assert result["clipped_fraction"] == pytest.approx(2 / 3)
    assert result["output_channels"] == 1


def test_normalise_turkish_keeps_letters_and_removes_punctuation() -> None:
    assert normalise_turkish("Şu Çocuğun, yaptığı!") == "şu çocuğun yaptığı"


def test_evaluate_rows_uses_injected_transcriber_and_encoder(tmp_path: Path) -> None:
    rows, bundle_root, embeddings = _make_eval_fixture(tmp_path)

    result = evaluate_rows(
        conversion_rows=rows,
        bundle_root=bundle_root,
        transcribe=lambda path: rows[0]["text"],
        embed=lambda path: embeddings[path.name],
    )

    assert len(result) == 1
    assert result[0]["cer"] == 0.0
    assert result[0]["speaker_similarity_delta"] > 0.8
    assert result[0]["duration_pass"] is True
    assert result[0]["clipping_pass"] is True


def test_evaluate_rows_preserves_audio_metrics_when_models_fail(tmp_path: Path) -> None:
    rows, bundle_root, _ = _make_eval_fixture(tmp_path)

    def fail(path: Path):
        raise RuntimeError("model unavailable")

    result = evaluate_rows(
        conversion_rows=rows,
        bundle_root=bundle_root,
        transcribe=fail,
        embed=fail,
        allow_model_failure=True,
    )

    assert result[0]["duration_pass"] is True
    assert result[0]["cer"] is None
    assert result[0]["speaker_similarity_delta"] is None
    assert any("model unavailable" in error for error in result[0]["evaluation_errors"])


def test_evaluate_rows_rejects_missing_audio(tmp_path: Path) -> None:
    rows, bundle_root, embeddings = _make_eval_fixture(tmp_path)
    Path(rows[1]["path"]).unlink()
    with pytest.raises(FileNotFoundError, match="probe_01_e150"):
        evaluate_rows(
            conversion_rows=rows,
            bundle_root=bundle_root,
            transcribe=lambda path: rows[0]["text"],
            embed=lambda path: embeddings[path.name],
        )


def test_summarize_gates_requires_four_probe_gains_and_metric_thresholds() -> None:
    rows = []
    for probe in range(1, 6):
        rows.append(
            {
                "probe_id": f"probe_{probe:02d}",
                "index_rate": 0.5,
                "cer": 0.04,
                "raw_cer": 0.02,
                "speaker_similarity_delta": 0.1 if probe <= 4 else -0.01,
                "duration_pass": True,
                "clipping_pass": True,
            }
        )

    summary = summarize_gates(rows)

    rate = summary["by_index_rate"]["0.50"]
    assert rate["similarity_improved_probes"] == 4
    assert rate["median_cer_delta"] == pytest.approx(0.02)
    assert rate["automatic_pass"] is True

    rows[0]["cer"] = 0.20
    rows[1]["cer"] = 0.20
    rows[2]["cer"] = 0.20
    assert summarize_gates(rows)["by_index_rate"]["0.50"]["automatic_pass"] is False
