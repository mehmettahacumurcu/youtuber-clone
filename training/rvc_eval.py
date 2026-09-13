"""Evaluate RVC comparison WAVs with audio, Turkish CER, and ECAPA metrics."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import soundfile as sf
import torch

Transcriber = Callable[[Path], str]
Embedder = Callable[[Path], torch.Tensor]


def normalise_turkish(text: str) -> str:
    """Normalize comparison text without discarding Turkish letters."""
    value = re.sub(r"[^\w\s]", "", text.casefold(), flags=re.UNICODE)
    return " ".join(value.split())


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference = normalise_turkish(reference)
    hypothesis = normalise_turkish(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(reference, hypothesis) / len(reference)


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if waveform.shape[1] != 1:
        raise ValueError(f"evaluation audio must be mono: {path}")
    waveform = waveform[:, 0]
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"evaluation audio must contain finite samples: {path}")
    return waveform, sample_rate


def audio_metrics(source_path: Path, output_path: Path) -> dict[str, float | int | bool]:
    """Return source-relative duration and output integrity metrics."""
    source, source_rate = _read_mono(Path(source_path))
    output, output_rate = _read_mono(Path(output_path))
    source_duration = len(source) / source_rate
    output_duration = len(output) / output_rate
    duration_ratio = output_duration / source_duration
    peak = float(np.max(np.abs(output)))
    clipped_fraction = float(np.mean(np.abs(output) >= 0.999))
    return {
        "source_duration_s": source_duration,
        "output_duration_s": output_duration,
        "duration_ratio": duration_ratio,
        "duration_pass": 0.95 <= duration_ratio <= 1.05,
        "output_sample_rate": output_rate,
        "output_channels": 1,
        "peak": peak,
        "clipped_fraction": clipped_fraction,
        "clipping_pass": clipped_fraction < 0.001,
    }


def _normalized_embedding(embed: Embedder, path: Path) -> torch.Tensor:
    value = torch.as_tensor(embed(path), dtype=torch.float32).flatten()
    if value.numel() == 0 or not torch.isfinite(value).all() or value.norm() == 0:
        raise ValueError(f"invalid speaker embedding: {path}")
    return torch.nn.functional.normalize(value, dim=0)


def _mean_similarity(value: torch.Tensor, references: Sequence[torch.Tensor]) -> float:
    return float(torch.stack([torch.dot(value, reference) for reference in references]).mean())


def _resolve_path(path_value: str, bundle_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else bundle_root / path


def evaluate_rows(
    *,
    conversion_rows: Sequence[dict],
    bundle_root: Path,
    transcribe: Transcriber,
    embed: Embedder,
    allow_model_failure: bool = False,
) -> list[dict]:
    """Evaluate converted rows against their matching raw probe and target refs."""
    bundle_root = Path(bundle_root)
    bundle_manifest = json.loads(
        (bundle_root / "manifest.json").read_text(encoding="utf-8")
    )
    reference_paths = [
        bundle_root / row["path"] for row in bundle_manifest.get("eval_refs", [])
    ]
    if not reference_paths:
        raise ValueError("bundle manifest contains no eval_refs")

    raw_by_probe = {
        row["probe_id"]: _resolve_path(str(row["path"]), bundle_root)
        for row in conversion_rows
        if row.get("kind") == "raw"
    }
    reference_embeddings: list[torch.Tensor] | None
    shared_errors: list[str] = []
    try:
        reference_embeddings = [
            _normalized_embedding(embed, path) for path in reference_paths
        ]
    except Exception as error:
        if not allow_model_failure:
            raise
        reference_embeddings = None
        shared_errors.append(f"ECAPA setup/reference failure: {error}")

    results: list[dict] = []
    for row in conversion_rows:
        if row.get("kind") != "converted":
            continue
        probe_id = str(row["probe_id"])
        if probe_id not in raw_by_probe:
            raise ValueError(f"converted row has no raw source: {probe_id}")
        source_path = raw_by_probe[probe_id]
        output_path = _resolve_path(str(row["path"]), bundle_root)
        result = {**row, **audio_metrics(source_path, output_path)}
        errors = list(shared_errors)

        try:
            raw_hypothesis = transcribe(source_path)
            output_hypothesis = transcribe(output_path)
            result["raw_transcript"] = raw_hypothesis
            result["transcript"] = output_hypothesis
            result["raw_cer"] = character_error_rate(str(row["text"]), raw_hypothesis)
            result["cer"] = character_error_rate(str(row["text"]), output_hypothesis)
        except Exception as error:
            if not allow_model_failure:
                raise
            result.update({"raw_transcript": None, "transcript": None, "raw_cer": None, "cer": None})
            errors.append(f"Whisper failure: {error}")

        if reference_embeddings is not None:
            try:
                raw_embedding = _normalized_embedding(embed, source_path)
                output_embedding = _normalized_embedding(embed, output_path)
                raw_similarity = _mean_similarity(raw_embedding, reference_embeddings)
                output_similarity = _mean_similarity(output_embedding, reference_embeddings)
                result["raw_speaker_similarity"] = raw_similarity
                result["speaker_similarity"] = output_similarity
                result["speaker_similarity_delta"] = output_similarity - raw_similarity
            except Exception as error:
                if not allow_model_failure:
                    raise
                result.update(
                    {
                        "raw_speaker_similarity": None,
                        "speaker_similarity": None,
                        "speaker_similarity_delta": None,
                    }
                )
                errors.append(f"ECAPA failure: {error}")
        else:
            result.update(
                {
                    "raw_speaker_similarity": None,
                    "speaker_similarity": None,
                    "speaker_similarity_delta": None,
                }
            )
        result["evaluation_errors"] = errors
        results.append(result)
    return results


def summarize_gates(rows: Sequence[dict]) -> dict:
    """Summarize automatic gates independently for every retrieval rate."""
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("index_rate") is not None:
            grouped[float(row["index_rate"])].append(row)
    by_rate: dict[str, dict] = {}
    for rate in sorted(grouped):
        values = grouped[rate]
        gains = sum(
            row.get("speaker_similarity_delta") is not None
            and row["speaker_similarity_delta"] > 0
            for row in values
        )
        cer_deltas = [
            row["cer"] - row["raw_cer"]
            for row in values
            if row.get("cer") is not None and row.get("raw_cer") is not None
        ]
        median_cer_delta = statistics.median(cer_deltas) if cer_deltas else None
        technical = all(
            bool(row.get("duration_pass")) and bool(row.get("clipping_pass"))
            for row in values
        )
        complete = len({row["probe_id"] for row in values}) == 5
        automatic_pass = bool(
            complete
            and gains >= 4
            and median_cer_delta is not None
            and median_cer_delta <= 0.05
            and technical
        )
        by_rate[f"{rate:.2f}"] = {
            "probe_count": len({row["probe_id"] for row in values}),
            "similarity_improved_probes": gains,
            "median_cer_delta": median_cer_delta,
            "technical_audio_pass": technical,
            "automatic_pass": automatic_pass,
            "human_identity_pass": None,
        }
    return {"by_index_rate": by_rate, "human_review_required": True}


def load_transcriber(device: str) -> Transcriber:
    """Load Whisper lazily so unit tests and audio-only fallback stay lightweight."""
    from faster_whisper import WhisperModel

    model = WhisperModel(
        "large-v3",
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
    )

    def transcribe(path: Path) -> str:
        segments, _ = model.transcribe(str(path), language="tr")
        return " ".join(segment.text for segment in segments).strip()

    return transcribe


def load_ecapa_encoder(device: str, cache_dir: Path) -> Embedder:
    """Load the same ECAPA identity model used by the corpus pipeline."""
    import torchaudio
    from speechbrain.pretrained import EncoderClassifier

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(cache_dir),
        run_opts={"device": device},
    )

    def embed(path: Path) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(str(path))
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16_000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        with torch.inference_mode():
            value = classifier.encode_batch(waveform.to(device)).squeeze().detach().cpu()
        return torch.nn.functional.normalize(value, dim=0)

    return embed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/.cache/speechbrain/rvc_eval"))
    parser.add_argument("--allow-model-failure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    conversion = json.loads(args.conversion_manifest.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup_error: Exception | None = None
    try:
        transcribe = load_transcriber(device)
        embed = load_ecapa_encoder(device, args.cache_dir)
    except Exception as error:
        if not args.allow_model_failure:
            raise
        setup_error = error

        def fail(path: Path):
            raise RuntimeError(f"evaluation model setup failed: {setup_error}")

        transcribe = fail
        embed = fail

    rows = evaluate_rows(
        conversion_rows=conversion["rows"],
        bundle_root=args.bundle_root,
        transcribe=transcribe,
        embed=embed,
        allow_model_failure=args.allow_model_failure,
    )
    report = {
        "schema": "rvc_metrics.v1",
        "rows": rows,
        "gates": summarize_gates(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"metrics: {len(rows)} converted rows -> {args.output}")


if __name__ == "__main__":
    main()
