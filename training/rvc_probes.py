"""Generate fixed UTF-8 XTTS source probes for the RVC proof of concept."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import soundfile as sf

PROBE_TEXTS: tuple[str, ...] = (
    "Bak şimdi sana açık açık söyleyeyim, bu işin aslı tamamen başka.",
    "Türkiye'nin başkenti Ankara'dır ve en büyük şehri İstanbul'dur.",
    "Sence bu ülkede gerçekten adalet var mı, yoksa her şey anlatıldığı kadar basit mi?",
    "Şu çocuğun yaptığı şey gerçekten çok yanlıştı, değil mi?",
    "Geçen hafta arkadaşlarımla buluştuk, uzun uzun konuştuk, sonunda hiçbir yere varamadık.",
)

Synthesizer = Callable[[str], tuple[np.ndarray, int]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pick_dark_references(wavs_dir: Path, *, count: int = 6) -> list[Path]:
    """Match the Studio's dark/calm conditioning-reference policy."""
    import librosa

    candidates: list[tuple[float, Path]] = []
    for path in sorted(Path(wavs_dir).glob("*.wav"))[:160]:
        info = sf.info(path)
        if info.frames / info.samplerate < 6.0:
            continue
        waveform, sample_rate = librosa.load(path, sr=None, mono=True)
        centroid = float(
            librosa.feature.spectral_centroid(y=waveform, sr=sample_rate).mean()
        )
        candidates.append((centroid, path))
    candidates.sort(key=lambda item: (item[0], item[1].name))
    references = [path for _, path in candidates[:count]]
    if len(references) < count:
        references = sorted(Path(wavs_dir).glob("*.wav"))[:count]
    if len(references) < count:
        raise ValueError(f"need {count} conditioning references, found {len(references)}")
    return references


def load_xtts_synthesizer(
    model_dir: Path, wavs_dir: Path
) -> tuple[Synthesizer, tuple[str, ...], Path]:
    """Load the current XTTS checkpoint once and return the fixed inference closure."""
    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    model_dir = Path(model_dir)
    checkpoint = model_dir / "best_model.pth"
    config_path = model_dir / "config.json"
    vocab_path = model_dir / "vocab.json"
    for required in (checkpoint, config_path, vocab_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    config = XttsConfig()
    config.load_json(str(config_path))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=str(checkpoint),
        vocab_path=str(vocab_path),
        use_deepspeed=False,
    )
    if torch.cuda.is_available():
        model.cuda()
    references = pick_dark_references(Path(wavs_dir))
    gpt_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[str(path) for path in references],
        gpt_cond_len=30,
        max_ref_length=30,
        sound_norm_refs=True,
    )

    def synthesize(text: str) -> tuple[np.ndarray, int]:
        result = model.inference(
            text,
            "tr",
            gpt_latent,
            speaker_embedding,
            temperature=0.7,
            repetition_penalty=1.3,
            length_penalty=1.0,
        )
        return np.asarray(result["wav"], dtype=np.float32), 24_000

    return synthesize, tuple(path.name for path in references), checkpoint


def write_probes(
    *,
    output_dir: Path,
    synthesizer: Synthesizer,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    reference_names: Sequence[str] = (),
    texts: Sequence[str] = PROBE_TEXTS,
) -> Path:
    """Synthesize every fixed probe and publish a manifest only after readback."""
    if len(texts) != 5:
        raise ValueError(f"exactly 5 probe texts are required, got {len(texts)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    rows: list[dict[str, object]] = []

    for index, text in enumerate(texts, start=1):
        waveform, sample_rate = synthesizer(text)
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError(f"probe {index} must contain non-empty finite mono samples")
        if sample_rate != 24_000:
            raise ValueError(f"probe {index} must be 24000 Hz, got {sample_rate}")

        name = f"probe_{index:02d}.wav"
        final_path = output_dir / name
        temporary = output_dir / f"probe_{index:02d}.tmp.wav"
        temporary.unlink(missing_ok=True)
        sf.write(temporary, waveform, sample_rate, subtype="PCM_16")
        info = sf.info(temporary)
        if info.channels != 1 or info.samplerate != sample_rate or info.frames == 0:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"probe {index} failed WAV readback")
        os.replace(temporary, final_path)
        rows.append(
            {
                "id": f"probe_{index:02d}",
                "path": name,
                "text": text,
                "samples": info.frames,
                "duration_s": info.frames / info.samplerate,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "sha256": sha256_file(final_path),
            }
        )

    manifest = {
        "schema": "rvc_xtts_probes.v1",
        "checkpoint": {
            "path": str(Path(checkpoint_path)),
            "sha256": checkpoint_sha256,
        },
        "references": list(reference_names),
        "language": "tr",
        "temperature": 0.7,
        "repetition_penalty": 1.3,
        "length_penalty": 1.0,
        "probes": rows,
    }
    temporary_manifest = output_dir / "manifest.tmp.json"
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    json.loads(temporary_manifest.read_text(encoding="utf-8"))
    os.replace(temporary_manifest, manifest_path)
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--wavs-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    synthesize, references, checkpoint = load_xtts_synthesizer(
        args.model_dir, args.wavs_dir
    )
    manifest_path = write_probes(
        output_dir=args.output_dir,
        synthesizer=synthesize,
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        reference_names=references,
    )
    print(f"probes: 5 UTF-8 Turkish WAVs -> {args.output_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
