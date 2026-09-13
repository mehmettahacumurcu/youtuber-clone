from __future__ import annotations

import builtins
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from runtime.paths import RuntimePaths
from ui.voice_client import VoiceAudio, VoiceClient


class _Component:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def click(self, *args, **kwargs):
        return self

    def then(self, *args, **kwargs):
        return self

    def submit(self, *args, **kwargs):
        return self

    def change(self, *args, **kwargs):
        return self

    def load(self, *args, **kwargs):
        return self


def _fake_gradio() -> SimpleNamespace:
    return SimpleNamespace(
        Blocks=lambda *args, **kwargs: _Component(),
        Row=lambda *args, **kwargs: _Component(),
        Column=lambda *args, **kwargs: _Component(),
        Accordion=lambda *args, **kwargs: _Component(),
        HTML=lambda *args, **kwargs: _Component(),
        Chatbot=lambda *args, **kwargs: _Component(),
        Textbox=lambda *args, **kwargs: _Component(),
        Button=lambda *args, **kwargs: _Component(),
        Dropdown=lambda *args, **kwargs: _Component(),
        Slider=lambda *args, **kwargs: _Component(),
        Audio=lambda *args, **kwargs: _Component(),
        Markdown=lambda *args, **kwargs: _Component(),
        update=lambda **kwargs: kwargs,
        themes=SimpleNamespace(Base=lambda: object()),
    )


def _studio(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise RuntimeError("Studio imported torch eagerly")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "gradio", _fake_gradio())
    monkeypatch.delitem(sys.modules, "ui.speaker_studio", raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return importlib.import_module("ui.speaker_studio")


def _write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(24_000, dtype=np.float32), 24_000, subtype="PCM_16")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _production_manifest(paths: RuntimePaths) -> None:
    root = paths.voice_root
    xtts = root / "xtts"
    xtts.mkdir(parents=True)
    for name in ("best_model.pth", "config.json", "vocab.json"):
        (xtts / name).write_text(name, encoding="utf-8")
    reference = root / "references" / "approved.wav"
    _write_wav(reference)
    other = (
        "rvc/speaker_epoch_200.pth",
        "rvc/speaker.index",
        "rvc/models/embedders/contentvec/config.json",
        "rvc/models/embedders/contentvec/pytorch_model.bin",
        "rvc/models/predictors/rmvpe.pt",
    )
    files = []
    for path in (xtts / "best_model.pth", xtts / "config.json", xtts / "vocab.json", reference):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _hash(path),
            }
        )
    files.extend({"path": path, "size": 1, "sha256": "a" * 64} for path in other)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "youtuber.voice.v1",
                "runtime_api": 1,
                "sample_rate": 24_000,
                "audio_format": "WAV",
                "files": files,
                "xtts": {
                    "checkpoint": "xtts/best_model.pth",
                    "config": "xtts/config.json",
                    "vocab": "xtts/vocab.json",
                    "reference_wavs": ["references/approved.wav"],
                },
                "rvc": {
                    "weight": "rvc/speaker_epoch_200.pth",
                    "index": "rvc/speaker.index",
                    "epoch": 200,
                    "index_rate": 0.75,
                    "pitch": 0,
                    "f0_method": "rmvpe",
                    "protect": 0.5,
                },
                "contentvec": {
                    "config": "rvc/models/embedders/contentvec/config.json",
                    "model": "rvc/models/embedders/contentvec/pytorch_model.bin",
                },
                "rmvpe": "rvc/models/predictors/rmvpe.pt",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_studio_uses_the_isolated_voice_service_in_packaged_mode(monkeypatch, tmp_path: Path):
    studio = _studio(monkeypatch)
    paths = RuntimePaths.developer(tmp_path)
    _production_manifest(paths)

    options = studio.discover_tts(paths, packaged=True)

    assert list(options) == ["Speaker voice service"]


def test_studio_does_not_scan_or_load_developer_xtts_checkpoints(monkeypatch, tmp_path: Path):
    studio = _studio(monkeypatch)
    paths = RuntimePaths.developer(tmp_path)
    final = paths.xtts_assets_dir / "xtts_speaker_v2"
    final.mkdir(parents=True)
    for name in ("best_model.pth", "config.json", "vocab.json"):
        (final / name).write_text(name, encoding="utf-8")
    base = paths.xtts_assets_dir / "xtts_base"
    base.mkdir()
    (base / "model.pth").write_text("base", encoding="utf-8")
    _write_wav(paths.reference_wavs_dir / "legacy.wav")

    options = studio.discover_tts(paths, packaged=False)

    assert list(options) == ["Speaker voice service"]


def test_studio_sends_only_final_assistant_text_to_the_voice_service(monkeypatch):
    studio = _studio(monkeypatch)
    sent: list[tuple[str, str]] = []
    stream = __import__("io").BytesIO()
    sf.write(stream, np.zeros(24_000, dtype=np.float32), 24_000, format="WAV")

    class FakeVoiceClient:
        def synthesize(self, text, request_id):
            sent.append((text, request_id))
            return VoiceAudio(
                wav_bytes=stream.getvalue(), sample_rate=24_000, duration_seconds=1.0,
                sha256=hashlib.sha256(stream.getvalue()).hexdigest(), request_id=request_id,
            )

    monkeypatch.setattr(studio, "_voice_client", lambda: FakeVoiceClient(), raising=False)
    monkeypatch.setattr(studio, "TTS_OPTS", {"Speaker voice service": object()})
    audio = studio.read_last(
        [{"role": "assistant", "content": "Doğal yanıt."}],
        "Speaker voice service",
        "speaker-model",
    )

    assert audio[0] == 24_000
    assert sent and sent[0][0] == "Doğal yanıt."
    assert "citation" not in sent[0][0]


def test_studio_forwards_a_long_assistant_answer_without_truncation(monkeypatch):
    studio = _studio(monkeypatch)
    text = ("Speaker answer. " * 499) + "Speaker answer."
    sent: list[str] = []
    stream = __import__("io").BytesIO()
    sf.write(stream, np.zeros(24_000, dtype=np.float32), 24_000, format="WAV")
    data = stream.getvalue()

    class Response:
        status_code = 200
        headers = {
            "content-type": "audio/wav",
            "content-length": str(len(data)),
            "x-youtuber-sha256": hashlib.sha256(data).hexdigest(),
            "x-youtuber-sample-rate": "24000",
            "x-youtuber-duration-seconds": "1.000000",
            "x-youtuber-rvc-epoch": "200",
            "x-youtuber-index-rate": "0.75",
            "x-youtuber-request-id": "long-answer",
        }

        def iter_content(self, chunk_size):
            yield data

        def close(self):
            return None

    def post(*args, **kwargs):
        sent.append(kwargs["json"]["text"])
        return Response()

    monkeypatch.setattr(
        studio,
        "VoiceClient",
        lambda *_args, **_kwargs: VoiceClient(
            "http://127.0.0.1:7862", "s" * 64, post=post
        ),
    )
    monkeypatch.setattr(studio, "TTS_OPTS", {"Speaker voice service": object()})
    monkeypatch.setattr(studio.uuid, "uuid4", lambda: SimpleNamespace(hex="long-answer"))

    audio = studio.read_last(
        [{"role": "assistant", "content": text}],
        "Speaker voice service",
        "speaker-model",
    )

    assert len(text) > 5_000
    assert audio[0] == 24_000
    assert sent == [text]


def test_studio_passes_selected_answer_model_to_voice_handoff(monkeypatch):
    studio = _studio(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(studio, "RUNTIME_SETTINGS", SimpleNamespace(low_vram_voice=True))
    monkeypatch.setattr(studio, "speak", lambda text, label, answer_model: calls.append(answer_model) or (24_000, np.zeros(1)))

    studio.read_last([{"role": "assistant", "content": "Doğal yanıt."}], "voice", "answer-model")

    assert calls == ["answer-model"]


def test_studio_keeps_answer_model_resident_on_high_vram_voice(monkeypatch):
    studio = _studio(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(studio, "RUNTIME_SETTINGS", SimpleNamespace(low_vram_voice=False))
    monkeypatch.setattr(studio, "ensure_answer_model_absent", calls.append, raising=False)
    monkeypatch.setattr(studio, "speak", lambda text, label, *args: (24_000, np.zeros(1)))

    studio.read_last([{"role": "assistant", "content": "Doğal yanıt."}], "voice", "answer-model")

    assert calls == []


def test_studio_free_text_voice_request_uses_low_vram_answer_barrier(monkeypatch):
    studio = _studio(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(studio, "RUNTIME_SETTINGS", SimpleNamespace(low_vram_voice=True))
    monkeypatch.setattr(studio, "ensure_answer_model_absent", calls.append, raising=False)
    monkeypatch.setattr(studio, "TTS_OPTS", {"voice": object()})
    monkeypatch.setattr(studio, "_voice_client", lambda: (_ for _ in ()).throw(RuntimeError("stop")), raising=False)

    with __import__("pytest").raises(RuntimeError, match="stop"):
        studio.speak("Serbest metin.", "voice", "answer-model")

    assert calls == ["answer-model"]


@__import__("pytest").mark.parametrize("answer_model", [None, "", "   "])
def test_studio_low_vram_voice_rejects_missing_answer_model_before_voice_request(monkeypatch, answer_model):
    studio = _studio(monkeypatch)
    monkeypatch.setattr(studio, "RUNTIME_SETTINGS", SimpleNamespace(low_vram_voice=True))
    monkeypatch.setattr(studio, "TTS_OPTS", {"voice": object()})
    monkeypatch.setattr(studio, "_voice_client", lambda: (_ for _ in ()).throw(AssertionError("must not request voice")), raising=False)

    with __import__("pytest").raises(ValueError, match="answer model"):
        studio.speak("Doğal yanıt.", "voice", answer_model)
