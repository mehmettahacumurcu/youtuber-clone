from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice.pipeline import PipelineBusyError, VoicePipeline


def _wav(path: Path, value: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.full(24_000, value, dtype=np.float32), 24_000, format="WAV")


class FakeXtts:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.unloads = 0

    def synthesize(self, text: str, output: Path) -> None:
        self.events.append(f"xtts:{text}")
        _wav(output)

    def unload(self) -> None:
        self.unloads += 1


class FakeRvc:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.unloads = 0

    def convert(self, source: Path, output: Path) -> None:
        self.events.append(f"rvc:{source.name}")
        shutil.copyfile(source, output)

    def unload(self) -> None:
        self.unloads += 1


def test_voice_pipeline_runs_xtts_then_rvc_and_promotes_only_valid_wavs(tmp_path: Path):
    events: list[str] = []
    xtts = FakeXtts(events)
    rvc = FakeRvc(events)

    result = VoicePipeline(xtts, rvc, tmp_path).synthesize("Merhaba.", "req-001")

    assert events == ["xtts:Merhaba.", "rvc:raw_xtts.wav"]
    assert result.final_path.name == "converted.wav"
    assert result.rvc_epoch == 200
    assert result.index_rate == 0.75
    assert result.sample_rate == 24_000
    assert result.final_path.is_file()
    assert not (tmp_path / "audio" / "req-001" / "raw_xtts.wav").exists()
    assert not list((tmp_path / "audio" / "req-001").glob("*.partial"))


def test_voice_pipeline_deletes_partials_preserves_prior_output_and_records_error(tmp_path: Path):
    class FailingRvc(FakeRvc):
        def convert(self, source: Path, output: Path) -> None:
            _wav(output)
            raise RuntimeError("conversion failed")

    old = tmp_path / "audio" / "req-002" / "converted.wav"
    _wav(old)
    original = old.read_bytes()
    pipeline = VoicePipeline(FakeXtts([]), FailingRvc([]), tmp_path)

    with pytest.raises(RuntimeError, match="conversion failed"):
        pipeline.synthesize("Merhaba.", "req-002")

    assert old.read_bytes() == original
    assert not (tmp_path / "audio" / "req-002" / "raw_xtts.wav").exists()
    assert pipeline.status == "error"
    assert not list(old.parent.glob("*.partial"))


def test_keep_raw_retry_restores_the_prior_raw_and_converted_pair_after_failure(tmp_path: Path):
    class FailingRvc(FakeRvc):
        def convert(self, source: Path, output: Path) -> None:
            _wav(output)
            raise RuntimeError("conversion failed")

    directory = tmp_path / "audio" / "req-restore"
    raw, converted = directory / "raw_xtts.wav", directory / "converted.wav"
    _wav(raw, 0.1)
    _wav(converted, 0.2)
    old_raw, old_converted = raw.read_bytes(), converted.read_bytes()

    with pytest.raises(RuntimeError, match="conversion failed"):
        VoicePipeline(FakeXtts([]), FailingRvc([]), tmp_path, keep_intermediate_audio=True).synthesize("Merhaba.", "req-restore")

    assert raw.read_bytes() == old_raw
    assert converted.read_bytes() == old_converted
    assert not list(directory.glob("*.partial"))
    assert not list(directory.glob("*.backup"))


def test_low_vram_pipeline_unloads_xtts_before_rvc_and_rvc_after_failure(tmp_path: Path):
    events: list[str] = []
    xtts = FakeXtts(events)
    rvc = FakeRvc(events)

    VoicePipeline(xtts, rvc, tmp_path, low_vram_voice=True).synthesize("Merhaba.", "req-003")

    assert xtts.unloads == 1
    assert rvc.unloads == 1


def test_pipeline_rejects_a_second_synthesis_while_the_first_is_running(tmp_path: Path):
    started = __import__("threading").Event()
    release = __import__("threading").Event()

    class BlockingXtts(FakeXtts):
        def synthesize(self, text: str, output: Path) -> None:
            started.set()
            assert release.wait(2)
            _wav(output)

    pipeline = VoicePipeline(BlockingXtts([]), FakeRvc([]), tmp_path)
    thread = __import__("threading").Thread(
        target=lambda: pipeline.synthesize("Birinci.", "req-one"), daemon=True
    )
    thread.start()
    assert started.wait(2)
    with pytest.raises(PipelineBusyError):
        pipeline.synthesize("İkinci.", "req-two")
    release.set()
    thread.join(2)


def test_pipeline_gate_is_shared_by_all_pipeline_instances(tmp_path: Path):
    started = __import__("threading").Event()
    release = __import__("threading").Event()

    class BlockingXtts(FakeXtts):
        def synthesize(self, text: str, output: Path) -> None:
            started.set()
            assert release.wait(2)
            _wav(output)

    first = VoicePipeline(BlockingXtts([]), FakeRvc([]), tmp_path / "first")
    second = VoicePipeline(FakeXtts([]), FakeRvc([]), tmp_path / "second")
    thread = __import__("threading").Thread(
        target=lambda: first.synthesize("Birinci.", "req-one"), daemon=True
    )
    thread.start()
    assert started.wait(2)
    with pytest.raises(PipelineBusyError):
        second.synthesize("İkinci.", "req-two")
    release.set()
    thread.join(2)


def test_pipeline_rejects_blank_text_before_it_can_reach_xtts(tmp_path: Path):
    events: list[str] = []

    with pytest.raises(ValueError, match="blank"):
        VoicePipeline(FakeXtts(events), FakeRvc(events), tmp_path).synthesize("   ", "req-blank")

    assert events == []
