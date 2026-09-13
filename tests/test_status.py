import json

from pipeline.manifest import save_manifest
from pipeline.status import compute_status


def _write(dir_, vid, payload=None):
    (dir_ / f"{vid}.json").write_text(json.dumps(payload or {}), encoding="utf-8")


def test_corpus_from_manifest(settings):
    save_manifest(["a", "b", "c", "d"], settings, generated_at="t")
    _write(settings.paths.diarize_dir, "a")
    _write(settings.paths.diarize_dir, "b")
    _write(settings.paths.identify_dir, "a")
    st = compute_status(settings)
    assert st["corpus"] == 4
    assert st["manifest_known"] is True
    assert st["stages"]["diarize"] == 2
    assert st["stages"]["identify"] == 1
    assert st["pending"]["diarize"] == ["c", "d"]


def test_corpus_falls_back_to_outputs_without_manifest(settings):
    _write(settings.paths.transcribe_dir, "x")
    _write(settings.paths.transcribe_dir, "y")
    st = compute_status(settings)
    assert st["corpus"] == 2
    assert st["manifest_known"] is False
    assert st["stages"]["transcribe"] == 2


def test_status_includes_filter_stage(settings):
    save_manifest(["a", "b"], settings, generated_at="t")
    _write(settings.paths.transcribe_dir, "a")
    _write(settings.paths.filter_dir, "a")
    st = compute_status(settings)
    assert st["stages"]["filter"] == 1
    assert st["pending"]["filter"] == ["b"]
