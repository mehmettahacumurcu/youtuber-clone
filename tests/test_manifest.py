from pipeline.manifest import load_manifest, pending_ids, save_manifest


def test_save_then_load_roundtrip(settings):
    save_manifest(["a", "b", "c"], settings, generated_at="2026-06-15T00:00:00+00:00")
    assert load_manifest(settings) == ["a", "b", "c"]


def test_save_dedups_preserving_order(settings):
    save_manifest(["a", "b", "a", "c", "b"], settings, generated_at="t")
    assert load_manifest(settings) == ["a", "b", "c"]


def test_load_missing_returns_empty(settings):
    assert load_manifest(settings) == []


def test_pending_ids_excludes_completed(settings):
    save_manifest(["a", "b", "c"], settings, generated_at="t")
    (settings.paths.transcribe_dir / "b.json").write_text("{}", encoding="utf-8")
    assert pending_ids(settings, settings.paths.transcribe_dir) == ["a", "c"]
