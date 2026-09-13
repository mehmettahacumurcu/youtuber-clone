from pipeline.manifest import save_manifest
from colab.run_corpus import main, select_pending


def test_select_pending_skips_videos_with_filter_output(tmp_path):
    fdir = tmp_path / "filter"
    fdir.mkdir()
    (fdir / "b.json").write_text("{}", encoding="utf-8")
    assert select_pending(["a", "b", "c"], fdir) == ["a", "c"]


def test_main_dry_run_executes_no_stages(settings):
    save_manifest(["vid1", "vid2"], settings, generated_at="t")
    cfg = settings.paths.data_dir.parent / "config.yaml"
    rc = main(["--config", str(cfg), "--dry-run", "--no-checkpoint"])
    assert rc == 0
    assert not list(settings.paths.transcribe_dir.glob("*.json"))
    assert not list(settings.paths.filter_dir.glob("*.json"))


class _Result:
    def __init__(self, returncode):
        self.returncode = returncode


def test_main_continues_after_failure_and_returns_nonzero(settings, monkeypatch):
    import colab.run_corpus as rc
    save_manifest(["vid1", "vid2"], settings, generated_at="t")
    cfg = settings.paths.data_dir.parent / "config.yaml"
    attempted = []

    def fake_run(cmd, *a, **k):
        attempted.append((cmd[2], cmd[-1]))  # (module, video_id)
        if cmd[-1] == "vid1" and cmd[2] == "pipeline.download":
            return _Result(1)
        return _Result(0)

    monkeypatch.setattr(rc.subprocess, "run", fake_run)
    code = rc.main(["--config", str(cfg), "--no-checkpoint"])
    assert code == 1                                    # at least one failure
    assert any(vid == "vid2" for _, vid in attempted)   # continued past the failed vid1
    assert [s for s, vid in attempted if vid == "vid1"] == ["pipeline.download"]  # vid1 stopped at first failed stage


def test_main_max_minutes_zero_processes_nothing(settings, monkeypatch):
    import colab.run_corpus as rc
    save_manifest(["vid1"], settings, generated_at="t")
    cfg = settings.paths.data_dir.parent / "config.yaml"
    calls = []
    monkeypatch.setattr(rc, "run_video", lambda *a, **k: calls.append(a) or True)
    code = rc.main(["--config", str(cfg), "--no-checkpoint", "--max-minutes", "0"])
    assert code == 0
    assert calls == []  # budget exhausted before the first video
