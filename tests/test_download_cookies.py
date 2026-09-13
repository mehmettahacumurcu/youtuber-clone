"""Cookie support for yt-dlp (needed when YouTube bot-gates downloads, e.g. on Colab)."""
from pipeline.download.runner import _resolve_cookiefile, _ydl_opts


def test_resolve_cookiefile_none_when_nothing_exists(tmp_path, monkeypatch):
    # Run from an empty cwd so the relative default locations don't resolve to real files.
    monkeypatch.chdir(tmp_path)
    assert _resolve_cookiefile(None) is None


def test_resolve_cookiefile_uses_explicit_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cf = tmp_path / "my_cookies.txt"
    cf.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    assert _resolve_cookiefile(str(cf)) == str(cf)


def test_resolve_cookiefile_explicit_missing_falls_through_to_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _resolve_cookiefile(str(tmp_path / "does_not_exist.txt")) is None


def test_resolve_cookiefile_autodetects_default_location(tmp_path, monkeypatch):
    # An unset config (None) still finds a cookies.txt sitting in the cwd.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cookies.txt").write_text("# cookies\n", encoding="utf-8")
    assert _resolve_cookiefile(None) == "cookies.txt"


def test_ydl_opts_injects_cookiefile_when_configured(settings, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cf = tmp_path / "c.txt"
    cf.write_text("x", encoding="utf-8")
    settings.download.cookies_file = str(cf)
    opts = _ydl_opts(settings, no_playlist=True, out_dir=tmp_path)
    assert opts["cookiefile"] == str(cf)


def test_ydl_opts_omits_cookiefile_when_none(settings, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings.download.cookies_file = None
    opts = _ydl_opts(settings, no_playlist=True, out_dir=tmp_path)
    assert "cookiefile" not in opts


def test_ydl_opts_enables_ejs_remote_components(settings, tmp_path):
    # Required so yt-dlp can fetch + run YouTube's n-challenge JS solver.
    opts = _ydl_opts(settings, no_playlist=True, out_dir=tmp_path)
    assert opts["remote_components"] == ["ejs:github"]
