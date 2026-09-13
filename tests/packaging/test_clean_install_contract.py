from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_clean_install_harness_requires_isolated_explicit_inputs_and_evidence():
    text = (ROOT / "scripts" / "test_clean_install.ps1").read_text(encoding="utf-8")
    for parameter in ("Installer", "DataRoot", "FixtureManifestUrl", "FixtureManifestSha256", "ExpectedOutcome", "TimeoutSeconds"):
        assert f"${parameter}" in text
    for evidence in ("launcher-logs", "worker-logs", "inventory", "versions", "hashes", "ports", "process-cleanup"):
        assert evidence in text
    assert "/DIR=" in text
    assert "--fixture-workers" in text
    assert "/DATA_ROOT" not in text
    assert "finally" in text
    assert "unins*.exe" in text
    assert "--prepare-uninstall" not in text
    assert "/REMOVEOWNEDDATA" in text
    assert "YOUTUBER_FIXTURE_DATA_ROOT" in text
    assert "Get-FileHash" in text
    assert "versions = @()" not in text
    assert "hashes = @()" not in text
    assert "E:\\youtuber-clone" not in text


def test_fixture_manifest_is_bound_before_fixture_workers_start():
    text = (ROOT / "app" / "src" / "YouTuber.Launcher" / "Web" / "LauncherSessionPlanFactory.cs").read_text(encoding="utf-8")
    assert "parsed.Scheme != Uri.UriSchemeHttp" in text
    assert "!parsed.IsLoopback" in text
    assert "UseProxy = false" in text
    assert "AllowAutoRedirect = false" in text
    assert "LoadIntoBufferAsync(1024 * 1024).WaitAsync(cancellationToken)" in text
    assert "CryptographicOperations.FixedTimeEquals" in text
    assert "CreateFixturePlan(arguments, cancellationToken)" in text


def test_fixture_uninstaller_routes_owned_data_choice_through_real_uninstall_run():
    text = (ROOT / "packaging" / "windows" / "YouTuberStudio.iss").read_text(encoding="utf-8")
    assert "CmdLineParamExists('REMOVEOWNEDDATA')" in text
    assert "GetFixtureDataRoot" in text
    assert "YOUTUBER_FIXTURE_DATA_ROOT" in text
    assert "--data-root=" in text
    assert "--uninstall-choice-file --data-root=" in text
    assert "expanded while Setup records this UninstallRun entry" in text


def test_fixture_worker_must_be_packaged_beside_the_launcher():
    text = (ROOT / "app" / "src" / "YouTuber.Launcher" / "Web" / "LauncherSessionPlanFactory.cs").read_text(encoding="utf-8")
    assert 'Path.Combine(AppContext.BaseDirectory, "YouTuber.WorkerFixture.dll")' in text
    assert "Fixture worker payload is missing beside the launcher." in text
    assert '"app", "tests", "YouTuber.WorkerFixture"' not in text
    assert "new DirectoryInfo(AppContext.BaseDirectory)" not in text


def test_fixture_data_root_is_bound_when_setup_records_uninstall_run():
    harness = (ROOT / "scripts" / "test_clean_install.ps1").read_text(encoding="utf-8")
    assert harness.index("$env:YOUTUBER_FIXTURE_DATA_ROOT = $DataRoot") < harness.index("Start-Process -FilePath $Installer")
    assert harness.index("$env:YOUTUBER_FIXTURE_DATA_ROOT = $previousFixtureDataRoot") < harness.index("Start-Process -FilePath $uninstaller.FullName")


def test_clean_machine_runbook_lists_required_failure_matrix():
    text = (ROOT / "docs" / "runbooks" / "clean-machine-release-test.md").read_text(encoding="utf-8")
    for scenario in ("fresh install", "interrupted download", "corrupted component", "insufficient disk", "unsupported GPU", "missing WebView2", "update rollback", "app-only uninstall", "full owned-data uninstall"):
        assert scenario in text
