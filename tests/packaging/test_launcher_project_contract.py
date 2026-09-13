from pathlib import Path


def test_launcher_is_self_contained_net8_win_x64():
    project = Path("app/src/YouTuber.Launcher/YouTuber.Launcher.csproj").read_text("utf-8")
    assert "<TargetFramework>net8.0-windows</TargetFramework>" in project
    assert "<RuntimeIdentifier>win-x64</RuntimeIdentifier>" in project
    assert "<SelfContained>true</SelfContained>" in project
    assert "<UseWPF>true</UseWPF>" in project


def test_webview2_version_is_centrally_pinned():
    packages = Path("app/Directory.Packages.props").read_text("utf-8")
    assert 'Include="Microsoft.Web.WebView2" Version="1.0.4078.44"' in packages


def test_production_app_composes_locator_ownership_recovery_and_ready_revalidation():
    source = Path("app/src/YouTuber.Launcher/App.xaml.cs").read_text("utf-8")
    locator_save = source.index("await locator.SaveAsync")
    ownership_bootstrap = source.index("await ProductionOwnershipBootstrap.InitializeAsync")
    setup_journal = source.index("new AtomicFileSetupJournal")

    assert "locator.TryLoad()" in source
    assert "window.ShowDataRootSelection(selection)" in source
    assert locator_save < ownership_bootstrap < setup_journal
    assert "await _setupActions.CreateCoordinatorAsync" in source
    assert "await runner.RevalidateReadyAsync" in source
