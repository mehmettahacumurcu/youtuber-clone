from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
PACKAGING = ROOT / "packaging" / "windows"
ISS = PACKAGING / "YouTuberStudio.iss"
BUILD = PACKAGING / "build-installer.ps1"
ASSETS = PACKAGING / "assets"
FIXTURE_CATALOG = ROOT / "tests" / "fixtures" / "distribution" / "bootstrap-production-fixture.json"


def _normalized(path: Path) -> str:
    return path.read_text("utf-8").replace("\r\n", "\n")


def test_inno_is_per_user_x64_and_uses_dynamic_dark_mode() -> None:
    source = _normalized(ISS)
    assert "PrivilegesRequired=lowest" in source
    assert "ArchitecturesAllowed=x64compatible" in source
    assert "ArchitecturesInstallIn64BitMode=x64compatible" in source
    assert r"DefaultDirName={localappdata}\Programs\YouTuberStudio" in source
    assert "WizardStyle=modern dynamic" in source
    assert "CloseApplications=yes" in source
    assert "AppMutex=YouTuberStudio.Launcher" in source
    assert "runascurrentuser" not in source.lower()


def test_installer_contains_only_launcher_not_downloadable_payloads() -> None:
    source = _normalized(ISS).lower()
    assert r'source: "{#publishdir}\*"' in source
    assert "recursesubdirs" in source and "createallsubdirs" in source
    for forbidden in ("*.gguf", "*.pth", "*.index", "training/", "data/rag", "python worker"):
        assert forbidden not in source


def test_shortcuts_postinstall_launch_and_safe_uninstall_hook_are_declared() -> None:
    source = _normalized(ISS)
    assert 'Name: "desktopicon"' in source and "Flags: unchecked" in source
    assert r'{group}\YouTuber Studio' in source
    assert r'{autodesktop}\YouTuber Studio' in source
    assert 'Filename: "{app}\\YouTuberStudio.exe"; Description: "Launch YouTuber Studio"' in source
    assert "Flags: nowait postinstall skipifsilent" in source
    assert "[UninstallRun]" in source
    assert "--prepare-uninstall" in source
    assert 'Parameters: "--prepare-uninstall --uninstall-choice-file"' in source
    assert "[UninstallDelete]" in source
    assert 'Name: "{app}\\uninstall-choice.txt"' in source
    app_source = _normalized(ROOT / "app" / "src" / "YouTuber.Launcher" / "App.xaml.cs")
    preparation = _normalized(ROOT / "app" / "src" / "YouTuber.Launcher" / "Uninstall" / "UninstallPreparation.cs")
    assert "UninstallPreparation.IsRequested(eventArgs.Args)" in app_source
    assert 'name: "YouTuberStudio.Launcher"' in app_source
    assert '"--prepare-uninstall"' in preparation


def test_versioned_output_and_all_disclosures_are_packaged() -> None:
    source = _normalized(ISS)
    assert "OutputBaseFilename=YouTuberStudio-Setup-{#AppVersion}" in source
    assert "SetupIconFile={#AssetsDir}\\youtuber.ico" in source
    required = {
        "LICENSE-code.txt", "LICENSE-data.txt", "AI-DISCLOSURE.txt",
        "THIRD-PARTY-NOTICES.txt", "README-SMARTSCREEN.txt", "youtuber.ico",
    }
    assert required <= {path.name for path in ASSETS.iterdir()}
    for name in required - {"youtuber.ico"}:
        assert f'{name}"' in source
        assert (ASSETS / name).stat().st_size > 100
    assert (ASSETS / "youtuber.ico").read_bytes()[:4] == b"\x00\x00\x01\x00"


def test_inno_source_has_no_absolute_developer_paths() -> None:
    source = _normalized(ISS)
    assert not re.search(r"(?i)[a-z]:\\", source)
    assert "E:\\youtuber-clone" not in source
    assert "C:\\Users\\Developer" not in source


def test_build_script_pins_toolchain_and_runs_all_release_gates() -> None:
    source = _normalized(BUILD)
    required_fragments = (
        '"7.0.2"', "Get-AuthenticodeSignature", "BootstrapCatalog",
        "environment", "production", "dotnet test", "dotnet publish",
        "SelfContained=true", "BootstrapCatalogPath", "ISCC.exe",
        "Get-FileHash", "SHA256", "installer-metadata.json", "FixtureMode",
    )
    for fragment in required_fragments:
        assert fragment in source
    assert "UsingDevelopmentBootstrapCatalog=false" in source
    assert "bootstrap-catalog.json" in source


def test_fixture_catalog_is_production_shaped_and_immutable() -> None:
    payload = json.loads(FIXTURE_CATALOG.read_text("utf-8"))
    assert payload["schema"] == "youtuber.bootstrap-catalog.v1"
    assert payload["environment"] == "production"
    assert payload["manifest_url"].startswith("https://huggingface.co/")
    assert re.fullmatch(r"[0-9a-f]{64}", payload["manifest_sha256"])
    assert set(payload["manifest_sha256"]) != {"0"}


@pytest.mark.parametrize(
    "manifest_url",
    [
        "https://huggingface.co/owner/repo/resolve/main/manifest.json",
        "https://huggingface.co/owner/repo/blob/1111111111111111111111111111111111111111/manifest.json",
        "https://huggingface.co/owner/repo/resolve/111111111111111111111111111111111111111/manifest.json",
        "https://user:secret@huggingface.co/owner/repo/resolve/1111111111111111111111111111111111111111/manifest.json",
        "https://huggingface.co:444/owner/repo/resolve/1111111111111111111111111111111111111111/manifest.json",
        "https://huggingface.co/owner/repo/resolve/1111111111111111111111111111111111111111/manifest.json?download=true",
        "https://huggingface.co/owner/repo/resolve/1111111111111111111111111111111111111111/manifest.json#fragment",
        "https://huggingface.co/owner/repo/resolve/1111111111111111111111111111111111111111/folder%2Fmanifest.json",
        "https://huggingface.co/owner/repo/resolve/1111111111111111111111111111111111111111/folder/../manifest.json",
    ],
)
def test_build_script_rejects_noncanonical_or_mutable_manifest_urls(tmp_path: Path, manifest_url: str) -> None:
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    assert powershell is not None, "PowerShell is required for Windows packaging validation"
    catalog = tmp_path / "bootstrap.json"
    catalog.write_text(
        json.dumps(
            {
                "schema": "youtuber.bootstrap-catalog.v1",
                "environment": "production",
                "manifest_url": manifest_url,
                "manifest_sha256": "a" * 64,
            }
        ),
        "utf-8",
    )

    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(BUILD),
            "-Version",
            "0.0.1-test",
            "-BootstrapCatalog",
            str(catalog),
            "-FixtureMode",
            "-ArtifactsRoot",
            str(tmp_path / "artifacts"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    assert result.returncode != 0
    assert "canonical immutable Hugging Face resolve URL" in result.stdout + result.stderr
