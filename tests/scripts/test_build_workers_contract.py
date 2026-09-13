from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_workers.ps1"


def _script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _create_windows_junction(link: Path, target: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows junction regression")
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            "Windows junction creation unexpectedly failed: "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    assert link.exists()


def _run_clean_against_junction(
    tmp_path: Path, *, nested: bool
) -> tuple[subprocess.CompletedProcess, Path]:
    repository = tmp_path / ("nested-case" if nested else "root-case")
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    copied_script = scripts / SCRIPT.name
    shutil.copy2(SCRIPT, copied_script)
    external = tmp_path / ("nested-victim" if nested else "root-victim")
    external.mkdir()
    victim = external / "must-survive.txt"
    victim.write_text("safe", encoding="utf-8")
    build_root = repository / "build" / "pyinstaller"
    build_root.parent.mkdir(parents=True)
    if nested:
        build_root.mkdir()
        junction = build_root / "escape"
    else:
        junction = build_root
    _create_windows_junction(junction, external)

    completed = subprocess.run(
        [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(copied_script),
            "-Clean",
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if junction.exists():
        os.rmdir(junction)
    return completed, victim


def test_build_script_uses_only_the_two_frozen_worker_environments():
    text = _script_text()

    assert '$StudioEnvironment = Join-Path $RepositoryRoot ".venv-build-studio"' in text
    assert '$VoiceEnvironment = Join-Path $RepositoryRoot ".venv-build-voice"' in text
    assert text.count("uv sync --project") == 2
    assert text.count("--frozen") >= 2
    assert '$env:PYTHONNOUSERSITE = "1"' in text
    scrubbed = text.replace(".venv-build-studio", "").replace(".venv-build-voice", "")
    assert not re.search(r"(?i)(?:^|[^\w-])\.venv(?:-tts)?(?:[^\w-]|$)", scrubbed)


def test_build_script_runs_both_onedir_specs_and_verifies_outputs():
    text = _script_text()

    assert "packaging/pyinstaller/studio-worker.spec" in text.replace("\\", "/")
    assert "packaging/pyinstaller/voice-worker.spec" in text.replace("\\", "/")
    assert text.count("-m scripts.write_component_manifest") == 2
    assert "verify_component_tree" in text
    assert text.count("--zip") == 2
    assert "scripts/fetch_applio_inference.py" in text.replace("\\", "/")
    assert "uv sync --project" in text
    assert text.count("Remove-ForbiddenBundleDebris") >= 3
    assert text.index("Remove-ForbiddenBundleDebris $VoiceBundle") < text.index("-m scripts.write_component_manifest")
    assert "[switch]$PackageOnly" in text
    assert '-Include "*.ipynb"' not in text
    assert '$_.Extension.ToLowerInvariant() -in @(".ipynb", ".log", ".pyc")' in text


def test_build_script_can_reseal_only_studio_after_a_studio_runtime_fix():
    """Break caught: a Studio-only integration fix needlessly rebuilds the 5.4GB Voice tree."""
    text = _script_text()

    assert "[switch]$StudioOnly" in text
    assert "if (-not $StudioOnly)" in text
    assert "$Bundles = @($StudioBundle)" in text
    assert "$Bundles += $VoiceBundle" in text


def test_build_script_cleans_only_containment_checked_generated_roots():
    text = _script_text()

    assert "Assert-ContainedGeneratedPath" in text
    assert "Remove-Item -LiteralPath" in text
    assert "build/pyinstaller" in text.replace("\\", "/")
    assert "dist/workers" in text.replace("\\", "/")


@pytest.mark.parametrize("nested", [False, True], ids=["root-junction", "nested-junction"])
def test_build_script_rejects_junctions_before_cleanup_and_preserves_external_victim(
    tmp_path, nested
):
    completed, victim = _run_clean_against_junction(tmp_path, nested=nested)

    assert completed.returncode != 0
    assert "reparse" in (completed.stdout + completed.stderr).casefold()
    assert victim.read_text(encoding="utf-8") == "safe"


def test_build_script_checks_reparse_points_before_recursive_enumeration_or_deletion():
    text = _script_text()

    assert "function Assert-NoReparsePoints" in text
    assert "FileAttributes]::ReparsePoint" in text
    assert "Assert-NoReparsePoints $Path" in text
    assert "Assert-NoReparsePoints $Bundle" in text
    cleanup_body = text[
        text.index("function Remove-GeneratedDirectory"):
        text.index("function Remove-ForbiddenBundleDebris")
    ]
    assert cleanup_body.index("Assert-NoReparsePoints $Path") < cleanup_body.index(
        "Remove-Item -LiteralPath"
    )
    debris_body = text[
        text.index("function Remove-ForbiddenBundleDebris"):
        text.index("Set-Location $RepositoryRoot")
    ]
    assert debris_body.index("Assert-NoReparsePoints $Bundle") < debris_body.index("Get-ChildItem")


def test_packaging_sources_contain_no_developer_paths_or_legacy_launcher():
    paths = [
        SCRIPT,
        ROOT / "packaging" / "pyinstaller" / "studio-worker.spec",
        ROOT / "packaging" / "pyinstaller" / "voice-worker.spec",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "E:\\youtuber-clone" not in combined
    assert "START_SPEAKER_PAYLAS.bat" not in combined
    assert "share=True" not in combined
    assert "0.0.0.0" not in combined


def test_specs_are_onedir_and_do_not_collect_forbidden_repository_payloads():
    for name in ("studio-worker.spec", "voice-worker.spec"):
        text = (ROOT / "packaging" / "pyinstaller" / name).read_text(encoding="utf-8")
        assert "COLLECT(" in text
        assert "onefile" not in text.casefold()
        assert "data/" not in text.replace("\\", "/")
        assert "models/" not in text.replace("\\", "/")
        assert "tests/" not in text.replace("\\", "/")


def test_tts_hook_materializes_dynamically_scanned_vocoder_configs():
    text = (ROOT / "packaging" / "pyinstaller" / "hooks" / "hook-TTS.py").read_text(
        encoding="utf-8"
    )

    assert 'collect_data_files("TTS", include_py_files=True)' in text
    assert 'collect_data_files("TTS", include_py_files=False)' not in text
    assert 'collect_data_files("TTS.vocoder.configs", include_py_files=True)' in text
    assert 'collect_submodules("TTS.vocoder.configs")' in text
    assert 'collect_data_files("inflect", include_py_files=True)' in text
    assert 'collect_data_files("gruut", include_py_files=False)' in text


def test_datasets_hook_materializes_packaged_module_sources_for_runtime_hashing():
    hook = ROOT / "packaging" / "pyinstaller" / "hooks" / "hook-datasets.py"
    text = hook.read_text(encoding="utf-8")

    assert 'collect_data_files("datasets.packaged_modules", include_py_files=True)' in text


def test_flagembedding_hook_materializes_sources_for_import_time_doc_decorators():
    hook = ROOT / "packaging" / "pyinstaller" / "hooks" / "hook-FlagEmbedding.py"
    text = hook.read_text(encoding="utf-8")

    assert 'collect_data_files("FlagEmbedding", include_py_files=True)' in text
    assert 'collect_data_files("FlagEmbedding", include_py_files=False)' not in text


def test_studio_spec_collects_runtime_version_files_read_by_gradio_dependencies():
    text = (ROOT / "packaging" / "pyinstaller" / "studio-worker.spec").read_text(
        encoding="utf-8"
    )

    assert 'collect_data_files("safehttpx")' in text
    assert 'collect_data_files("groovy")' in text


def test_studio_spec_materializes_gradio_sources_used_for_component_metadata():
    text = (ROOT / "packaging" / "pyinstaller" / "studio-worker.spec").read_text(
        encoding="utf-8"
    )

    assert 'collect_data_files("gradio", include_py_files=True)' in text


def test_studio_spec_contains_the_runtime_config_expected_under_its_install_root():
    text = (ROOT / "packaging" / "pyinstaller" / "studio-worker.spec").read_text(
        encoding="utf-8"
    )

    assert '(str(repo / "config.yaml"), "runtime")' in text


def test_studio_spec_includes_transformers_module_scanned_by_auto_tokenizer():
    """Break caught: frozen AutoTokenizer scans the mapping and imports metaclip_2."""
    text = (ROOT / "packaging" / "pyinstaller" / "studio-worker.spec").read_text(
        encoding="utf-8"
    )

    assert '"transformers.models.metaclip_2"' in text


def test_voice_spec_collects_pinned_applio_dynamic_import_dependencies():
    text = (ROOT / "packaging" / "pyinstaller" / "voice-worker.spec").read_text(
        encoding="utf-8"
    )

    for module in ("noisereduce", "pedalboard", "torchcrepe", "torchfcpe", "wget"):
        assert f'"{module}"' in text
