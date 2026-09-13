import os
import subprocess

import pytest

from distribution.hashing import ComponentVerificationError, sha256_file, verify_component_tree
from distribution.models import ComponentManifest


def _create_windows_junction(link, target):
    if os.name != "nt":
        pytest.skip("Windows junction regression")
    completed = subprocess.run(
        [
            "cmd.exe",
            "/d",
            "/c",
            "mklink",
            "/J",
            str(link),
            str(target),
        ],
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


def manifest_for(path, content):
    return ComponentManifest.model_validate({
        "schema": "youtuber.component.v1", "component": "voice_runtime", "version": "1.0.0",
        "compatibility": {"runtime_api": 1}, "entrypoint": path, "healthcheck": ["--healthcheck"],
        "files": [{"path": path, "size": len(content), "sha256": sha256_file_from_bytes(content)}],
    })


def sha256_file_from_bytes(content):
    import hashlib
    return hashlib.sha256(content).hexdigest()


def test_sha256_file_is_streaming_and_stable(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"youtuber")
    assert sha256_file(path) == "29dc5cd5439de3b6693ebc0370c22d62de838eabdcb03bc43789d65db8bdbeae"


def test_verify_component_tree_accepts_exact_hashed_tree(tmp_path):
    payload = b"youtuber"
    (tmp_path / "voice-worker.exe").write_bytes(payload)
    manifest = manifest_for("voice-worker.exe", payload)
    (tmp_path / "component-manifest.json").write_text(
        manifest.model_dump_json(by_alias=True), encoding="utf-8"
    )
    verify_component_tree(tmp_path, manifest)


def test_verify_component_tree_rejects_changed_or_unlisted_files(tmp_path):
    (tmp_path / "voice-worker.exe").write_bytes(b"changed")
    (tmp_path / "unexpected.txt").write_text("no", encoding="utf-8")
    manifest = manifest_for("voice-worker.exe", b"youtuber")
    (tmp_path / "component-manifest.json").write_text(
        manifest.model_dump_json(by_alias=True), encoding="utf-8"
    )
    with pytest.raises(ComponentVerificationError):
        verify_component_tree(tmp_path, manifest)


def test_verify_component_tree_rejects_junction_root(tmp_path):
    target = tmp_path / "outside-component"
    target.mkdir()
    payload = b"youtuber"
    (target / "voice-worker.exe").write_bytes(payload)
    manifest = manifest_for("voice-worker.exe", payload)
    (target / "component-manifest.json").write_text(
        manifest.model_dump_json(by_alias=True), encoding="utf-8"
    )
    junction = tmp_path / "component-junction"
    _create_windows_junction(junction, target)

    try:
        with pytest.raises(ComponentVerificationError, match="reparse"):
            verify_component_tree(junction, manifest)
    finally:
        if junction.exists():
            os.rmdir(junction)


def test_verify_component_tree_rejects_nested_junction_without_hashing_target(
    tmp_path, monkeypatch
):
    bundle = tmp_path / "voice-worker"
    bundle.mkdir()
    payload = b"youtuber"
    (bundle / "voice-worker.exe").write_bytes(payload)
    manifest = manifest_for("voice-worker.exe", payload)
    (bundle / "component-manifest.json").write_text(
        manifest.model_dump_json(by_alias=True), encoding="utf-8"
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    victim = outside / "victim.bin"
    victim.write_bytes(b"must never be hashed")
    junction = bundle / "escape"
    _create_windows_junction(junction, outside)

    hashed = []
    original_sha256_file = sha256_file

    def recording_sha256_file(path):
        hashed.append(path)
        return original_sha256_file(path)

    monkeypatch.setattr("distribution.hashing.sha256_file", recording_sha256_file)
    try:
        with pytest.raises(ComponentVerificationError, match="reparse"):
            verify_component_tree(bundle, manifest)
        assert victim not in hashed
        assert victim.read_bytes() == b"must never be hashed"
    finally:
        if junction.exists():
            os.rmdir(junction)
