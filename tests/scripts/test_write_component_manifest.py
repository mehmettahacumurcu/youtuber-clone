from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

from distribution.hashing import ComponentVerificationError, verify_component_tree
from scripts.write_component_manifest import (
    build_component_manifest,
    write_component_manifest,
    write_deterministic_zip,
)


def _fake_bundle(root: Path) -> None:
    (root / "nested").mkdir(parents=True)
    (root / "studio-worker.exe").write_bytes(b"worker")
    (root / "nested" / "z.txt").write_text("z", encoding="utf-8")
    (root / "nested" / "a.txt").write_text("Turkce: ğüşıöç", encoding="utf-8")


def test_manifest_is_sorted_hashed_stable_and_excludes_itself(tmp_path):
    bundle = tmp_path / "studio-worker"
    _fake_bundle(bundle)

    first = write_component_manifest(
        bundle,
        component="studio_runtime",
        version="1.0.0",
        entrypoint="studio-worker.exe",
    )
    first_bytes = (bundle / "component-manifest.json").read_bytes()
    second = write_component_manifest(
        bundle,
        component="studio_runtime",
        version="1.0.0",
        entrypoint="studio-worker.exe",
    )

    assert first == second
    assert first_bytes == (bundle / "component-manifest.json").read_bytes()
    assert first_bytes.endswith(b"\n")
    assert [entry.path for entry in first.files] == [
        "nested/a.txt", "nested/z.txt", "studio-worker.exe",
    ]
    assert all(entry.path != "component-manifest.json" for entry in first.files)
    for entry in first.files:
        payload = (bundle / entry.path).read_bytes()
        assert entry.size == len(payload)
        assert entry.sha256 == hashlib.sha256(payload).hexdigest()
    assert json.loads(first_bytes)["schema"] == "youtuber.component.v1"
    verify_component_tree(bundle, first)


def test_manifest_rejects_forbidden_environment_file(tmp_path):
    bundle = tmp_path / "voice-worker"
    bundle.mkdir()
    (bundle / "voice-worker.exe").write_bytes(b"worker")
    (bundle / ".env").write_text("SECRET=leak", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden"):
        build_component_manifest(
            bundle,
            component="voice_runtime",
            version="1.0.0",
            entrypoint="voice-worker.exe",
        )


@pytest.mark.parametrize("relative", ["package/tests/case.bin", "torch/testing/helper.py"])
def test_manifest_rejects_bundled_test_payloads(tmp_path, relative):
    bundle = tmp_path / "studio-worker"
    bundle.mkdir()
    (bundle / "studio-worker.exe").write_bytes(b"worker")
    target = bundle / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not runtime data")

    with pytest.raises(ValueError, match="forbidden"):
        build_component_manifest(
            bundle,
            component="studio_runtime",
            version="1.0.0",
            entrypoint="studio-worker.exe",
        )


def test_manifest_rejects_zero_byte_payload(tmp_path):
    bundle = tmp_path / "studio-worker"
    bundle.mkdir()
    (bundle / "studio-worker.exe").write_bytes(b"worker")
    (bundle / "marker.txt").write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        build_component_manifest(
            bundle,
            component="studio_runtime",
            version="1.0.0",
            entrypoint="studio-worker.exe",
        )


def test_manifest_rejects_symlink_or_reparse_entry(tmp_path):
    bundle = tmp_path / "voice-worker"
    bundle.mkdir()
    (bundle / "voice-worker.exe").write_bytes(b"worker")
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    link = bundle / "escape.txt"
    try:
        os.symlink(target, link)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(ValueError, match="symlink|reparse"):
        build_component_manifest(
            bundle,
            component="voice_runtime",
            version="1.0.0",
            entrypoint="voice-worker.exe",
        )


def test_component_verifier_rejects_missing_extra_and_tampered_payloads(tmp_path):
    bundle = tmp_path / "studio-worker"
    _fake_bundle(bundle)
    manifest = write_component_manifest(
        bundle,
        component="studio_runtime",
        version="1.0.0",
        entrypoint="studio-worker.exe",
    )

    (bundle / "nested" / "z.txt").unlink()
    with pytest.raises(ComponentVerificationError):
        verify_component_tree(bundle, manifest)
    (bundle / "nested" / "z.txt").write_text("z", encoding="utf-8")

    (bundle / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ComponentVerificationError):
        verify_component_tree(bundle, manifest)
    (bundle / "extra.txt").unlink()

    (bundle / "nested" / "a.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ComponentVerificationError):
        verify_component_tree(bundle, manifest)


def test_deterministic_zip_has_stable_root_order_metadata_and_bytes(tmp_path):
    bundle = tmp_path / "stüdio worker"
    _fake_bundle(bundle)
    write_component_manifest(
        bundle,
        component="studio_runtime",
        version="1.0.0",
        entrypoint="studio-worker.exe",
    )
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    write_deterministic_zip(bundle, first)
    write_deterministic_zip(bundle, second)

    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        infos = archive.infolist()
        assert [info.filename for info in infos] == sorted(info.filename for info in infos)
        assert all(info.filename.startswith("stüdio worker/") for info in infos)
        assert all(info.date_time == (2026, 1, 1, 0, 0, 0) for info in infos)
        assert all("\\" not in info.filename for info in infos)
