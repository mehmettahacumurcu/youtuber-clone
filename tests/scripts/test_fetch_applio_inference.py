"""Behavior tests for the pinned, inference-only Applio source vendor."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import scripts.fetch_applio_inference as applio_vendor
from scripts.fetch_applio_inference import (
    APPLIO_COMMIT,
    APPLIO_REPOSITORY,
    AUDITED_LIB_PATHS,
    REQUIRED_VOICE_ARTIFACT_FILES,
    VendorLock,
    _refresh_lock,
    stage_vendor_tree,
    verify_vendor_tree,
)


LOCK_PATH = Path("third_party/applio.lock.json")
TRACE_PATH = Path("third_party/applio.inference-trace.json")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _complete_contents(overrides: dict[str, bytes] | None = None) -> dict[str, bytes]:
    contents = {
        "rvc/configs/40000.json": b"{}",
        "rvc/infer/infer.py": b"pass\n",
        **{path: b"pass\n" for path in AUDITED_LIB_PATHS},
    }
    contents.update(overrides or {})
    return contents


@pytest.fixture
def vendor_lock() -> VendorLock:
    contents = _complete_contents()
    return VendorLock.model_validate(
        {
            "schema": "youtuber.applio.vendor.v1",
            "repository": APPLIO_REPOSITORY,
            "commit": APPLIO_COMMIT,
            "files": {
                path: {"size": len(content), "sha256": _sha256(content)}
                for path, content in sorted(contents.items())
            },
            "auxiliary_assets": {
                path: {"supplied_by": "voice_artifact"}
                for path in REQUIRED_VOICE_ARTIFACT_FILES
            },
        }
    )


def test_vendor_lock_pins_the_audited_applio_commit_and_round_trips():
    lock = VendorLock.load(LOCK_PATH)

    assert lock.repository == "https://github.com/IAHispano/Applio.git"
    assert lock.commit == "3e5e248c2bd003f65f3e128dd12677533def27be"
    assert "rvc/infer/infer.py" in lock.files
    assert lock.model_dump(by_alias=True)["schema"] == "youtuber.applio.vendor.v1"
    assert VendorLock.model_validate_json(lock.model_dump_json()) == lock


def test_committed_inference_trace_records_a_complete_declared_only_conversion():
    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    lock = VendorLock.load(LOCK_PATH)

    assert trace["schema"] == "youtuber.applio.inference-trace.v1"
    assert trace["repository"] == APPLIO_REPOSITORY
    assert trace["commit"] == APPLIO_COMMIT
    assert trace["conversion"]["nonempty"] is True
    assert trace["conversion"]["sample_rate"] == 32_000
    assert trace["conversion"]["channels"] == 1
    assert trace["conversion"]["frames"] > 0
    assert trace["audit"]["undeclared_applio_source_or_config_opened"] == []
    assert trace["audit"]["pretrained_paths_opened"] == []
    assert set(trace["audit"]["applio_source_or_config_opened"]).issubset(lock.files)
    assert trace["audit"]["native_index_paths_used"] == [
        "voice-artifact/speaker_rvc_full_v1.index"
    ]


def test_vendor_lock_declares_only_voice_supplied_contentvec_and_rmvpe_assets():
    lock = VendorLock.load(LOCK_PATH)

    assert tuple(lock.auxiliary_assets) == REQUIRED_VOICE_ARTIFACT_FILES
    assert not any("rvc/models/pretraineds" in path for path in lock.files)
    assert not any("rvc/models/pretraineds" in path for path in lock.auxiliary_assets)


def test_vendor_tree_accepts_an_exact_hashed_tree(tmp_path, vendor_lock):
    for relative, content in _complete_contents().items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    verify_vendor_tree(tmp_path, vendor_lock)


def test_vendor_tree_rejects_an_unexpected_source_file(tmp_path, vendor_lock):
    unexpected = tmp_path / "rvc" / "unexpected.py"
    unexpected.parent.mkdir()
    unexpected.write_text("bad", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected"):
        verify_vendor_tree(tmp_path, vendor_lock)


def test_vendor_tree_rejects_a_modified_declared_file(tmp_path, vendor_lock):
    for relative, content in _complete_contents().items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    target = tmp_path / "rvc" / "infer" / "infer.py"
    target.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="modified"):
        verify_vendor_tree(tmp_path, vendor_lock)


def test_vendor_tree_rejects_pretrained_assets_even_if_they_are_not_locked(tmp_path, vendor_lock):
    target = tmp_path / "rvc" / "models" / "pretraineds" / "hifi-gan" / "f0G40k.pth"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not allowed")

    with pytest.raises(ValueError, match="unexpected|pretrained"):
        verify_vendor_tree(tmp_path, vendor_lock)


def test_vendor_lock_rejects_case_collisions_and_unsafe_paths():
    payload = {
        "schema": "youtuber.applio.vendor.v1",
        "repository": APPLIO_REPOSITORY,
        "commit": APPLIO_COMMIT,
        "files": {
            "rvc/INFER/INFER.py": {"size": 1, "sha256": "0" * 64},
            "rvc/infer/infer.py": {"size": 1, "sha256": "0" * 64},
        },
        "auxiliary_assets": {
            path: {"supplied_by": "voice_artifact"}
            for path in REQUIRED_VOICE_ARTIFACT_FILES
        },
    }

    with pytest.raises(ValidationError, match="duplicate Windows paths"):
        VendorLock.model_validate(payload)

    payload["files"] = {"../rvc/infer/infer.py": {"size": 1, "sha256": "0" * 64}}
    with pytest.raises(ValidationError, match="safe relative"):
        VendorLock.model_validate(payload)


def test_vendor_lock_json_is_deterministic():
    lock = VendorLock.load(LOCK_PATH)

    parsed = json.loads(lock.model_dump_json())
    assert list(parsed["files"]) == sorted(parsed["files"])
    assert list(parsed["auxiliary_assets"]) == list(REQUIRED_VOICE_ARTIFACT_FILES)


def test_vendor_lock_rejects_a_missing_audited_runtime_lib_file():
    payload = VendorLock.load(LOCK_PATH).model_dump(by_alias=True)
    payload["files"].pop("rvc/lib/utils.py")

    with pytest.raises(ValidationError, match="exact audited rvc/lib closure"):
        VendorLock.model_validate(payload)


def test_refresh_lock_reports_the_exact_pin_and_a_deterministic_no_change_diff(
    tmp_path, vendor_lock, capsys
):
    source = tmp_path / "source"
    for relative, content in _complete_contents().items():
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    lock_path = tmp_path / "applio.lock.json"
    lock_path.write_text(vendor_lock.json_text(), encoding="utf-8")

    _refresh_lock(lock_path, source, vendor_lock)

    output = capsys.readouterr().out
    assert f"repository={APPLIO_REPOSITORY} commit={APPLIO_COMMIT}" in output
    assert "No hash changes." in output
    assert lock_path.read_text(encoding="utf-8") == vendor_lock.json_text()


def test_refresh_lock_prints_a_deterministic_old_new_hash_diff(tmp_path, vendor_lock, capsys):
    source = tmp_path / "source"
    for relative, content in _complete_contents({"rvc/infer/infer.py": b"changed\n"}).items():
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    lock_path = tmp_path / "applio.lock.json"
    lock_path.write_text(vendor_lock.json_text(), encoding="utf-8")

    _refresh_lock(lock_path, source, vendor_lock)

    output = capsys.readouterr().out
    assert "--- old/applio.lock.json" in output
    assert "+++ new/applio.lock.json" in output
    assert VendorLock.load(lock_path).files["rvc/infer/infer.py"].sha256 == _sha256(b"changed\n")


def _write_vendor_tree(root: Path, lock: VendorLock, contents: dict[str, bytes]) -> None:
    for relative in lock.files:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents[relative])


def _lock_for_contents(contents: dict[str, bytes]) -> VendorLock:
    contents = _complete_contents(contents)
    return VendorLock.model_validate(
        {
            "schema": "youtuber.applio.vendor.v1",
            "repository": APPLIO_REPOSITORY,
            "commit": APPLIO_COMMIT,
            "files": {
                path: {"size": len(content), "sha256": _sha256(content)}
                for path, content in sorted(contents.items())
            },
            "auxiliary_assets": {
                path: {"supplied_by": "voice_artifact"}
                for path in REQUIRED_VOICE_ARTIFACT_FILES
            },
        }
    )


def test_stage_vendor_tree_replaces_an_existing_verified_destination(tmp_path):
    old_contents = {
        "rvc/configs/40000.json": b"old-config",
        "rvc/infer/infer.py": b"old-infer",
        "rvc/lib/utils.py": b"old-utils",
    }
    new_contents = {
        "rvc/configs/40000.json": b"new-config",
        "rvc/infer/infer.py": b"new-infer",
        "rvc/lib/utils.py": b"new-utils",
    }
    source = tmp_path / "source"
    destination = tmp_path / "vendor" / "applio"
    new_lock = _lock_for_contents(new_contents)
    old_lock = _lock_for_contents(old_contents)
    _write_vendor_tree(source, new_lock, _complete_contents(new_contents))
    _write_vendor_tree(destination, old_lock, _complete_contents(old_contents))

    stage_vendor_tree(source, destination, new_lock)

    verify_vendor_tree(destination, new_lock)
    assert (destination / "rvc/infer/infer.py").read_bytes() == b"new-infer"


def test_stage_vendor_tree_rolls_back_an_existing_destination_when_promotion_fails(
    tmp_path, monkeypatch
):
    old_contents = {
        "rvc/configs/40000.json": b"old-config",
        "rvc/infer/infer.py": b"old-infer",
        "rvc/lib/utils.py": b"old-utils",
    }
    new_contents = {
        "rvc/configs/40000.json": b"new-config",
        "rvc/infer/infer.py": b"new-infer",
        "rvc/lib/utils.py": b"new-utils",
    }
    old_lock = _lock_for_contents(old_contents)
    new_lock = _lock_for_contents(new_contents)
    source = tmp_path / "source"
    destination = tmp_path / "vendor" / "applio"
    _write_vendor_tree(source, new_lock, _complete_contents(new_contents))
    _write_vendor_tree(destination, old_lock, _complete_contents(old_contents))
    before = {path: (destination / path).read_bytes() for path in old_lock.files}
    real_replace = applio_vendor.os.replace

    def fail_new_stage(source_path, destination_path):
        if Path(destination_path) == destination and Path(source_path).name == "applio":
            raise OSError("injected promotion failure")
        return real_replace(source_path, destination_path)

    monkeypatch.setattr(applio_vendor.os, "replace", fail_new_stage)

    with pytest.raises(OSError, match="injected promotion failure"):
        stage_vendor_tree(source, destination, new_lock)

    verify_vendor_tree(destination, old_lock)
    assert {path: (destination / path).read_bytes() for path in old_lock.files} == before
    assert not list(destination.parent.glob(".applio-stage-*"))
    assert not list(destination.parent.glob(".applio-backup-*"))
