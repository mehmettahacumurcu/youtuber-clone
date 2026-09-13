import stat
import zipfile

import pytest

from distribution.archive import UnsafeArchiveError, safe_extract_zip


def test_safe_extract_rejects_parent_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "no")
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(archive, tmp_path / "out")


def test_safe_extract_rejects_unix_symlink_and_nonempty_destination(tmp_path):
    archive = tmp_path / "bad.zip"
    info = zipfile.ZipInfo("link")
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(info, "target")
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(archive, destination)


def test_safe_extract_rejects_duplicate_normalized_windows_paths(tmp_path):
    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("runtime/voice-worker.exe", b"first")
        handle.writestr("RUNTIME/./VOICE-WORKER.EXE", b"second")
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(archive, destination)


@pytest.mark.parametrize("colliding_name", ["RUNTIME/VOICE-WORKER.EXE.", "RUNTIME/VOICE-WORKER.EXE "])
def test_safe_extract_rejects_windows_trailing_name_collisions_before_writing(tmp_path, colliding_name):
    archive = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("runtime/voice-worker.exe", b"first")
        handle.writestr(colliding_name, b"second")
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(archive, destination)
    assert list(destination.iterdir()) == []


def test_safe_extract_rejects_file_ancestor_before_writing(tmp_path):
    archive = tmp_path / "ancestor-conflict.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("runtime", b"file")
        handle.writestr("runtime/voice-worker.exe", b"worker")
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(UnsafeArchiveError):
        safe_extract_zip(archive, destination)
    assert list(destination.iterdir()) == []


def test_safe_extract_writes_regular_nested_files_to_empty_existing_destination(tmp_path):
    archive = tmp_path / "good.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("runtime/voice-worker.exe", b"worker")
    destination = tmp_path / "out"
    destination.mkdir()
    safe_extract_zip(archive, destination)
    assert (destination / "runtime" / "voice-worker.exe").read_bytes() == b"worker"
