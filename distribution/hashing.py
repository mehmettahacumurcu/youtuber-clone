"""Streaming file hashing and complete packaged-tree verification."""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from distribution.models import ComponentManifest, _windows_path_key


_BLOCK_SIZE = 8 * 1024 * 1024
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)


class ComponentVerificationError(ValueError):
    """Raised when an unpacked component differs from its manifest."""


def _checked_lstat(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ComponentVerificationError(f"cannot inspect component path: {path}") from error
    attributes = getattr(metadata, "st_file_attributes", 0)
    if stat.S_ISLNK(metadata.st_mode) or attributes & _REPARSE_POINT:
        raise ComponentVerificationError(f"symlink or reparse point in component tree: {path}")
    return metadata


def _component_entries(root: Path):
    """Walk a component without following links or Windows reparse directories."""
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as error:
            raise ComponentVerificationError(
                f"cannot enumerate component directory: {directory}"
            ) from error
        child_directories: list[Path] = []
        for entry in entries:
            path = Path(entry.path)
            metadata = _checked_lstat(path)
            yield path, metadata
            if stat.S_ISDIR(metadata.st_mode):
                child_directories.append(path)
            elif not stat.S_ISREG(metadata.st_mode):
                raise ComponentVerificationError(f"special file in component tree: {path}")
        pending.extend(reversed(child_directories))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def verify_component_tree(root: Path, manifest: ComponentManifest) -> None:
    root = Path(root)
    root_metadata = _checked_lstat(root)
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ComponentVerificationError("component root must be a real directory")
    expected = {entry.path: entry for entry in manifest.files}
    found: dict[str, Path] = {}
    found_windows_keys: set[str] = set()
    for path, metadata in _component_entries(root):
        relative = path.relative_to(root).as_posix()
        key = _windows_path_key(relative)
        if key in found_windows_keys:
            raise ComponentVerificationError(f"case-colliding path in component tree: {relative}")
        found_windows_keys.add(key)
        if stat.S_ISREG(metadata.st_mode):
            found[relative] = path
    manifest_path = root / "component-manifest.json"
    manifest_metadata = _checked_lstat(manifest_path)
    if not stat.S_ISREG(manifest_metadata.st_mode):
        raise ComponentVerificationError("component manifest metadata is missing")
    try:
        on_disk = ComponentManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ComponentVerificationError("component manifest metadata is invalid") from error
    if on_disk != manifest:
        raise ComponentVerificationError("component manifest metadata does not match")
    payload_found = set(found) - {"component-manifest.json"}
    if payload_found != set(expected) or set(found) != set(expected) | {"component-manifest.json"}:
        raise ComponentVerificationError("component tree files do not match manifest")
    for relative, entry in expected.items():
        actual = found[relative]
        actual_metadata = _checked_lstat(actual)
        if (
            not stat.S_ISREG(actual_metadata.st_mode)
            or actual_metadata.st_size != entry.size
            or sha256_file(actual) != entry.sha256
        ):
            raise ComponentVerificationError(f"component file does not match manifest: {relative}")
