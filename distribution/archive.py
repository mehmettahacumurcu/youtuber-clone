"""Safe ZIP extraction for component staging directories."""
from __future__ import annotations

import stat
import zipfile
from pathlib import Path

from distribution.models import _validate_relative_path, _windows_path_key


class UnsafeArchiveError(ValueError):
    """Raised when an archive cannot be extracted without escaping its root."""


def _is_unix_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK


def safe_extract_zip(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        infos = handle.infolist()
        entries: list[tuple[zipfile.ZipInfo, str, str]] = []
        normalized: set[str] = set()
        file_keys: set[str] = set()
        for info in infos:
            name = info.filename.rstrip("/")
            if not name:
                continue
            try:
                _validate_relative_path(name)
            except ValueError as error:
                raise UnsafeArchiveError(str(error)) from error
            try:
                key = _windows_path_key(name)
            except ValueError as error:
                raise UnsafeArchiveError(str(error)) from error
            if key in normalized:
                raise UnsafeArchiveError("archive contains duplicate Windows paths")
            normalized.add(key)
            if _is_unix_symlink(info):
                raise UnsafeArchiveError("archive contains a symlink")
            entries.append((info, name, key))
            if not info.is_dir():
                file_keys.add(key)

        for _, _, key in entries:
            parents = key.split("/")
            if any("/".join(parents[:index]) in file_keys for index in range(1, len(parents))):
                raise UnsafeArchiveError("archive contains a file ancestor conflict")

        if not destination.is_dir() or any(destination.iterdir()):
            raise UnsafeArchiveError("destination must be an existing empty staging directory")
        root = destination.resolve()
        for info, name, _ in entries:
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise UnsafeArchiveError("archive entry escapes destination")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(info) as source, target.open("xb") as output:
                while block := source.read(8 * 1024 * 1024):
                    output.write(block)
