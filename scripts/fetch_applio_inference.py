"""Fetch and stage the hash-pinned, inference-only Applio source tree.

This script deliberately vendors source only.  ContentVec and RMVPE are
hash-pinned files of the separate Speaker voice artifact and are never fetched
from the Applio repository.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


APPLIO_REPOSITORY = "https://github.com/IAHispano/Applio.git"
APPLIO_COMMIT = "3e5e248c2bd003f65f3e128dd12677533def27be"
LOCK_SCHEMA = "youtuber.applio.vendor.v1"
REQUIRED_VOICE_ARTIFACT_FILES = (
    "rvc/models/embedders/contentvec/config.json",
    "rvc/models/embedders/contentvec/pytorch_model.bin",
    "rvc/models/predictors/rmvpe.pt",
)

# This is the static import closure of rvc.infer.infer and rvc.infer.pipeline
# for a ContentVec + RMVPE conversion at the pinned commit.  `rvc/configs` and
# `rvc/infer` are copied in full because Config opens each JSON configuration.
AUDITED_LIB_PATHS = frozenset(
    {
        "rvc/lib/algorithm/attentions.py",
        "rvc/lib/algorithm/commons.py",
        "rvc/lib/algorithm/encoders.py",
        "rvc/lib/algorithm/generators/hifigan.py",
        "rvc/lib/algorithm/generators/hifigan_mrf.py",
        "rvc/lib/algorithm/generators/hifigan_nsf.py",
        "rvc/lib/algorithm/generators/refinegan.py",
        "rvc/lib/algorithm/modules.py",
        "rvc/lib/algorithm/normalization.py",
        "rvc/lib/algorithm/residuals.py",
        "rvc/lib/algorithm/synthesizers.py",
        "rvc/lib/predictors/RMVPE.py",
        "rvc/lib/predictors/f0.py",
        "rvc/lib/tools/split_audio.py",
        "rvc/lib/utils.py",
    }
)
_SHA256 = r"^[0-9a-f]{64}$"
_FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
_WINDOWS_RESERVED_NAMES = {
    "con", "prn", "aux", "nul", *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


class ImmutableDict(dict):
    """A JSON-serializable mapping that cannot be mutated after validation."""

    @staticmethod
    def _immutable(*_args, **_kwargs):
        raise TypeError("mapping is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _safe_relative_path(value: str) -> str:
    """Validate a portable relative path before it is joined to a root."""
    if not isinstance(value, str) or not value or "\\" in value or ":" in value or "\x00" in value:
        raise ValueError("path must be a safe relative POSIX path")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("path must be a safe relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts:
        raise ValueError("path must be a safe relative POSIX path")
    for part in path.parts:
        if part.rstrip(" .") != part or part.rstrip(" .").casefold() in _WINDOWS_RESERVED_NAMES:
            raise ValueError("path must be safe on Windows")
    return path.as_posix()


def _windows_path_key(value: str) -> str:
    return "/".join(part.casefold() for part in PurePosixPath(_safe_relative_path(value)).parts)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_regular_file(path: Path) -> bool:
    return stat.S_ISREG(path.stat(follow_symlinks=False).st_mode)


def _is_link_or_reparse_point(path: Path) -> bool:
    """Reject POSIX links and Windows reparse points before touching contents."""
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


class VendorFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    size: int = Field(gt=0)
    sha256: str = Field(pattern=_SHA256)


class AuxiliaryAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    supplied_by: Literal["voice_artifact"]


class VendorLock(BaseModel):
    """Strict, deterministic inventory for the checked-in Applio source lock."""

    model_config = ConfigDict(extra="forbid", frozen=True, serialize_by_alias=True)

    schema_: Literal[LOCK_SCHEMA] = Field(alias="schema")
    repository: Literal[APPLIO_REPOSITORY]
    commit: Literal[APPLIO_COMMIT]
    files: Mapping[str, VendorFile]
    auxiliary_assets: Mapping[str, AuxiliaryAsset]

    @property
    def schema(self) -> Literal[LOCK_SCHEMA]:
        return self.schema_

    @model_validator(mode="after")
    def _validate_contract(self) -> "VendorLock":
        if not self.files:
            raise ValueError("vendor lock must declare source files")
        paths = tuple(self.files)
        if paths != tuple(sorted(paths)):
            raise ValueError("vendor lock source files must be sorted")
        keys = [_windows_path_key(path) for path in paths]
        if len(keys) != len(set(keys)):
            raise ValueError("vendor lock contains duplicate Windows paths")
        lib_paths = {path for path in paths if path.startswith("rvc/lib/")}
        if lib_paths != AUDITED_LIB_PATHS:
            raise ValueError("vendor lock must contain the exact audited rvc/lib closure")
        for path in paths:
            _safe_relative_path(path)
            if path.startswith("rvc/models/pretraineds/"):
                raise ValueError("training pretrained assets are forbidden")
            if path.startswith("rvc/lib/"):
                if path not in AUDITED_LIB_PATHS:
                    raise ValueError("vendor lock contains an undeclared rvc/lib file")
            elif not (path.startswith("rvc/configs/") or path.startswith("rvc/infer/")):
                raise ValueError("vendor lock contains a path outside the audited inference source")

        assets = tuple(self.auxiliary_assets)
        if assets != REQUIRED_VOICE_ARTIFACT_FILES:
            raise ValueError("vendor lock must declare exactly the required voice artifact assets")
        asset_keys = [_windows_path_key(path) for path in assets]
        if len(asset_keys) != len(set(asset_keys)):
            raise ValueError("vendor lock contains duplicate Windows paths")
        for path in assets:
            _safe_relative_path(path)
        object.__setattr__(self, "files", ImmutableDict(self.files))
        object.__setattr__(self, "auxiliary_assets", ImmutableDict(self.auxiliary_assets))
        return self

    @classmethod
    def load(cls, path: Path) -> "VendorLock":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def json_text(self) -> str:
        """Return the sole accepted on-disk representation, with stable ordering."""
        return json.dumps(self.model_dump(by_alias=True), ensure_ascii=False, indent=2) + "\n"


def _iter_regular_files(root: Path) -> dict[str, Path]:
    if _is_link_or_reparse_point(root) or not root.is_dir():
        raise ValueError("vendor root must be a real directory")
    files: dict[str, Path] = {}
    case_keys: set[str] = set()
    for candidate in root.rglob("*"):
        if _is_link_or_reparse_point(candidate):
            raise ValueError(f"symlink in vendor tree: {candidate}")
        if candidate.is_dir():
            continue
        if not _is_regular_file(candidate):
            raise ValueError(f"special file in vendor tree: {candidate}")
        relative = candidate.relative_to(root).as_posix()
        _safe_relative_path(relative)
        key = _windows_path_key(relative)
        if key in case_keys:
            raise ValueError(f"case-colliding paths in vendor tree: {relative}")
        case_keys.add(key)
        files[relative] = candidate
    return files


def verify_vendor_tree(root: Path, lock: VendorLock) -> None:
    """Fail closed unless a staging tree is exactly the lock inventory."""
    found = _iter_regular_files(Path(root))
    expected = set(lock.files)
    actual = set(found)
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unexpected:
        raise ValueError("unexpected source files in vendor tree: " + ", ".join(unexpected))
    if missing:
        raise ValueError("missing declared source files in vendor tree: " + ", ".join(missing))
    for relative, entry in lock.files.items():
        candidate = found[relative]
        if candidate.stat().st_size != entry.size or _sha256_file(candidate) != entry.sha256:
            raise ValueError(f"modified declared source file: {relative}")


def _source_config_and_infer_files(source: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for relative_root in ("rvc/configs", "rvc/infer"):
        subtree = source / relative_root
        if _is_link_or_reparse_point(subtree) or not subtree.is_dir():
            raise ValueError(f"required Applio source directory is missing or unsafe: {relative_root}")
        for relative, path in _iter_regular_files(subtree).items():
            discovered[f"{relative_root}/{relative}"] = path
    return discovered


def audited_source_paths(source: Path) -> tuple[str, ...]:
    """Return the complete audited source inventory for the pinned checkout."""
    source = Path(source)
    configured = _source_config_and_infer_files(source)
    required = set(configured)
    required.update(AUDITED_LIB_PATHS)
    for relative in AUDITED_LIB_PATHS:
        candidate = source / PurePosixPath(relative)
        if _is_link_or_reparse_point(candidate) or not _is_regular_file(candidate):
            raise ValueError(f"required Applio inference import is missing or unsafe: {relative}")
    return tuple(sorted(required))


def lock_from_source(source: Path, *, existing: VendorLock | None = None) -> VendorLock:
    """Hash the exact audited inventory; callers decide whether writing is allowed."""
    paths = tuple(existing.files) if existing else audited_source_paths(source)
    source = Path(source)
    files: dict[str, VendorFile] = {}
    for relative in paths:
        _safe_relative_path(relative)
        candidate = source / PurePosixPath(relative)
        if _is_link_or_reparse_point(candidate) or not _is_regular_file(candidate):
            raise ValueError(f"declared Applio source file is missing or unsafe: {relative}")
        files[relative] = VendorFile(size=candidate.stat().st_size, sha256=_sha256_file(candidate))
    lock = VendorLock(
        schema=LOCK_SCHEMA,
        repository=APPLIO_REPOSITORY,
        commit=APPLIO_COMMIT,
        files=dict(sorted(files.items())),
        auxiliary_assets={path: AuxiliaryAsset(supplied_by="voice_artifact") for path in REQUIRED_VOICE_ARTIFACT_FILES},
    )
    if existing and tuple(lock.files) != tuple(existing.files):
        raise ValueError("refresh-lock may update hashes only; the source inventory changed")
    return lock


def verify_audited_source(source: Path, lock: VendorLock) -> None:
    """Reject drift in copied roots before any source is staged."""
    source = Path(source)
    config_and_infer = _source_config_and_infer_files(source)
    locked_config_and_infer = {
        path for path in lock.files if path.startswith("rvc/configs/") or path.startswith("rvc/infer/")
    }
    if set(config_and_infer) != locked_config_and_infer:
        raise ValueError("undeclared source/config files in audited Applio checkout")
    refreshed = lock_from_source(source, existing=lock)
    if refreshed != lock:
        raise ValueError("modified declared source/config file in audited Applio checkout")


def stage_vendor_tree(source: Path, destination: Path, lock: VendorLock) -> None:
    """Copy only lock entries into an atomically replaced, verified staging tree."""
    source = Path(source)
    destination = Path(destination)
    verify_audited_source(source, lock)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".applio-stage-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "applio"
        staging.mkdir()
        for relative in lock.files:
            source_file = source / PurePosixPath(relative)
            target = staging / PurePosixPath(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, target, follow_symlinks=False)
        verify_vendor_tree(staging, lock)
        if not destination.exists():
            os.replace(staging, destination)
            return
        if _is_link_or_reparse_point(destination) or not destination.is_dir():
            raise ValueError("vendor destination must be a real directory when it already exists")
        backup = destination.parent / f".applio-backup-{uuid.uuid4().hex}"
        os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except BaseException as promotion_error:
            try:
                os.replace(backup, destination)
            except BaseException as rollback_error:
                raise RuntimeError("Applio promotion failed and rollback could not restore destination") from rollback_error
            raise promotion_error
        try:
            shutil.rmtree(backup)
        except BaseException as cleanup_error:
            raise RuntimeError("Applio promotion succeeded but backup cleanup failed") from cleanup_error


def _run_git(args: list[str], *, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, check=False, text=True, capture_output=True
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def fetch_pinned_checkout(build_root: Path) -> tempfile.TemporaryDirectory[str]:
    """Clone with blob filtering and return a context-managed detached checkout."""
    build_root.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix=".applio-fetch-", dir=build_root)
    checkout = Path(temporary.name) / "checkout"
    try:
        _run_git(["clone", "--filter=blob:none", "--no-checkout", APPLIO_REPOSITORY, str(checkout)])
        if _run_git(["remote", "get-url", "origin"], cwd=checkout) != APPLIO_REPOSITORY:
            raise RuntimeError("Applio checkout remote does not match the pinned repository")
        _run_git(["fetch", "--filter=blob:none", "origin", APPLIO_COMMIT], cwd=checkout)
        _run_git(["checkout", "--detach", APPLIO_COMMIT], cwd=checkout)
        if _run_git(["rev-parse", "HEAD"], cwd=checkout) != APPLIO_COMMIT:
            raise RuntimeError("Applio checkout does not match the pinned commit")
    except BaseException:
        temporary.cleanup()
        raise
    return temporary


def _refresh_lock(lock_path: Path, source: Path, old: VendorLock) -> None:
    new = lock_from_source(source, existing=old)
    old_text = old.json_text().splitlines(keepends=True)
    new_text = new.json_text().splitlines(keepends=True)
    print(f"Refreshing Applio lock: repository={APPLIO_REPOSITORY} commit={APPLIO_COMMIT}")
    diff = "".join(
        difflib.unified_diff(
            old_text, new_text, fromfile="old/applio.lock.json", tofile="new/applio.lock.json"
        )
    )
    print(diff if diff else "No hash changes.", end="" if diff else "\n")
    lock_path.write_text(new.json_text(), encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path("third_party/applio.lock.json"))
    parser.add_argument("--destination", type=Path, default=Path("build/vendor/applio"))
    parser.add_argument("--refresh-lock", action="store_true")
    args = parser.parse_args(argv)
    lock = VendorLock.load(args.lock)
    build_root = args.destination.parent.parent
    with fetch_pinned_checkout(build_root) as checkout_path:
        checkout = Path(checkout_path) / "checkout"
        if args.refresh_lock:
            _refresh_lock(args.lock, checkout, lock)
            lock = VendorLock.load(args.lock)
        stage_vendor_tree(checkout, args.destination, lock)
    print(f"Verified Applio inference source at {args.destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
