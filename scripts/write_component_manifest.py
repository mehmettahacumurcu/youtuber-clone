"""Create and verify deterministic worker manifests and ZIP archives."""
from __future__ import annotations

import argparse
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath

from distribution.hashing import sha256_file, verify_component_tree
from distribution.models import ComponentManifest, FileEntry, _windows_path_key


MANIFEST_NAME = "component-manifest.json"
ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)
ZIP_COMPRESSION_LEVEL = 1
_REPARSE_POINT = 0x0400
_FORBIDDEN_PARTS = {
    ".git", ".pytest_cache", "__pycache__", "tests", "test", "logs",
    ".venv", ".venv-tts", ".venv-dist-dev", ".venv-build-studio",
    ".venv-build-voice", "notebooks", "training", "testing", "test_data",
}
_FORBIDDEN_MODEL_SUFFIXES = {".ckpt", ".gguf", ".index", ".pth", ".pt", ".safetensors"}


def _is_link_or_reparse(path: Path) -> bool:
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & _REPARSE_POINT)


def _validate_payload_path(relative: str) -> None:
    parts = tuple(part.casefold() for part in PurePosixPath(relative).parts)
    name = parts[-1]
    if any(part in _FORBIDDEN_PARTS for part in parts):
        raise ValueError(f"forbidden path in component bundle: {relative}")
    if name == ".env" or name.startswith(".env."):
        raise ValueError(f"forbidden environment file in component bundle: {relative}")
    if PurePosixPath(relative).suffix.casefold() in _FORBIDDEN_MODEL_SUFFIXES:
        raise ValueError(f"forbidden model artifact in component bundle: {relative}")
    if name.endswith((".ipynb", ".log", ".pyc")):
        raise ValueError(f"forbidden generated file in component bundle: {relative}")


def _payload_files(root: Path) -> tuple[tuple[str, Path], ...]:
    root = Path(root)
    if _is_link_or_reparse(root) or not root.is_dir():
        raise ValueError("component root must be a real directory")
    found: list[tuple[str, Path]] = []
    windows_keys: set[str] = set()
    for candidate in root.rglob("*"):
        if _is_link_or_reparse(candidate):
            raise ValueError(f"symlink or reparse point in component bundle: {candidate}")
        if candidate.is_dir():
            continue
        if not stat.S_ISREG(candidate.stat(follow_symlinks=False).st_mode):
            raise ValueError(f"special file in component bundle: {candidate}")
        relative = candidate.relative_to(root).as_posix()
        if relative == MANIFEST_NAME:
            continue
        _validate_payload_path(relative)
        key = _windows_path_key(relative)
        if key in windows_keys:
            raise ValueError(f"case-colliding paths in component bundle: {relative}")
        windows_keys.add(key)
        if candidate.stat().st_size <= 0:
            raise ValueError(f"empty file in component bundle: {relative}")
        found.append((relative, candidate))
    return tuple(sorted(found, key=lambda item: item[0]))


def build_component_manifest(
    root: Path,
    *,
    component: str,
    version: str,
    entrypoint: str,
    healthcheck: tuple[str, ...] = ("--healthcheck",),
) -> ComponentManifest:
    files = tuple(
        FileEntry(path=relative, size=path.stat().st_size, sha256=sha256_file(path))
        for relative, path in _payload_files(Path(root))
    )
    return ComponentManifest.model_validate(
        {
            "schema": "youtuber.component.v1",
            "component": component,
            "version": version,
            "compatibility": {"runtime_api": 1},
            "entrypoint": entrypoint,
            "healthcheck": list(healthcheck),
            "files": [entry.model_dump() for entry in files],
        }
    )


def _manifest_bytes(manifest: ComponentManifest) -> bytes:
    payload = manifest.model_dump(mode="json", by_alias=True)
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_component_manifest(
    root: Path,
    *,
    component: str,
    version: str,
    entrypoint: str,
    healthcheck: tuple[str, ...] = ("--healthcheck",),
) -> ComponentManifest:
    root = Path(root)
    manifest = build_component_manifest(
        root,
        component=component,
        version=version,
        entrypoint=entrypoint,
        healthcheck=healthcheck,
    )
    target = root / MANIFEST_NAME
    temporary = root / (MANIFEST_NAME + ".tmp")
    temporary.write_bytes(_manifest_bytes(manifest))
    temporary.replace(target)
    verify_component_tree(root, manifest)
    return manifest


def write_deterministic_zip(root: Path, archive: Path) -> None:
    root = Path(root)
    archive = Path(archive)
    manifest_path = root / MANIFEST_NAME
    manifest = ComponentManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    verify_component_tree(root, manifest)
    if archive.resolve(strict=False).is_relative_to(root.resolve(strict=False)):
        raise ValueError("archive output must be outside the component root")
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(archive.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        entries = []
        for relative, path in _payload_files(root):
            entries.append((f"{root.name}/{relative}", path))
        entries.append((f"{root.name}/{MANIFEST_NAME}", manifest_path))
        entries.sort(key=lambda item: item[0])
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=ZIP_COMPRESSION_LEVEL,
            strict_timestamps=True,
        ) as handle:
            for name, path in entries:
                info = zipfile.ZipInfo(name, date_time=ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                info.flag_bits |= 0x800
                handle.writestr(
                    info,
                    path.read_bytes(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=ZIP_COMPRESSION_LEVEL,
                )
        temporary.replace(archive)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--component", choices=("studio_runtime", "voice_runtime"), required=True)
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--zip", dest="archive", type=Path)
    args = parser.parse_args(argv)
    manifest = write_component_manifest(
        args.bundle,
        component=args.component,
        version=args.version,
        entrypoint=args.entrypoint,
    )
    if args.archive:
        write_deterministic_zip(args.bundle, args.archive)
    print(
        json.dumps(
            {"component": manifest.component, "files": len(manifest.files), "bundle": str(args.bundle)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
