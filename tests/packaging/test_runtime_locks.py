"""Contract tests for the isolated PyInstaller build environments."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = ROOT / "packaging" / "runtime"


STUDIO_DIRECT = {
    "datasets": "==3.6.0",
    "fastapi": "==0.136.1",
    "flagembedding": "==1.4.0",
    "gradio": "==5.50.0",
    "numpy": "==1.26.4",
    "pydantic": "==2.12.3",
    "pydantic-settings": "==2.14.1",
    "pyinstaller": "==6.21.0",
    "pyyaml": "==6.0.3",
    "qdrant-client": "==1.12.1",
    "rank-bm25": "==0.2.2",
    "requests": "==2.34.2",
    "soundfile": "==0.13.1",
    "torch": "==2.7.1+cpu",
    "transformers": "==4.57.6",
    "uvicorn": "==0.47.0",
}

VOICE_DIRECT = {
    "coqui-tts": "==0.26.2",
    "einops": "==0.8.2",
    "faiss-cpu": "==1.13.2",
    "fastapi": "==0.140.13",
    "librosa": "==0.11.0",
    "noisereduce": "==3.0.3",
    "numpy": "==2.4.4",
    "pedalboard": "==0.9.24",
    "pydantic": "==2.12.3",
    "pydantic-settings": "==2.14.2",
    "pyinstaller": "==6.21.0",
    "pyyaml": "==6.0.3",
    "requests": "==2.34.2",
    "scipy": "==1.17.1",
    "soundfile": "==0.13.1",
    "soxr": "==1.1.0",
    "torch": "==2.7.1+cu128",
    "torchaudio": "==2.7.1+cu128",
    "torchcrepe": "==0.0.24",
    "torchfcpe": "==0.0.4",
    "transformers": "==4.48.3",
    "uvicorn": "==0.51.0",
    "wget": "==3.2",
}


def _project(name: str) -> tuple[Path, dict[str, object]]:
    path = RUNTIME_ROOT / name / "pyproject.toml"
    assert path.is_file(), f"missing runtime project: {path}"
    return path, tomllib.loads(path.read_text(encoding="utf-8"))


def _direct_dependencies(project: dict[str, object]) -> dict[str, str]:
    values = project["project"]["dependencies"]  # type: ignore[index]
    result: dict[str, str] = {}
    for dependency in values:  # type: ignore[union-attr]
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)\s*(==[^;\s]+)", dependency)
        assert match, f"direct dependency must be an exact unmarked pin: {dependency!r}"
        result[match.group(1).lower()] = match.group(2)
    return result


@pytest.mark.parametrize(
    ("name", "expected", "torch_index", "forbidden"),
    [
        ("studio", STUDIO_DIRECT, "pytorch-cpu", {"coqui-tts", "tts", "torchaudio", "faiss-cpu"}),
        (
            "voice",
            VOICE_DIRECT,
            "pytorch-cu128",
            {"qdrant-client", "flagembedding", "gradio", "rank-bm25", "datasets"},
        ),
    ],
)
def test_runtime_projects_are_exact_and_isolated(name, expected, torch_index, forbidden):
    path, project = _project(name)
    assert project["project"]["requires-python"] == ">=3.11,<3.12"
    assert project["tool"]["uv"]["package"] is False
    dependencies = _direct_dependencies(project)
    assert dependencies == expected
    assert not (dependencies.keys() & forbidden)

    indexes = project["tool"]["uv"]["index"]
    assert indexes == [
        {
            "name": torch_index,
            "url": f"https://download.pytorch.org/whl/{torch_index.removeprefix('pytorch-')}",
            "explicit": True,
        }
    ]
    sources = project["tool"]["uv"]["sources"]
    assert sources == {"torch": {"index": torch_index}, **({"torchaudio": {"index": torch_index}} if name == "voice" else {})}
    assert "path" not in path.read_text(encoding="utf-8").lower()
    assert "editable" not in path.read_text(encoding="utf-8").lower()


@pytest.mark.parametrize(
    ("name", "torch_version", "required_markers"),
    [
        ("studio", "2.7.1+cpu", ("win_amd64", "cp311")),
        ("voice", "2.7.1+cu128", ("win_amd64", "cp311")),
    ],
)
def test_locks_are_immutable_and_windows_resolved(name, torch_version, required_markers):
    lock = RUNTIME_ROOT / name / "uv.lock"
    assert lock.is_file(), f"missing runtime lock: {lock}"
    text = lock.read_text(encoding="utf-8")
    assert re.search(r"^version = 1$", text, flags=re.MULTILINE)
    assert re.search(r"^revision = \d+$", text, flags=re.MULTILINE)
    assert f'name = "torch"\nversion = "{torch_version}"' in text
    assert "sha256:" in text and "https://" in text
    assert "editable = true" not in text.lower()
    assert not re.search(r"(?:[A-Za-z]:\\|file:|path =|@ git\+|git\+https?://.*@(?![0-9a-f]{40}))", text, re.IGNORECASE)
    assert text.count('name = "torch"\n') == 1
    for marker in required_markers:
        assert marker in text


@pytest.mark.parametrize(
    ("name", "expected", "forbidden"),
    [
        ("studio", STUDIO_DIRECT, {"coqui-tts", "tts", "torchaudio", "faiss-cpu"}),
        ("voice", VOICE_DIRECT, {"qdrant-client", "flagembedding", "gradio", "rank-bm25"}),
    ],
)
def test_lock_package_inventory_is_complete_and_has_no_conflicting_sources(name, expected, forbidden):
    lock = RUNTIME_ROOT / name / "uv.lock"
    resolved = tomllib.loads(lock.read_text(encoding="utf-8"))["package"]
    names = {package["name"].lower() for package in resolved}
    assert set(expected).issubset(names)
    assert not (names & forbidden)

    versions: dict[str, set[str]] = {}
    for package in resolved:
        package_name = package["name"].lower()
        versions.setdefault(package_name, set()).add(package["version"])
        source = package.get("source", {})
        assert not ({"directory", "editable", "git", "url"} & set(source))
    assert all(len(resolved_versions) == 1 for resolved_versions in versions.values())
