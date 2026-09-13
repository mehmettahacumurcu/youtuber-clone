from pathlib import Path

import pytest
from pydantic import ValidationError

from distribution.models import AssetManifest, ComponentManifest, DistributionManifest


FIXTURE = Path("tests/fixtures/distribution/component-manifest.json")
ASSET_FIXTURE = Path("tests/fixtures/distribution/asset-manifest.json")
DISTRIBUTION_FIXTURES = {
    "distribution-valid.json": True,
    "distribution-invalid-main-revision.json": False,
    "distribution-invalid-uppercase-hash.json": False,
    "distribution-invalid-zero-size.json": False,
    "distribution-invalid-traversal.json": False,
    "distribution-invalid-runtime-api-2.json": False,
    "distribution-invalid-duplicate-install-path.json": False,
    "distribution-invalid-zero-hash.json": False,
    "distribution-invalid-unknown-property.json": False,
    "distribution-invalid-missing-repo.json": False,
    "distribution-invalid-string-size.json": False,
    "distribution-invalid-boolean-runtime-api.json": False,
    "distribution-invalid-empty-path-segment.json": False,
    "distribution-invalid-dot-path-segment.json": False,
    "distribution-invalid-whitespace-path.json": False,
    "distribution-invalid-model-role.json": False,
}


def ollama_payload():
    return {
        "installer": {
            "repo": "owner/ollama", "revision": "1" * 40,
            "path": "windows/ollama-installer.exe", "size": 1,
            "sha256": "b" * 64, "publisher": "Ollama, Inc.", "install_size": 2,
        },
        "release_version": "1.0.0", "release_identity": "ollama-1.0.0",
        "models": {
            "verifier": {
                "name": "qwen3:4b",
                "source": {
                    "repo": "owner/models", "revision": "2" * 40,
                    "path": "qwen3/model.gguf", "size": 3, "sha256": "e" * 64,
                },
                "installed_digest": "f" * 64,
                "installed_size": 4,
            },
            "speaker": {
                "name": "speaker-v5-a636",
                "gguf": {"repo": "owner/models", "revision": "2" * 40, "path": "model.gguf", "size": 1, "sha256": "c" * 64},
                "modelfile": {"repo": "owner/models", "revision": "2" * 40, "path": "Modelfile", "size": 1, "sha256": "d" * 64},
                "installed_size": 5,
            },
        },
    }


def webview2_payload():
    return {"installer": {
        "repo": "owner/webview2", "revision": "3" * 40,
        "path": "windows/webview2.exe", "size": 1,
        "sha256": "e" * 64, "publisher": "Microsoft Corporation", "install_size": 6,
    }}


def component_payload(**overrides):
    payload = {
        "schema": "youtuber.component.v1",
        "component": "voice_runtime",
        "version": "1.0.0",
        "compatibility": {"runtime_api": 1},
        "entrypoint": "voice-worker.exe",
        "healthcheck": ["--healthcheck"],
        "files": [{"path": "voice-worker.exe", "size": 1, "sha256": "0" * 64}],
    }
    payload.update(overrides)
    return payload


def test_component_manifest_accepts_the_committed_fixture():
    manifest = ComponentManifest.model_validate_json(FIXTURE.read_text("utf-8"))
    assert manifest.schema == "youtuber.component.v1"
    assert manifest.model_dump(by_alias=True)["schema"] == "youtuber.component.v1"
    assert manifest.component == "voice_runtime"
    assert manifest.entrypoint == "voice-worker.exe"
    assert manifest.files[0].sha256 == "a" * 64


def test_component_manifest_default_json_uses_schema_alias_and_round_trips():
    manifest = ComponentManifest.model_validate_json(FIXTURE.read_text("utf-8"))
    serialized = manifest.model_dump_json()
    assert '"schema"' in serialized
    assert "schema_" not in serialized
    assert ComponentManifest.model_validate_json(serialized) == manifest


def test_data_asset_manifest_has_the_same_exact_inventory_contract_as_the_launcher():
    manifest = AssetManifest.model_validate_json(ASSET_FIXTURE.read_text("utf-8"))

    assert manifest.schema == "youtuber.asset.v1"
    assert manifest.component == "rag"
    assert manifest.files[0].path == "index/corpus.txt"


@pytest.mark.parametrize("path", ["../escape.exe", "/absolute.exe", "C:/escape.exe"])
def test_component_manifest_rejects_unsafe_paths(path):
    with pytest.raises(ValidationError):
        ComponentManifest.model_validate(component_payload(files=[{"path": path, "size": 1, "sha256": "0" * 64}]))


@pytest.mark.parametrize("field,value", [("schema", "youtuber.component.v2"), ("files", [{"path": "voice-worker.exe", "size": 0, "sha256": "0" * 64}])])
def test_component_manifest_rejects_unknown_schema_and_zero_byte_files(field, value):
    with pytest.raises(ValidationError):
        ComponentManifest.model_validate(component_payload(**{field: value}))


def test_component_manifest_rejects_uppercase_hash_duplicate_windows_path_and_missing_entrypoint():
    files = [
        {"path": "bin/voice-worker.exe", "size": 1, "sha256": "A" * 64},
        {"path": "BIN/VOICE-WORKER.EXE", "size": 1, "sha256": "0" * 64},
    ]
    with pytest.raises(ValidationError):
        ComponentManifest.model_validate(component_payload(entrypoint="missing.exe", files=files))


def test_distribution_manifest_accepts_immutable_runtime_component_catalog():
    manifest = DistributionManifest.model_validate(
        {
            "schema": "youtuber.distribution.v1",
            "release": "1.0.0",
            "minimum": {"windows": "10.0.19045", "vram_bytes": 1, "ram_bytes": 1, "disk_bytes": 1, "runtime_api": 1},
            "ollama": ollama_payload(),
            "webview2": webview2_payload(),
            "components": {
                "voice_runtime": {
                    "repo": "your-account/youtuber-runtime",
                    "revision": "1" * 40,
                    "path": "windows/voice-worker.zip",
                    "size": 1,
                    "sha256": "a" * 64,
                    "manifest_sha256": "b" * 64,
                    "package_type": "runtime_archive",
                    "install_root": "runtime",
                    "archive_root": "voice-worker",
                    "expanded_size": 2,
                    "install_size": 1,
                    "peak_space": 1,
                    "entrypoint": "voice-worker.exe",
                    "healthcheck": ["--healthcheck"],
                    "compatibility": {"runtime_api": 1},
                    "licenses": ["THIRD_PARTY_NOTICES.txt"],
                }
            },
        }
    )
    assert manifest.components["voice_runtime"].revision == "1" * 40
    serialized = manifest.model_dump_json()
    assert '"schema"' in serialized
    assert "schema_" not in serialized
    assert DistributionManifest.model_validate_json(serialized) == manifest
    with pytest.raises(TypeError):
        manifest.components["voice_runtime"] = manifest.components["voice_runtime"]
    with pytest.raises(TypeError):
        del manifest.components["voice_runtime"]


def test_distribution_carries_prerequisites_model_roles_and_component_classification():
    manifest = DistributionManifest.model_validate_json(
        (Path("tests/fixtures/distribution") / "distribution-valid.json").read_text("utf-8")
    )

    assert manifest.webview2.installer.publisher == "Microsoft Corporation"
    assert manifest.ollama.models.verifier.name == "qwen3:4b"
    assert manifest.ollama.models.verifier.source.size == 3
    assert manifest.ollama.models.verifier.installed_size == 4
    assert manifest.ollama.models.speaker.installed_size == 5
    assert manifest.ollama.installer.install_size == 2
    assert manifest.webview2.installer.install_size == 6
    assert manifest.components["voice_runtime"].expanded_size == 2
    assert manifest.ollama.models.speaker.name == "speaker-v5-a636"
    assert manifest.components["voice_runtime"].package_type == "runtime_archive"
    assert manifest.components["voice_runtime"].install_root == "runtime"


@pytest.mark.parametrize("override", [
    {"revision": "main"},
    {"sha256": "A" * 64},
    {"size": 0},
    {"archive_root": "../escape"},
    {"compatibility": {"runtime_api": 2}},
    {"package_type": "data_archive", "install_root": "models"},
])
def test_distribution_manifest_rejects_unpinned_or_incompatible_component(override):
    payload = {
        "schema": "youtuber.distribution.v1", "release": "1.0.0",
        "minimum": {"windows": "10.0.19045", "vram_bytes": 1, "ram_bytes": 1, "disk_bytes": 1, "runtime_api": 1},
        "ollama": ollama_payload(),
        "webview2": webview2_payload(),
        "components": {"voice_runtime": {
            "repo": "your-account/youtuber-runtime", "revision": "1" * 40,
            "path": "windows/voice-worker.zip", "size": 1, "sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "package_type": "runtime_archive", "install_root": "runtime",
            "archive_root": "voice-worker", "expanded_size": 2, "install_size": 1, "peak_space": 1,
            "entrypoint": "voice-worker.exe", "healthcheck": ["--healthcheck"],
            "compatibility": {"runtime_api": 1}, "licenses": ["THIRD_PARTY_NOTICES.txt"],
        }},
    }
    payload["components"]["voice_runtime"].update(override)
    with pytest.raises(ValidationError):
        DistributionManifest.model_validate(payload)


@pytest.mark.parametrize(("filename", "accepted"), DISTRIBUTION_FIXTURES.items())
def test_distribution_contract_fixtures_match_launcher_validation(filename, accepted):
    content = (Path("tests/fixtures/distribution") / filename).read_text("utf-8")

    if accepted:
        manifest = DistributionManifest.model_validate_json(content)
        assert manifest.release == "1.0.0"
        assert manifest.components["voice_runtime"].revision == "1" * 40
    else:
        with pytest.raises(ValidationError):
            DistributionManifest.model_validate_json(content)
