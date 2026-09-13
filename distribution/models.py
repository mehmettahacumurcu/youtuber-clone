"""Fail-closed manifests shared by runtime packagers and the launcher."""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


ComponentName = Literal["studio_runtime", "voice_runtime"]
DistributionComponentName = Literal[
    "studio_runtime", "voice_runtime", "llm", "voice", "rag", "embedder",
    "reranker", "ollama_installer", "webview2_runtime",
]
_SEMVER = r"^[0-9]+\.[0-9]+\.[0-9]+$"
_SHA256 = r"^[0-9a-f]{64}$"
_REVISION = r"^[0-9a-f]{40}$"
_MAX_INT64 = (1 << 63) - 1


class ImmutableDict(dict):
    """A JSON-serializable mapping that rejects every mutating dictionary API."""

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


def _validate_relative_path(value: str) -> str:
    if not value or not value.strip() or "\\" in value or ":" in value:
        raise ValueError("path must be a non-empty relative POSIX path")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("path must not contain empty, current, or parent segments")
    path = PurePosixPath(value)
    if path.anchor or not path.parts or ".." in path.parts:
        raise ValueError("path must not be absolute or traverse parents")
    return value


def _windows_path_key(value: str) -> str:
    """Return the Windows-equivalent key for an already validated POSIX path."""
    parts = tuple(part.rstrip(" .").casefold() for part in PurePosixPath(value).parts)
    if not all(parts):
        raise ValueError("path contains a Windows-empty name")
    return "/".join(parts)


class Compatibility(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_api: Literal[1]

    @model_validator(mode="before")
    @classmethod
    def require_integer_runtime_api(cls, value: object) -> object:
        if not isinstance(value, Mapping) or type(value.get("runtime_api")) is not int:
            raise ValueError("runtime_api must be an integer")
        return value


class FileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    size: int = Field(gt=0)
    sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def validate_path(self) -> "FileEntry":
        _validate_relative_path(self.path)
        return self


class ComponentManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, serialize_by_alias=True)

    schema_: Literal["youtuber.component.v1"] = Field(alias="schema")
    component: ComponentName
    version: str = Field(pattern=_SEMVER)
    compatibility: Compatibility
    entrypoint: str
    healthcheck: tuple[str, ...]
    files: tuple[FileEntry, ...]

    @property
    def schema(self) -> Literal["youtuber.component.v1"]:
        """Public schema name while avoiding Pydantic's deprecated class method."""
        return self.schema_

    @model_validator(mode="after")
    def validate_file_contract(self) -> "ComponentManifest":
        _validate_relative_path(self.entrypoint)
        if not self.files:
            raise ValueError("component manifest must contain files")
        normalized = [_windows_path_key(entry.path) for entry in self.files]
        if len(normalized) != len(set(normalized)):
            raise ValueError("component manifest contains duplicate Windows paths")
        if _windows_path_key(self.entrypoint) not in normalized:
            raise ValueError("entrypoint must appear in files")
        return self


class AssetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, serialize_by_alias=True)

    schema_: Literal["youtuber.asset.v1"] = Field(alias="schema")
    component: DistributionComponentName
    version: str = Field(pattern=_SEMVER)
    files: tuple[FileEntry, ...]

    @property
    def schema(self) -> Literal["youtuber.asset.v1"]:
        return self.schema_

    @model_validator(mode="after")
    def validate_file_contract(self) -> "AssetManifest":
        if not self.files:
            raise ValueError("asset manifest must contain files")
        normalized = [_windows_path_key(entry.path) for entry in self.files]
        if len(normalized) != len(set(normalized)):
            raise ValueError("asset manifest contains duplicate Windows paths")
        return self


class MinimumRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    windows: str = Field(min_length=1)
    vram_bytes: int = Field(gt=0, le=_MAX_INT64, strict=True)
    ram_bytes: int = Field(gt=0, le=_MAX_INT64, strict=True)
    disk_bytes: int = Field(gt=0, le=_MAX_INT64, strict=True)
    runtime_api: Literal[1]

    @model_validator(mode="before")
    @classmethod
    def require_integer_runtime_api(cls, value: object) -> object:
        if not isinstance(value, Mapping) or type(value.get("runtime_api")) is not int:
            raise ValueError("runtime_api must be an integer")
        return value


class DistributionComponent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    revision: str = Field(pattern=_REVISION)
    path: str
    size: int = Field(gt=0, le=_MAX_INT64, strict=True)
    sha256: str = Field(pattern=_SHA256)
    manifest_sha256: str = Field(pattern=_SHA256)
    package_type: Literal["runtime_archive", "data_archive"]
    install_root: Literal["runtime", "models", "rag"]
    archive_root: str
    expanded_size: int = Field(gt=0, le=_MAX_INT64, strict=True)
    install_size: int = Field(gt=0, le=_MAX_INT64, strict=True)
    peak_space: int = Field(gt=0, le=_MAX_INT64, strict=True)
    entrypoint: str
    healthcheck: tuple[str, ...]
    compatibility: Compatibility
    licenses: tuple[str, ...]

    @model_validator(mode="after")
    def validate_paths(self) -> "DistributionComponent":
        for value in (self.path, self.archive_root, self.entrypoint, *self.licenses):
            _validate_relative_path(value)
        if not self.healthcheck:
            raise ValueError("healthcheck must not be empty")
        if not self.licenses:
            raise ValueError("licenses must not be empty")
        if self.sha256 == "0" * 64 or self.manifest_sha256 == "0" * 64:
            raise ValueError("sha256 must not be all zeroes")
        if (self.package_type == "runtime_archive") != (self.install_root == "runtime"):
            raise ValueError("runtime archives must use the runtime install root")
        return self


class ImmutableArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repo: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    revision: str = Field(pattern=_REVISION)
    path: str
    size: int = Field(gt=0, le=_MAX_INT64, strict=True)
    sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def validate_immutable_artifact(self) -> "ImmutableArtifact":
        _validate_relative_path(self.path)
        if self.sha256 == "0" * 64:
            raise ValueError("artifact sha256 must not be all zeroes")
        return self


class SignedInstaller(ImmutableArtifact):
    publisher: str = Field(min_length=1)
    install_size: int = Field(gt=0, le=_MAX_INT64, strict=True)


class OllamaVerifier(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["qwen3:4b"]
    source: ImmutableArtifact
    installed_digest: str = Field(pattern=_SHA256)
    installed_size: int = Field(gt=0, le=_MAX_INT64, strict=True)

    @model_validator(mode="after")
    def validate_installed_digest(self) -> "OllamaVerifier":
        if self.installed_digest == "0" * 64:
            raise ValueError("verifier installed digest must not be all zeroes")
        return self


class SpeakerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[A-Za-z0-9_.-]+(?::[A-Za-z0-9_.-]+)?$")
    gguf: ImmutableArtifact
    modelfile: ImmutableArtifact
    installed_size: int = Field(gt=0, le=_MAX_INT64, strict=True)


class OllamaModelRoles(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    verifier: OllamaVerifier
    speaker: SpeakerModel


class OllamaPublication(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    installer: SignedInstaller
    release_version: str = Field(pattern=_SEMVER)
    release_identity: str = Field(min_length=1)
    models: OllamaModelRoles

    @model_validator(mode="after")
    def validate_immutable_installer(self) -> "OllamaPublication":
        if self.installer.publisher != "Ollama, Inc.":
            raise ValueError("Ollama publisher is invalid")
        return self


class WebView2Publication(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    installer: SignedInstaller

    @model_validator(mode="after")
    def validate_publisher(self) -> "WebView2Publication":
        if self.installer.publisher != "Microsoft Corporation":
            raise ValueError("WebView2 publisher is invalid")
        return self


class DistributionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, serialize_by_alias=True)

    schema_: Literal["youtuber.distribution.v1"] = Field(alias="schema")
    release: str = Field(pattern=_SEMVER)
    minimum: MinimumRequirements
    ollama: OllamaPublication
    webview2: WebView2Publication
    components: Mapping[DistributionComponentName, DistributionComponent]

    @property
    def schema(self) -> Literal["youtuber.distribution.v1"]:
        """Public schema name while avoiding Pydantic's deprecated class method."""
        return self.schema_

    @model_validator(mode="after")
    def validate_runtime_api(self) -> "DistributionManifest":
        if not self.components:
            raise ValueError("distribution manifest must contain components")
        expected_classification = {
            "studio_runtime": ("runtime_archive", "runtime"),
            "voice_runtime": ("runtime_archive", "runtime"),
            "rag": ("data_archive", "rag"),
            "llm": ("data_archive", "models"),
            "voice": ("data_archive", "models"),
            "embedder": ("data_archive", "models"),
            "reranker": ("data_archive", "models"),
        }
        for kind, component in self.components.items():
            if component.compatibility.runtime_api != self.minimum.runtime_api:
                raise ValueError("component runtime API must match distribution minimum")
            if kind not in expected_classification or (component.package_type, component.install_root) != expected_classification[kind]:
                raise ValueError(f"component kind {kind!r} has an invalid package classification")
        archive_roots = [_windows_path_key(component.archive_root) for component in self.components.values()]
        if len(archive_roots) != len(set(archive_roots)):
            raise ValueError("distribution manifest contains duplicate Windows install paths")
        try:
            total = self.minimum.disk_bytes
            total += self.ollama.installer.size + self.ollama.installer.install_size
            total += self.webview2.installer.size + self.webview2.installer.install_size
            total += self.ollama.models.verifier.source.size + self.ollama.models.verifier.installed_size
            total += self.ollama.models.speaker.gguf.size + self.ollama.models.speaker.modelfile.size
            total += self.ollama.models.speaker.installed_size
            if total > _MAX_INT64:
                raise OverflowError
            for component in self.components.values():
                total += component.size + component.expanded_size + component.install_size + component.peak_space
                if total > _MAX_INT64:
                    raise OverflowError
        except OverflowError as error:
            raise ValueError("distribution byte requirements overflow Int64") from error
        object.__setattr__(self, "components", ImmutableDict(self.components))
        return self
