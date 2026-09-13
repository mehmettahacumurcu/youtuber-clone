"""Resolved, containment-checked paths for a runtime process."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from runtime.settings import RuntimeSettings


@dataclass(frozen=True)
class RuntimePaths:
    packaged: bool
    code_root: Path
    install_root: Path
    data_root: Path
    cache_root: Path
    config_path: Path
    clean_corpus: Path
    cards_dir: Path
    qdrant_dir: Path
    voice_root: Path
    xtts_assets_dir: Path
    rvc_weight: Path
    rvc_index: Path
    temporary_audio_dir: Path
    logs_dir: Path
    reference_wavs_dir: Path
    card_catalog_path: Path
    card_manifest_path: Path

    @classmethod
    def developer(cls, root: Path | None = None) -> "RuntimePaths":
        root = (root or Path(__file__).resolve().parents[1]).resolve(strict=False)
        return cls._build(False, root, root, root / "data", root / "data")

    @classmethod
    def from_settings(cls, settings: RuntimeSettings) -> "RuntimePaths":
        if not settings.packaged:
            return cls.developer()
        assert settings.install_root and settings.data_root and settings.cache_root
        return cls._build(True, settings.install_root, settings.install_root, settings.data_root, settings.cache_root)

    @classmethod
    def _build(cls, packaged: bool, code_root: Path, install_root: Path, data_root: Path, cache_root: Path) -> "RuntimePaths":
        code_root = code_root.resolve(strict=False)
        install_root = install_root.resolve(strict=False)
        data_root = data_root.resolve(strict=False)
        cache_root = cache_root.resolve(strict=False)
        config_path = cls._under(install_root, Path("runtime/config.yaml")) if packaged else cls._under(code_root, Path("config.yaml"))
        voice_root = cls._under(data_root, Path("models/voice")) if packaged else cls._under(code_root, Path("models"))
        return cls(
            packaged, code_root, install_root, data_root, cache_root, config_path,
            cls._under(data_root, Path("clean")),
            cls._under(data_root, Path("cards")),
            cls._under(data_root, Path("rag/qdrant")),
            voice_root,
            cls._under(voice_root, Path("xtts")) if packaged else voice_root,
            cls._under(voice_root, Path("rvc/speaker-e200.pth")),
            cls._under(voice_root, Path("rvc/speaker-e200.index")),
            cls._under(cache_root, Path("generated-audio")),
            cls._under(cache_root, Path("logs")) if packaged else cls._under(code_root, Path("logs")),
            cls._under(voice_root, Path("references")) if packaged else cls._under(data_root, Path("tts/wavs")),
            cls._under(data_root, Path("cards/speaker_cards.json")),
            cls._under(data_root, Path("cards/speaker_cards.index_manifest.json")),
        )

    @staticmethod
    def _under(root: Path, relative: Path) -> Path:
        resolved_root = root.resolve(strict=False)
        candidate = (resolved_root / relative).resolve(strict=False)
        if not candidate.is_relative_to(resolved_root):
            raise ValueError(f"path {relative} escapes configured root {root}")
        return candidate

    def data_path(self, relative: Path) -> Path:
        return self._under(self.data_root, relative)

    def cache_path(self, relative: Path) -> Path:
        return self._under(self.cache_root, relative)
