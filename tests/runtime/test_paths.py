from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pipeline.config import load_settings
from runtime.paths import RuntimePaths
from runtime.settings import RuntimeSettings


def _runtime_settings(root: Path) -> RuntimeSettings:
    return RuntimeSettings.model_validate({
        "install_root": root / "install",
        "data_root": root / "data",
        "cache_root": root / "cache",
        "studio_port": 49152,
        "rag_port": 49153,
        "voice_port": 49154,
        "ollama_origin": "http://127.0.0.1:49155",
        "answer_model": "speaker-v5-a636",
        "verifier_model": "qwen3:4b",
        "session_secret": "a" * 64,
    })


def _config(path: Path) -> None:
    path.write_text(textwrap.dedent("""
        paths:
          data_dir: data
          audio_dir: data/audio
          meta_dir: data/meta
          vad_dir: data/vad
          diarize_dir: data/diarize
          identify_dir: data/identify
          transcribe_dir: data/transcribe
          filter_dir: data/filter
          clean_dir: data/clean
          dataset_dir: data/dataset
          reference_clip: data/reference/speaker_reference.wav
          review_dir: data/review
          decisions_db: data/decisions.sqlite
          logs_dir: logs
        rag:
          store_path: data/rag/qdrant
          card_catalog_path: data/cards/speaker_cards.json
          card_manifest_path: data/cards/speaker_cards.index_manifest.json
    """), encoding="utf-8")


def test_runtime_paths_resolve_all_packaged_assets_under_launcher_roots(tmp_path):
    settings = _runtime_settings(tmp_path)
    paths = RuntimePaths.from_settings(settings)

    assert paths.config_path == (tmp_path / "install/runtime/config.yaml").resolve()
    assert paths.clean_corpus == (tmp_path / "data/clean").resolve()
    assert paths.cards_dir == (tmp_path / "data/cards").resolve()
    assert paths.qdrant_dir == (tmp_path / "data/rag/qdrant").resolve()
    assert paths.xtts_assets_dir == (tmp_path / "data/models/voice/xtts").resolve()
    assert paths.rvc_weight == (tmp_path / "data/models/voice/rvc/speaker-e200.pth").resolve()
    assert paths.rvc_index == (tmp_path / "data/models/voice/rvc/speaker-e200.index").resolve()
    assert paths.temporary_audio_dir == (tmp_path / "cache/generated-audio").resolve()
    assert paths.logs_dir == (tmp_path / "cache/logs").resolve()
    assert paths.logs_dir.is_relative_to(paths.cache_root)


def test_runtime_paths_rejects_escape_from_configured_root(tmp_path):
    paths = RuntimePaths.from_settings(_runtime_settings(tmp_path))

    with pytest.raises(ValueError, match="escapes"):
        paths.data_path(Path("..") / "outside")


def test_config_loader_resolves_developer_and_packaged_layouts(tmp_path):
    developer_root = tmp_path / "developer"
    developer_root.mkdir()
    developer_config = developer_root / "config.yaml"
    _config(developer_config)
    developer_paths = RuntimePaths.developer(developer_root)

    developer = load_settings(developer_config, runtime_paths=developer_paths)

    packaged_root = tmp_path / "packaged"
    packaged_settings = _runtime_settings(packaged_root)
    packaged_paths = RuntimePaths.from_settings(packaged_settings)
    packaged_paths.config_path.parent.mkdir(parents=True)
    _config(packaged_paths.config_path)
    packaged = load_settings(runtime_paths=packaged_paths)

    assert developer.paths.clean_dir == (developer_root / "data/clean").resolve()
    assert developer.rag.store_path == (developer_root / "data/rag/qdrant").resolve()
    assert packaged.paths.clean_dir == packaged_paths.clean_corpus
    assert packaged.rag.store_path == packaged_paths.qdrant_dir
    assert packaged.rag.card_catalog_path == packaged_paths.card_catalog_path
