from pathlib import Path

from pipeline.config import RagConfig, load_settings


def test_rag_config_loads_with_defaults():
    s = load_settings()
    assert s.rag.embedder == "BAAI/bge-m3"
    assert s.rag.reranker == "BAAI/bge-reranker-v2-m3"
    assert s.rag.collection == "speaker_clean"
    assert s.rag.retrieve_top_k == 20
    assert s.rag.rerank_top_n == 3
    assert s.rag.source in ("clean", "filter")
    assert s.rag.ollama_model == "speaker-v5-a636"


def test_rag_store_path_resolves_absolute():
    s = load_settings()
    assert str(s.rag.store_path).endswith("qdrant")


def test_evidence_verifier_defaults_are_exact():
    defaults = RagConfig()
    r = load_settings().rag
    assert r.evidence_collection == "speaker_clean"
    assert r.retrieval_mode == "clean_with_card_hints"
    assert r.card_collection == "speaker_cards"
    assert r.card_hint_top_n == 2
    assert r.card_hint_min_score == -1.0
    assert r.original_candidate_n == 8
    assert r.hint_candidate_n == 2
    assert r.verifier_candidate_max == 12
    assert r.verifier_span_char_cap == 1800
    assert r.verifier_span_seconds_cap == 180.0
    assert r.answer_span_max == 5
    assert r.answer_span_char_cap == 1800
    assert r.answer_context_char_cap == 9000
    assert defaults.verifier_model == "qwen3:4b"
    assert defaults.verifier_num_ctx == 8192
    assert defaults.verifier_num_predict == 1200
    assert defaults.verifier_timeout_s == 120
    assert r.verifier_model == "qwen3:4b"
    assert r.verifier_temperature == 0.0
    assert r.verifier_seed == 42
    assert r.verifier_num_ctx == 8192
    assert r.verifier_num_predict == 1200
    assert r.verifier_timeout_s == 120


def test_card_paths_resolve_from_repository_root():
    r = load_settings().rag
    assert r.card_catalog_path.is_absolute()
    assert r.card_manifest_path.is_absolute()
    assert r.card_catalog_path == Path(__file__).resolve().parents[2] / "data/cards/speaker_cards.json"
    assert r.card_manifest_path == (
        Path(__file__).resolve().parents[2] / "data/cards/speaker_cards.index_manifest.json"
    )


def test_legacy_low_level_fields_remain_available():
    r = load_settings().rag
    assert r.collection == "speaker_clean"
    assert r.max_per_video == 2
    assert r.gate_threshold == 0.0
