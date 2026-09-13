"""Config loader for all pipeline stages.

Reads ``config.yaml`` at the repo root by default. Any value can be overridden
via env vars prefixed with ``YOUTUBER_`` using ``__`` as the section separator,
e.g. ``YOUTUBER_IDENTIFY__SIMILARITY_THRESHOLD=0.7``.

Stages should call ``load_settings()`` rather than reading config.yaml directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from runtime.paths import RuntimePaths
from runtime.settings import RuntimeSettings


# Backward-compatible developer root for pipeline stages that own repository-local
# caches and reference-pool metadata. Runtime-facing paths continue to resolve
# through ``RuntimePaths`` below.
REPO_ROOT = Path(__file__).resolve().parent.parent


class PathsConfig(BaseModel):
    data_dir: Path
    audio_dir: Path
    meta_dir: Path
    vad_dir: Path
    diarize_dir: Path
    identify_dir: Path
    transcribe_dir: Path
    filter_dir: Path
    clean_dir: Path = Path("data/clean")
    dataset_dir: Path
    reference_clip: Path
    review_dir: Path
    decisions_db: Path
    logs_dir: Path


class DownloadConfig(BaseModel):
    channel_url: str | None = None
    video_urls: list[str] = Field(default_factory=list)
    audio_format: str = "wav"
    audio_quality: str = "0"
    sample_rate_hz: int = 16000
    channels: int = 1
    max_videos: int = 10
    # Path to a Netscape-format cookies.txt for yt-dlp. Needed when YouTube bot-gates the
    # request (e.g. datacenter IPs like Colab). Unset -> auto-detect a few known locations
    # (see download.runner._resolve_cookiefile); none existing -> run without cookies.
    cookies_file: str | None = None
    # Anti-bot-gate throttling: randomized pause (seconds) before each video download so a
    # bulk drain looks less like a scraper. yt-dlp sleeps a random value in
    # [sleep_interval_s, max_sleep_interval_s]. 0 = off (default; existing behavior). Raise
    # these (e.g. 10/30) when a datacenter IP like Colab is getting rate-limited.
    sleep_interval_s: float = 0.0
    max_sleep_interval_s: float = 0.0


class VADConfig(BaseModel):
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 100
    speech_pad_ms: int = 30


class DiarizeConfig(BaseModel):
    # Stage-3 method. "diarization" = full pyannote (segmentation + embedding + single-threaded
    # CPU AgglomerativeClustering — the CPU bottleneck). "segmentation" = clustering-free,
    # segmentation-3.0 only (GPU), no global speaker labels. Downstream (Stage 4 identify) uses
    # only segment (start,end) + its own per-segment ECAPA similarity, so the cluster labels are
    # unused metadata — "segmentation" escapes the CPU bottleneck at the cost of collapsing
    # speakers (a segment may straddle a turn when there's no pause; the similarity gate still
    # drops mostly-not-him audio).
    method: str = "diarization"
    model: str = "pyannote/speaker-diarization-3.1"
    segmentation_model: str = "pyannote/segmentation-3.0"
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    # Bigger batches saturate the GPU and cut per-window CPU glue in the segmentation/
    # embedding phases. Lower these if the GPU OOMs on a small card.
    segmentation_batch_size: int = 256
    embedding_batch_size: int = 256
    # VAD post-processing on the segmentation activations (only used when method="segmentation"):
    seg_min_duration_on: float = 0.0     # drop speech islands shorter than this (0 = keep all)
    seg_min_duration_off: float = 0.25   # bridge silences shorter than this so breaths don't
                                         # fragment his speech; real turn-pauses still split it
    seg_max_segment_s: float = 28.0      # split longer speech regions (stay under
                                         # filter.max_segment_s so long monologue isn't dropped)


class IdentifyConfig(BaseModel):
    embedding_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    similarity_threshold: float = 0.65
    borderline_low: float = 0.50
    borderline_high: float = 0.70
    # Sliding-window sub-segmentation: slide a short ECAPA window across each segment and
    # keep only the contiguous spans matching the reference, so guest-only spans inside a
    # speaker-mixed segment (from the clustering-free Stage 3) are excised. Segments shorter
    # than window_s use a single whole-segment embed (legacy behavior). Disable by setting
    # window_s larger than diarize.seg_max_segment_s.
    window_s: float = 1.5             # ECAPA needs ~>=1s for a stable embedding
    hop_s: float = 0.75               # 50% overlap so each turn boundary is covered twice
    win_keep_threshold: float | None = None  # per-window keep gate; None -> similarity_threshold
    bridge_max_dip_windows: int = 1   # flip a single dropped window flanked by keeps (transition)
    min_keep_span_s: float = 2.0      # drop coalesced spans shorter than this (matches the 2.0s
                                      # transcribe/filter floor so we don't emit un-transcribable spans)


class TranscribeConfig(BaseModel):
    model: str = "large-v3"
    language: str | None = None
    compute_type: str = "float16"
    batch_size: int = 8
    word_timestamps: bool = True


class FilterConfig(BaseModel):
    min_avg_logprob: float = -1.0
    min_segment_s: float = 2.0
    max_segment_s: float = 30.0
    max_words_per_second: float = 6.0
    drop_patterns: list[str] = Field(default_factory=list)
    # In-place ASR-artifact strip (delete the match, KEEP surrounding speech) — distinct from
    # drop_patterns (whole-segment drop). For artifacts spliced mid-sentence (e.g. the
    # "Altyazı M.K." subtitle credit). A segment that becomes empty after scrubbing is dropped.
    # Validated zero-false-positive over the corpus (see known_transcription_corrections memory).
    scrub_patterns: list[str] = Field(default_factory=list)


class CleanConfig(BaseModel):
    """Stage 6.5 — text-level contamination cleaning of filter output (no re-transcribe).

    Removes read-aloud third-party prose (datelines, newswire copy, reproduced statements)
    that speaker-ID correctly KEPT — it really is his voice — but which is not his style.
    Also applies verified ASR corrections in place. PROFANITY IS NEVER DROPPED: drop_patterns
    are structural (dateline/newswire/credit) only, never content/insult words.

    NOTE on the missing sim-gate: a corpus estimate showed ~99% of post-identify segments sit
    in the 0.40-0.70 ECAPA band (his authentic voice averages ~0.52, rarely >=0.70) — so a
    similarity gate here is non-discriminating and per-segment LLM would be ~179k calls. Guest
    contamination is instead handled by a VIDEO-LEVEL triage (title_flags + drop-ratio), with
    LLM adjudication scoped to flagged interview videos only — built as a later step.
    """

    # Whole-segment structural DROP patterns (regex). For read-aloud datelines/newswire/
    # boilerplate. Precision-biased: validate zero-false-positive on the corpus before trusting.
    drop_patterns: list[str] = Field(default_factory=list)
    # In-place verified ASR corrections {wrong: right}; each 'wrong' confirmed never legitimate.
    corrections: dict[str, str] = Field(default_factory=dict)
    # Lowercased Turkish title keywords that flag interview/panel/co-host videos for the
    # (later) guest-contamination review pass.
    title_flags: list[str] = Field(default_factory=list)
    # Flag a video for review when its structural drop ratio exceeds this (interview-heavy).
    max_drop_ratio_flag: float = 0.5
    # Write data/clean/{vid}.review.json (dropped spans + reasons) for spot-checking.
    emit_review: bool = True


class ReviewConfig(BaseModel):
    batch_size: int = 75
    context_neighbors: int = 2


class AssembleConfig(BaseModel):
    llm_skip_marker: str = "[SKIP]"
    skip_gap_s: float = 1.0
    rag_chunk_tokens: int = 500
    rag_chunk_overlap_tokens: int = 50


class TTSConfig(BaseModel):
    min_segment_s: float = 3.0
    max_segment_s: float = 15.0


class RagConfig(BaseModel):
    source: str = "clean"
    embedder: str = "BAAI/bge-m3"
    reranker: str = "BAAI/bge-reranker-v2-m3"
    store_path: Path = Path("data/rag/qdrant")
    collection: str = "speaker_clean"
    evidence_collection: str = "speaker_clean"
    retrieval_mode: Literal["clean", "clean_with_card_hints"] = "clean_with_card_hints"
    card_collection: str = "speaker_cards"
    card_catalog_path: Path = Path("data/cards/speaker_cards.json")
    card_manifest_path: Path = Path("data/cards/speaker_cards.index_manifest.json")
    card_hint_top_n: int = 2
    card_hint_min_score: float = -1.0
    original_candidate_n: int = 8
    hint_candidate_n: int = 2
    verifier_candidate_max: int = 12
    verifier_span_char_cap: int = 1800
    verifier_span_seconds_cap: float = 180.0
    answer_span_max: int = 5
    answer_span_char_cap: int = 1800
    answer_context_char_cap: int = 9000
    verifier_model: str = "qwen3:4b"
    verifier_temperature: float = 0.0
    verifier_seed: int = 42
    verifier_num_ctx: int = 8192
    verifier_num_predict: int = 1200
    verifier_timeout_s: int = 120
    chunk_tokens: int = 500
    chunk_overlap: int = 50
    retrieve_top_k: int = 20
    rerank_top_n: int = 3
    # Source-diversity cap on reranked hits (0/None = off): at most this many chunks of any one
    # video in the top-N, so one video can't supply the whole context block.
    max_per_video: int = 2
    gate_threshold: float = 0.0
    ollama_model: str = "speaker-v5-a636"
    gen_temperature: float = 0.78
    gen_top_p: float = 0.85
    gen_repeat_penalty: float = 1.3
    gen_num_predict: int = 256


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="YOUTUBER_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    paths: PathsConfig
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    diarize: DiarizeConfig = Field(default_factory=DiarizeConfig)
    identify: IdentifyConfig = Field(default_factory=IdentifyConfig)
    transcribe: TranscribeConfig = Field(default_factory=TranscribeConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    clean: CleanConfig = Field(default_factory=CleanConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    assemble: AssembleConfig = Field(default_factory=AssembleConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    rag: RagConfig = Field(default_factory=RagConfig)

    def resolve_paths(self, runtime_paths: RuntimePaths) -> None:
        """Resolve developer paths or apply the packaged data-root contract."""
        if runtime_paths.packaged:
            self.paths.data_dir = runtime_paths.data_root
            self.paths.audio_dir = runtime_paths.data_path(Path("audio"))
            self.paths.meta_dir = runtime_paths.data_path(Path("meta"))
            self.paths.vad_dir = runtime_paths.data_path(Path("vad"))
            self.paths.diarize_dir = runtime_paths.data_path(Path("diarize"))
            self.paths.identify_dir = runtime_paths.data_path(Path("identify"))
            self.paths.transcribe_dir = runtime_paths.data_path(Path("transcribe"))
            self.paths.filter_dir = runtime_paths.data_path(Path("filter"))
            self.paths.clean_dir = runtime_paths.clean_corpus
            self.paths.dataset_dir = runtime_paths.data_path(Path("dataset"))
            self.paths.reference_clip = runtime_paths.data_path(Path("reference/speaker_reference.wav"))
            self.paths.review_dir = runtime_paths.data_path(Path("review"))
            self.paths.decisions_db = runtime_paths.data_path(Path("decisions.sqlite"))
            self.paths.logs_dir = runtime_paths.logs_dir
            self.rag.store_path = runtime_paths.qdrant_dir
            self.rag.card_catalog_path = runtime_paths.card_catalog_path
            self.rag.card_manifest_path = runtime_paths.card_manifest_path
            return
        for field in type(self.paths).model_fields:
            value: Path = getattr(self.paths, field)
            if not value.is_absolute():
                setattr(self.paths, field, (runtime_paths.code_root / value).resolve())
        for field in ("store_path", "card_catalog_path", "card_manifest_path"):
            value: Path = getattr(self.rag, field)
            if not value.is_absolute():
                setattr(self.rag, field, (runtime_paths.code_root / value).resolve())


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_settings(
    config_path: Path | str | None = None,
    *,
    runtime_paths: RuntimePaths | None = None,
) -> Settings:
    """Load configuration through the active developer or packaged runtime layout."""
    runtime_paths = runtime_paths or RuntimePaths.from_settings(RuntimeSettings.from_environment())
    path = Path(config_path).resolve(strict=False) if config_path else runtime_paths.config_path
    if runtime_paths.packaged and path != runtime_paths.config_path:
        raise ValueError("packaged runtime config must be under YOUTUBER_INSTALL_ROOT/runtime")
    data = _read_yaml(path)
    settings = Settings(**data)
    settings.resolve_paths(runtime_paths)
    return settings
