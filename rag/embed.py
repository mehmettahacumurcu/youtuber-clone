"""BGE-M3 (dense+sparse) embedder and bge-reranker-v2-m3 cross-encoder, lazily loaded."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from runtime.settings import RuntimeSettings


def _runtime_device(requested: str | None) -> str | None:
    """Packaged retrieval is CPU-only; developer callers may opt into a device."""
    return "cpu" if RuntimeSettings.from_environment().packaged else requested


@dataclass
class EmbedOut:
    dense: np.ndarray
    sparse: list[dict[int, float]]


class Embedder:
    def __init__(self, model_name: str, device: str | None = None):
        import datasets  # must load before FlagEmbedding to avoid pyarrow DLL ordering crash
        from FlagEmbedding import BGEM3FlagModel
        self.device = _runtime_device(device)
        kwargs = {"use_fp16": self.device != "cpu"}
        if self.device is not None:
            kwargs["devices"] = self.device
        self._m = BGEM3FlagModel(model_name, **kwargs)

    def encode(self, texts: list[str]) -> EmbedOut:
        out = self._m.encode(texts, return_dense=True, return_sparse=True,
                             return_colbert_vecs=False)
        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        sparse = [{int(k): float(v) for k, v in lw.items()}
                  for lw in out["lexical_weights"]]
        return EmbedOut(dense=dense, sparse=sparse)


class Reranker:
    def __init__(self, model_name: str, device: str | None = None):
        import datasets  # must load before FlagEmbedding to avoid pyarrow DLL ordering crash
        from FlagEmbedding import FlagReranker
        self.device = _runtime_device(device)
        kwargs = {"use_fp16": self.device != "cpu"}
        if self.device is not None:
            kwargs["devices"] = self.device
        self._m = FlagReranker(model_name, **kwargs)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        scores = self._m.compute_score(pairs, normalize=False)
        return [float(s) for s in (scores if isinstance(scores, list) else [scores])]


@lru_cache(maxsize=2)
def _get_embedder(model_name: str, effective_device: str | None) -> Embedder:
    return Embedder(model_name, effective_device)


def get_embedder(model_name: str, device: str | None = None) -> Embedder:
    """Return a cached embedder keyed by the device actually permitted at runtime."""
    return _get_embedder(model_name, _runtime_device(device))


@lru_cache(maxsize=2)
def _get_reranker(model_name: str, effective_device: str | None) -> Reranker:
    return Reranker(model_name, effective_device)


def get_reranker(model_name: str, device: str | None = None) -> Reranker:
    """Return a cached reranker keyed by the device actually permitted at runtime."""
    return _get_reranker(model_name, _runtime_device(device))


get_embedder.cache_clear = _get_embedder.cache_clear
get_reranker.cache_clear = _get_reranker.cache_clear
