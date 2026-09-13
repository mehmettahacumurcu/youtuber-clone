"""Audited dynamic imports used by BGE-M3 embedding and reranking."""
from PyInstaller.utils.hooks import collect_data_files


hiddenimports = [
    "FlagEmbedding.inference.embedder.encoder_only.base",
    "FlagEmbedding.inference.embedder.encoder_only.m3",
    "FlagEmbedding.inference.reranker.encoder_only.base",
]
datas = collect_data_files("FlagEmbedding", include_py_files=True)
