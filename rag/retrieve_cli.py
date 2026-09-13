"""Retrieval bridge: query -> JSON hits on stdout, so the .venv-tts UI can call the .venv RAG
stack by subprocess (the two environments can't share a process: FlagEmbedding/qdrant live in
.venv, coqui-tts lives in .venv-tts).

  .venv\\Scripts\\python.exe -m rag.retrieve_cli "soru" --top-n 6
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

import requests

from pipeline.config import load_settings
from rag.prompt import gate
from rag.retrieve import retrieve
from runtime.settings import RuntimeSettings


_LOOPBACK_SESSION = requests.Session()
_LOOPBACK_SESSION.trust_env = False


def retrieve_packaged(
    query: str,
    mode: str,
    settings: Any,
    *,
    get: Callable[..., object] = _LOOPBACK_SESSION.get,
) -> dict[str, object]:
    """Use the supervised loopback RAG service in a packaged runtime."""
    if not settings.session_secret:
        raise RuntimeError("packaged retrieval requires a session secret")
    response = get(
        f"http://127.0.0.1:{settings.rag_port}/retrieve",
        params={"q": query, "mode": mode},
        headers={"Authorization": "Bearer " + settings.session_secret},
        timeout=660,
        allow_redirects=False,
    )
    response.raise_for_status()
    payload = response.json()
    if type(payload) is not dict:
        raise RuntimeError("packaged retrieval response is invalid")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top-n", type=int, default=None)
    ap.add_argument("--collection", default=None, help="override config collection (e.g. speaker_clean)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    runtime = RuntimeSettings.from_environment()
    if runtime.packaged:
        mode = "clean_with_card_hints" if args.collection == "speaker_cards" else "clean"
        print(json.dumps(retrieve_packaged(args.query, mode, runtime), ensure_ascii=False))
        return

    s = load_settings(args.config)
    r = s.rag
    coll = args.collection or r.collection
    hits = retrieve(args.query, store_path=str(r.store_path), collection=coll,
                    embedder_name=r.embedder, reranker_name=r.reranker,
                    top_k=r.retrieve_top_k, top_n=args.top_n or r.rerank_top_n,
                    max_per_video=r.max_per_video)
    dec = gate(hits, threshold=r.gate_threshold, ungrounded=False)
    out = {
        "grounded": dec.grounded,
        "threshold": r.gate_threshold,
        "hits": [{"text": h.chunk.text, "video_id": h.chunk.video_id, "title": h.chunk.title,
                  "start": h.chunk.start, "score": round(h.rerank_score, 3)} for h in hits],
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
