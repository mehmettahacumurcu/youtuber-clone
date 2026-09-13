"""Persistent schema-v2 evidence server; owns the local Qdrant reader lock."""
from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import math
import os
import threading
import traceback
import urllib.parse
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Literal

from pipeline.config import load_settings
from rag.evidence_pipeline import EvidenceDecision, assess_question
from rag.readiness import approved_identity
from rag.verifier import EvidenceVerifier
from runtime.settings import RuntimeSettings


PORT = RuntimeSettings.from_environment().rag_port
SCHEMA_VERSION = 2
RUNTIME_API = 1
_S = load_settings()
_R = _S.rag
_STATE: dict[str, object] = {"status": "warming", "error": None}
_IDENTITY: dict[str, object] | None = None


def _is_loopback(host: str | None) -> bool:
    try:
        return host is not None and ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def require_rag_bearer(header: str | None, secret: str | None, *, packaged: bool) -> bool:
    """Packaged RAG accepts only the launch bearer; development remains ergonomic."""
    if not packaged:
        return True
    expected = "Bearer " + (secret or "")
    return bool(secret) and isinstance(header, str) and hmac.compare_digest(header, expected)


def build_ready_payload(
    state: Mapping[str, object],
    *,
    corpus_version: str,
    index_version: str,
    embedder: str,
    reranker: str,
    corpus_fingerprint: str | None = None,
    index_fingerprint: str | None = None,
    card_fingerprint: str | None = None,
    clean_count: int | None = None,
    card_count: int | None = None,
) -> dict[str, object]:
    """Return only compatibility identities; never expose paths, corpus text, or secrets."""
    if state.get("status") != "ready":
        return {"status": "unready", "runtime_api": RUNTIME_API, "schema_version": SCHEMA_VERSION}
    payload: dict[str, object] = {
        "status": "ready", "runtime_api": RUNTIME_API, "schema_version": SCHEMA_VERSION,
        "corpus_version": corpus_version, "index_version": index_version,
        "embedder": embedder, "reranker": reranker,
    }
    optional = {
        "corpus_fingerprint": corpus_fingerprint, "index_fingerprint": index_fingerprint,
        "card_fingerprint": card_fingerprint, "clean_count": clean_count, "card_count": card_count,
    }
    payload.update({key: value for key, value in optional.items() if value is not None})
    return payload


def _ready_payload() -> dict[str, object]:
    if _IDENTITY is None:
        return {"status": "unready", "runtime_api": RUNTIME_API, "schema_version": SCHEMA_VERSION}
    return build_ready_payload(_STATE, **_IDENTITY)


def _probe_retrieval_models(embedder: object, reranker: object) -> None:
    """Exercise the inference paths required by the first real request."""
    encoded = embedder.encode(["readiness probe"])
    dense = getattr(encoded, "dense", ())
    sparse = getattr(encoded, "sparse", ())
    if len(dense) != 1 or len(sparse) != 1:
        raise RuntimeError("embedder readiness probe returned an invalid batch")
    scores = reranker.score("readiness probe", ["readiness probe"])
    if len(scores) != 1 or not math.isfinite(float(scores[0])):
        raise RuntimeError("reranker readiness probe returned an invalid score")


def _warm() -> None:
    global _IDENTITY
    try:
        _IDENTITY = approved_identity(_S)
        from rag.embed import get_embedder, get_reranker

        embedder = get_embedder(_R.embedder)
        reranker = get_reranker(_R.reranker)
        _probe_retrieval_models(embedder, reranker)
        _STATE["status"] = "ready"
        print(f"retrieve_server: READY on http://127.0.0.1:{PORT}", flush=True)
    except Exception as exc:
        _STATE["status"] = "error"
        _STATE["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        print(f"retrieve_server: WARMUP FAILED — {_STATE['error']}", flush=True)


def resolve_retrieval_mode(
    params: Mapping[str, list[str]],
    default_mode: str,
) -> tuple[Literal["clean", "clean_with_card_hints"], dict[str, object]]:
    diagnostics: dict[str, object] = {}
    explicit = params.get("mode", [None])[0]
    collection = params.get("collection", [None])[0]
    collection_mode = None
    if collection is not None:
        aliases = {
            "speaker_clean": "clean",
            "speaker_cards": "clean_with_card_hints",
        }
        if collection not in aliases:
            raise ValueError(f"unknown legacy collection {collection!r}")
        collection_mode = aliases[collection]
        diagnostics["deprecated_collection"] = collection
    mode = explicit or collection_mode or default_mode
    if mode not in {"clean", "clean_with_card_hints"}:
        raise ValueError(f"unknown retrieval mode {mode!r}")
    if explicit is not None and collection_mode is not None and explicit != collection_mode:
        raise ValueError("mode and collection select different retrieval policies")
    if "top_n" in params:
        diagnostics["ignored_top_n"] = params["top_n"][0]
    return mode, diagnostics


def decision_to_response(
    decision: EvidenceDecision,
    *,
    compatibility_diagnostics: Mapping[str, object] | None = None,
) -> dict[str, object]:
    evidence = [
        {
            "id": span.id,
            "video_id": span.video_id,
            "title": span.title,
            "start": span.start,
            "end": span.end,
            "text": span.text,
            "retrieval_score": span.retrieval_score,
            "score": span.retrieval_score,
            "source_chunk_ids": list(span.source_chunk_ids),
            "query_origins": list(span.query_origins),
        }
        for span in decision.selected_spans
    ]
    hints = [
        {
            "card_id": hint.card_id,
            "question": hint.question,
            "take": hint.take,
            "retrieval_score": hint.retrieval_score,
        }
        for hint in decision.hint_hits
    ]
    diagnostics = dict(decision.diagnostics)
    diagnostics.update(compatibility_diagnostics or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "status": decision.status,
        "grounded": decision.status in {"answerable", "partial"},
        "failure_reason": decision.failure_reason,
        "claims": [claim.model_dump(mode="json") for claim in decision.claims],
        "evidence_hits": evidence,
        "hint_hits": hints,
        "answer_question": decision.answer_question,
        "unsupported_claims": list(decision.unsupported_claims),
        "diagnostics": diagnostics,
        "hits": evidence,
        "expanded_video": None,
        "expanded_title": None,
        "expanded_spans": [],
        "threshold": None,
    }


def _error_payload(message: str, *, phase: str) -> dict[str, object]:
    decision = EvidenceDecision(
        status="error", claims=(), selected_spans=(), hint_hits=(), answer_question=None,
        unsupported_claims=(), failure_reason="retrieval_error",
        diagnostics={"phase": phase, "error": message},
    )
    return decision_to_response(decision)


def handle_retrieve(
    params: Mapping[str, list[str]],
    *,
    state: Mapping[str, object],
    default_mode: str,
    assess_fn: Callable[[str, str], EvidenceDecision],
) -> tuple[int, dict[str, object]]:
    if state.get("status") == "warming":
        return 503, _error_payload("retrieval service is warming", phase="warming")
    if state.get("status") == "error":
        return 500, _error_payload(str(state.get("error") or "warmup failed"), phase="warmup")
    question = params.get("q", [""])[0].strip()
    if not question:
        return 400, _error_payload("query parameter q must be non-empty", phase="request")
    try:
        mode, compatibility = resolve_retrieval_mode(params, default_mode)
    except ValueError as exc:
        return 400, _error_payload(str(exc), phase="request")
    try:
        decision = assess_fn(question, mode)
    except Exception as exc:
        return 500, _error_payload(f"{type(exc).__name__}: {exc}", phase="adapter")
    code = 500 if decision.status == "error" else 200
    return code, decision_to_response(
        decision, compatibility_diagnostics=compatibility,
    )


class Handler(BaseHTTPRequestHandler):
    def _json(self, value: object, code: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if not _is_loopback(self.client_address[0]):
            self._json({"detail": "loopback access required"}, 403)
            return
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/health/live":
            self._json({"status": "live", "runtime_api": RUNTIME_API})
            return
        settings = RuntimeSettings.from_environment()
        if not require_rag_bearer(
            self.headers.get("Authorization"), settings.session_secret, packaged=settings.packaged,
        ):
            self._json({"detail": "authentication required"}, 403)
            return
        if parsed.path == "/health/ready":
            payload = _ready_payload()
            self._json(payload, 200 if payload["status"] == "ready" else 503)
            return
        if parsed.path != "/retrieve":
            self._json(_error_payload("not found", phase="request"), 404)
            return
        params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)

        def assess(question: str, mode: str) -> EvidenceDecision:
            return assess_question(
                question,
                _S,
                verifier=EvidenceVerifier.from_config(_R),
                retrieval_mode=mode,
            )

        code, payload = handle_retrieve(
            params,
            state=_STATE,
            default_mode=_R.retrieval_mode,
            assess_fn=assess,
        )
        self._json(payload, code)

    def log_message(self, *args: object) -> None:
        return


def main(argv: list[str] | None = None) -> int:
    global _S, _R, PORT
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    _S = load_settings(args.config)
    _R = _S.rag
    PORT = args.port if args.port is not None else RuntimeSettings.from_environment().rag_port
    if not 1024 <= PORT <= 65535:
        raise SystemExit("RAG port must be between 1024 and 65535")
    if args.host != "127.0.0.1":
        raise SystemExit("RAG host must be 127.0.0.1")
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(
        f"retrieve_server: listening on http://127.0.0.1:{PORT} "
        f"(evidence={_R.evidence_collection}; mode={_R.retrieval_mode}; warming models)",
        flush=True,
    )
    threading.Thread(target=_warm, daemon=True).start()
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
