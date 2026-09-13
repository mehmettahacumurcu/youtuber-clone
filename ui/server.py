"""Authenticated FastAPI host for the packaged Studio worker."""
from __future__ import annotations

import ipaddress
import os
import secrets
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.requests import HTTPConnection

from runtime.paths import RuntimePaths
from runtime.managed_rag_process import BoundedLog, managed_popen
from runtime.loopback_http import loopback_json
from runtime.session_auth import SessionAuth
from runtime.settings import RuntimeSettings, session_secret_bytes
from rag.prompt import REFUSAL_TR
from ui.chat_runtime import (
    DEFAULT_ANSWER_TOKEN_CAP,
    GenerationStatus,
    MAX_ANSWER_TOKEN_CAP,
    MIN_ANSWER_TOKEN_CAP,
    _EMPTY_OUTPUT_TR,
    _ERROR_TR,
)


RUNTIME_API = 1
WORKER_VERSION = "studio-v1"
ANSWER_MODE_CONTRACT_VERSION = 1
RAG_STARTUP_TIMEOUT_SECONDS = 120.0


class ChatRequest(BaseModel):
    """Strict authenticated Studio generation request."""

    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(min_length=1, max_length=8000)
    answer_mode: Literal["grounded_strict", "grounded_fallback", "free"]
    retrieval_mode: Literal["clean", "clean_with_card_hints"]
    answer_token_cap: int = Field(
        ge=MIN_ANSWER_TOKEN_CAP, le=MAX_ANSWER_TOKEN_CAP
    )


class ChatResponse(BaseModel):
    """Stable response contract exposed by the frozen Studio process."""

    model_config = ConfigDict(extra="forbid", strict=True)

    runtime_api: Literal[1]
    answer_mode_contract_version: Literal[1]
    answer_model: str
    answer_mode: Literal["grounded_strict", "grounded_fallback", "free"]
    retrieval_mode: Literal["clean", "clean_with_card_hints"]
    answer_token_cap: int
    answer: str
    context_markdown: str
    metadata: str
    grounding_status: Literal["answerable", "partial", "unsupported", "error", "free"]
    evidence_count: int = Field(ge=0)
    generation_status: Literal["generated"]


class MaintenanceReleaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    lease_token: str = Field(min_length=32, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")


class GroundingTrace(BaseModel):
    """Internal trace validated before it crosses the frozen API boundary."""

    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["answerable", "partial", "unsupported", "error", "free"]
    evidence_count: int = Field(ge=0)


class GenerationTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal[
        "generated",
        "empty_output",
        "model_error",
        "policy_refusal",
        "retrieval_error",
        "no_input",
    ]


_STOCK_NON_ANSWERS = frozenset({_EMPTY_OUTPUT_TR, _ERROR_TR, REFUSAL_TR})


def _require_generated_answer(answer: object, status: GenerationStatus) -> str:
    if (
        status != "generated"
        or type(answer) is not str
        or not answer.strip()
        or answer.strip() in _STOCK_NON_ANSWERS
        or answer.startswith("[LLM error:")
    ):
        raise RuntimeError("Studio answer model returned no usable answer")
    return answer


def _production_chat(
    question: str,
    model: str,
    answer_mode: str,
    retrieval_mode: str,
    answer_token_cap: int,
) -> tuple[str, str, str, Mapping[str, object], GenerationStatus]:
    """Run generation in the frozen Studio process, never in its controller."""
    from ui.chat_runtime import chat_turn_with_trace

    result = chat_turn_with_trace(
        question,
        [],
        model,
        answer_mode,
        retrieval_mode,
        answer_token_cap=answer_token_cap,
    )
    history = result.history
    context = result.context
    metadata = result.metadata
    answer = history[-1]["content"] if history else ""
    answer = _require_generated_answer(answer, result.generation_status)
    return answer, context, metadata, result.trace, result.generation_status


class RagHealthClient(Protocol):
    def ready(self) -> Mapping[str, object]: ...


class OllamaHealthClient(Protocol):
    def ready(self) -> Mapping[str, object]: ...


class _HttpRagHealthClient:
    def __init__(self, settings: RuntimeSettings) -> None:
        self._url = f"http://127.0.0.1:{settings.rag_port}/health/ready"
        self._headers = (
            {"Authorization": "Bearer " + settings.session_secret}
            if settings.session_secret else {}
        )

    def ready(self) -> Mapping[str, object]:
        return loopback_json(
            self._url,
            bearer=self._headers.get("Authorization", "").removeprefix("Bearer ") or None,
            timeout_s=2.0,
        )


class _HttpOllamaHealthClient:
    def __init__(self, settings: RuntimeSettings) -> None:
        self._settings = settings
        self._url = settings.ollama_origin + "/api/tags"

    def ready(self) -> Mapping[str, object]:
        if self._settings.packaged:
            answer_model = self._settings.answer_model
            verifier_model = self._settings.verifier_model
        else:
            from pipeline.config import load_settings

            configured = load_settings().rag
            answer_model = configured.ollama_model
            verifier_model = configured.verifier_model
        payload = loopback_json(self._url, timeout_s=2.0)
        models = payload.get("models") if type(payload) is dict else None
        if type(models) is not list:
            raise RuntimeError("Ollama readiness response is invalid")
        names = {item.get("name") for item in models if type(item) is dict}
        answer = next(
            (
                candidate
                for candidate in (answer_model, answer_model + ":latest")
                if candidate in names
            ),
            None,
        )
        if answer is None or verifier_model not in names:
            raise RuntimeError("required Ollama models are unavailable")
        return {"answer_model": answer, "verifier_model": verifier_model}


def _is_loopback(host: str | None) -> bool:
    try:
        return host is not None and ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _spawn_rag(settings: RuntimeSettings) -> Any:
    secret = session_secret_bytes(settings.session_secret, allow_none=not settings.packaged)
    redactions = () if secret is None else (secret,)
    environment = os.environ.copy()
    if settings.session_secret:
        environment["YOUTUBER_SESSION_SECRET"] = settings.session_secret
    from ui.__main__ import build_rag_command

    python = sys.executable if settings.packaged else os.environ.get("SPEAKER_RAG_PY", sys.executable)
    paths = RuntimePaths.from_settings(settings)
    stdout = BoundedLog(paths.logs_dir, "rag")
    stderr = BoundedLog(paths.logs_dir, "rag-error")
    return managed_popen(
        build_rag_command(settings, executable=python),
        cwd=str(paths.code_root), env=environment, stdout=stdout, stderr=stderr,
        redactions=redactions,
    )


def _validate_rag_ready(payload: Mapping[str, object]) -> None:
    required = {
        "status": "ready", "runtime_api": RUNTIME_API, "schema_version": 2,
    }
    for name, expected in required.items():
        if payload.get(name) != expected:
            raise RuntimeError("RAG readiness contract is incompatible")
    for name in ("corpus_version", "index_version", "embedder", "reranker"):
        if type(payload.get(name)) is not str or not payload[name]:
            raise RuntimeError("RAG readiness contract is incomplete")


def _local_expected_rag_identity(settings: RuntimeSettings) -> Mapping[str, object] | None:
    """Read the same approved manifests the child must report, without loading models."""
    if not settings.packaged:
        return None
    from pipeline.config import load_settings
    from rag.card_manifest import load_card_catalog
    from rag.readiness import approved_identity, load_clean_manifest

    configured = load_settings(str(RuntimePaths.from_settings(settings).config_path))
    rag = configured.rag
    clean = load_clean_manifest(Path(rag.store_path).parent / "rebuild_manifest.json")
    card_count = len(load_card_catalog(rag.card_catalog_path).cards)

    def manifest_counts(_path: Path, clean_collection: str, card_collection: str) -> Mapping[str, int]:
        return {clean_collection: int(clean["clean_count"]), card_collection: card_count}

    return approved_identity(configured, store_probe=manifest_counts)


def _stop_process(process: Any | None) -> None:
    if process is None:
        return
    if hasattr(process, "close"):
        process.close()
        return
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def _mount_gradio(host: FastAPI, blocks: Any) -> None:
    # The only Gradio import path runs after authenticated RAG startup succeeds.
    import gradio as gr

    gr.mount_gradio_app(host, blocks, path="/")


def _build_studio_ui() -> Any:
    from ui.speaker_studio import build_app

    return build_app()


class SessionGuard:
    """ASGI-level guard so mounted static apps and websocket queues cannot bypass auth."""

    def __init__(
        self,
        app: Any,
        *,
        auth: SessionAuth,
        public: set[tuple[str, str]],
        bearer_or_cookie: set[tuple[str, str]] | None = None,
        bearer_only: set[tuple[str, str]] | None = None,
    ) -> None:
        self.app = app
        self.auth = auth
        self.public = public
        self.bearer_or_cookie = bearer_or_cookie or set()
        self.bearer_only = bearer_only or set()

    async def __call__(self, scope: Mapping[str, Any], receive: Any, send: Any) -> None:
        kind = scope.get("type")
        if kind not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        host = client[0] if isinstance(client, tuple) and client else None
        if not _is_loopback(host):
            await self._deny(scope, send, "loopback access required")
            return
        method = str(scope.get("method", ""))
        path = str(scope.get("path", ""))
        raw_path = scope.get("raw_path", b"")
        try:
            exact_path = isinstance(raw_path, bytes) and raw_path == path.encode("ascii", "strict")
        except UnicodeEncodeError:
            exact_path = False
        route = (method, path)
        if kind != "websocket" and exact_path and route in self.public:
            await self.app(scope, receive, send)
            return
        connection = HTTPConnection(scope)
        cookie_valid = self.auth.valid_cookie(connection.cookies.get(self.auth.cookie_name))
        bearer_valid = self.auth.valid_bearer(connection.headers.get("authorization"))
        if kind != "websocket" and exact_path and route in self.bearer_only:
            permitted = bearer_valid
        elif kind != "websocket" and exact_path and route in self.bearer_or_cookie:
            permitted = bearer_valid or cookie_valid
        else:
            permitted = cookie_valid
        if not permitted:
            await self._deny(scope, send, "session required")
            return
        await self.app(scope, receive, send)

    @staticmethod
    async def _deny(scope: Mapping[str, Any], send: Any, detail: str) -> None:
        if scope.get("type") == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = ("{\"detail\":\"" + detail + "\"}").encode("utf-8")
        await send({
            "type": "http.response.start", "status": 403,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode("ascii"))],
        })
        await send({"type": "http.response.body", "body": body})


def create_app(
    settings: RuntimeSettings,
    *,
    auth: SessionAuth | None = None,
    process_factory: Callable[[RuntimeSettings], Any] = _spawn_rag,
    rag_client: RagHealthClient | None = None,
    ollama_client: OllamaHealthClient | None = None,
    manifest_probe: Callable[[Mapping[str, object]], None] = _validate_rag_ready,
    expected_rag_identity: Callable[[RuntimeSettings], Mapping[str, object] | None] = _local_expected_rag_identity,
    ui_factory: Callable[[], Any] = _build_studio_ui,
    mount_ui: Callable[[FastAPI, Any], None] = _mount_gradio,
    chat_handler: Callable[
        [str, str, str, str, int],
        tuple[str, str, str, Mapping[str, object], GenerationStatus],
    ] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    startup_timeout_s: float = RAG_STARTUP_TIMEOUT_SECONDS,
    maintenance_lease_s: float = 30.0,
) -> FastAPI:
    """Build an import-light loopback service; UI/model work starts only in lifespan."""
    if not settings.session_secret:
        raise ValueError("Studio service requires an YOUTUBER session secret")
    if not 0 < maintenance_lease_s <= 300:
        raise ValueError("Studio maintenance lease duration is invalid")
    session = auth or SessionAuth(settings.session_secret, clock=clock)
    rag = rag_client or _HttpRagHealthClient(settings)
    ollama = ollama_client or _HttpOllamaHealthClient(settings)
    run_chat = chat_handler or _production_chat
    state: dict[str, Any] = {
        "process": None,
        "rag": None,
        "ollama": None,
        "mounted": False,
        "expected_rag_identity_loaded": False,
        "expected_rag_identity": None,
    }

    def verify_rag(payload: Mapping[str, object]) -> None:
        manifest_probe(payload)
        if not state["expected_rag_identity_loaded"]:
            state["expected_rag_identity"] = expected_rag_identity(settings)
            state["expected_rag_identity_loaded"] = True
        expected = state["expected_rag_identity"]
        if expected is not None:
            for key, value in expected.items():
                if payload.get(key) != value:
                    raise RuntimeError("RAG readiness identities differ from approved manifests")

    def await_rag() -> Mapping[str, object]:
        deadline = clock() + startup_timeout_s
        last_error: Exception | None = None
        while True:
            try:
                if state["process"].poll() is not None:
                    raise RuntimeError("RAG subprocess exited before readiness")
                payload = rag.ready()
                if payload.get("status") == "unready":
                    raise RuntimeError("RAG subprocess is still warming")
            except Exception as error:
                last_error = error
                if clock() >= deadline:
                    raise last_error
                sleep(0.1)
                continue
            verify_rag(payload)
            return payload

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            state["process"] = process_factory(settings)
            state["rag"] = await_rag()
            state["ollama"] = dict(ollama.ready())
            ui = ui_factory()
            mount_ui(_app, ui)
            state["mounted"] = True
            yield
        except Exception:
            _stop_process(state["process"])
            state["process"] = None
            raise
        finally:
            _stop_process(state["process"])

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    chat_gate = threading.Lock()
    maintenance_lock = threading.Lock()
    maintenance_token: str | None = None
    maintenance_timer: threading.Timer | None = None
    maintenance_generation = 0

    def expire_maintenance(token: str, generation: int) -> None:
        nonlocal maintenance_token, maintenance_timer
        with maintenance_lock:
            if maintenance_token is None or generation != maintenance_generation or not secrets.compare_digest(maintenance_token, token):
                return
            maintenance_token = None
            maintenance_timer = None
            chat_gate.release()

    def arm_maintenance_timer(token: str) -> None:
        nonlocal maintenance_timer, maintenance_generation
        maintenance_generation += 1
        timer = threading.Timer(maintenance_lease_s, expire_maintenance, args=(token, maintenance_generation))
        timer.daemon = True
        maintenance_timer = timer
        timer.start()
    public = {
        ("GET", "/health/live"), ("GET", "/health/ready"),
        ("POST", "/bootstrap"), ("POST", "/v1/browser-bootstrap"),
        ("GET", "/v1/browser-exchange"),
    }
    app.add_middleware(
        SessionGuard,
        auth=session,
        public=public,
        bearer_or_cookie={("GET", "/v1/status")},
        bearer_only={
            ("POST", "/v1/maintenance/acquire"),
            ("POST", "/v1/maintenance/renew"),
            ("POST", "/v1/maintenance/release"),
        },
    )

    def require_bearer(request: Request) -> bool:
        return session.valid_bearer(request.headers.get("authorization"))

    @app.get("/health/live")
    def health_live() -> dict[str, object]:
        return {"status": "live", "runtime_api": RUNTIME_API}

    @app.get("/health/ready")
    def health_ready() -> dict[str, object]:
        try:
            rag_ready = rag.ready()
            verify_rag(rag_ready)
            ollama_ready = dict(ollama.ready())
        except Exception:
            return JSONResponse({"status": "unready", "runtime_api": RUNTIME_API}, status_code=503)
        return {
            "status": "ready", "runtime_api": RUNTIME_API,
            "worker_version": WORKER_VERSION,
            "answer_mode_contract_version": ANSWER_MODE_CONTRACT_VERSION,
            "answer_token_cap": {"minimum": MIN_ANSWER_TOKEN_CAP, "default": DEFAULT_ANSWER_TOKEN_CAP, "maximum": MAX_ANSWER_TOKEN_CAP},
            "rag_corpus_version": rag_ready["corpus_version"],
            "rag_index_version": rag_ready["index_version"],
            "answer_model": ollama_ready["answer_model"],
            "verifier_model": ollama_ready["verifier_model"],
            "embedder": rag_ready["embedder"], "reranker": rag_ready["reranker"],
        }

    @app.get("/v1/status")
    def status() -> dict[str, str]:
        return {"status": "busy" if chat_gate.locked() else "idle"}

    @app.post("/v1/maintenance/acquire")
    def acquire_maintenance() -> dict[str, object]:
        nonlocal maintenance_token
        if not chat_gate.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Studio is busy")
        try:
            with maintenance_lock:
                if maintenance_token is not None:
                    raise RuntimeError("Studio maintenance state is inconsistent")
                maintenance_token = secrets.token_urlsafe(32)
                arm_maintenance_timer(maintenance_token)
                return {"lease_token": maintenance_token, "expires_in_seconds": maintenance_lease_s}
        except BaseException:
            chat_gate.release()
            raise

    @app.post("/v1/maintenance/release")
    def release_maintenance(payload: MaintenanceReleaseRequest) -> dict[str, str]:
        nonlocal maintenance_token, maintenance_timer, maintenance_generation
        with maintenance_lock:
            if maintenance_token is None or not secrets.compare_digest(maintenance_token, payload.lease_token):
                raise HTTPException(status_code=409, detail="Maintenance lease does not match")
            if maintenance_timer is not None:
                maintenance_timer.cancel()
                maintenance_timer = None
            maintenance_generation += 1
            maintenance_token = None
            chat_gate.release()
        return {"status": "released"}

    @app.post("/v1/maintenance/renew")
    def renew_maintenance(payload: MaintenanceReleaseRequest) -> dict[str, object]:
        nonlocal maintenance_timer
        with maintenance_lock:
            if maintenance_token is None or not secrets.compare_digest(maintenance_token, payload.lease_token):
                raise HTTPException(status_code=409, detail="Maintenance lease does not match")
            if maintenance_timer is not None:
                maintenance_timer.cancel()
            arm_maintenance_timer(maintenance_token)
        return {"status": "renewed", "expires_in_seconds": maintenance_lease_s}

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
        ollama_ready = state.get("ollama")
        if type(ollama_ready) is not dict or type(ollama_ready.get("answer_model")) is not str:
            raise HTTPException(status_code=503, detail="Studio is not ready")
        if not chat_gate.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Studio is busy")
        try:
            answer, context, metadata, raw_trace, raw_generation_status = run_chat(
                payload.question,
                ollama_ready["answer_model"],
                payload.answer_mode,
                payload.retrieval_mode,
                payload.answer_token_cap,
            )
            trace = GroundingTrace.model_validate(raw_trace)
            generation = GenerationTrace.model_validate(
                {"status": raw_generation_status}
            )
            answer = _require_generated_answer(answer, generation.status)
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Studio generation failed") from exc
        finally:
            chat_gate.release()
        return ChatResponse(
            runtime_api=RUNTIME_API,
            answer_mode_contract_version=ANSWER_MODE_CONTRACT_VERSION,
            answer_model=ollama_ready["answer_model"],
            answer_mode=payload.answer_mode,
            retrieval_mode=payload.retrieval_mode,
            answer_token_cap=payload.answer_token_cap,
            answer=answer,
            context_markdown=context,
            metadata=metadata,
            grounding_status=trace.status,
            evidence_count=trace.evidence_count,
            generation_status="generated",
        )

    @app.post("/bootstrap")
    def bootstrap(request: Request):
        if not require_bearer(request):
            return JSONResponse({"detail": "authentication required"}, status_code=403)
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(
            key=session.cookie_name, value=session.cookie_token,
            httponly=True, samesite="strict", path="/",
        )
        return response

    @app.post("/v1/browser-bootstrap")
    def browser_bootstrap(request: Request) -> JSONResponse:
        if not require_bearer(request):
            return JSONResponse({"detail": "authentication required"}, status_code=403)
        return JSONResponse({"url": session.create_browser_exchange_url(settings.studio_port)})

    @app.get("/v1/browser-exchange")
    def browser_exchange(request: Request):
        if not session.consume_browser_nonce(request.query_params.get("nonce")):
            return JSONResponse({"detail": "browser exchange is invalid"}, status_code=403)
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(
            key=session.cookie_name, value=session.cookie_token,
            httponly=True, samesite="strict", path="/",
        )
        return response

    return app
