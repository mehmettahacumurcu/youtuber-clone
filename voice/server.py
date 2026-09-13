"""Authenticated, loopback-only FastAPI boundary around the voice pipeline."""
from __future__ import annotations

import hashlib
import hmac
import importlib
import ipaddress
import os
import secrets
import threading
from collections.abc import Callable
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Response

from runtime.paths import RuntimePaths
from runtime.settings import RuntimeSettings
from voice.api_models import MaintenanceReleaseRequest, ReadinessResponse, StatusResponse, SynthesisRequest
from voice.models import VoiceArtifactManifest, XttsSettings
from voice.pipeline import PipelineBusyError, PipelineError, VoicePipeline
from voice.rvc_engine import RvcEngine, probe_applio_runtime
from voice.xtts_engine import XttsEngine


ReadinessProbe = Callable[[], None]
RuntimeImportProbe = Callable[[RuntimePaths], None]


def _is_loopback(host: str | None) -> bool:
    try:
        return host is not None and ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _loopback_only(request: Request) -> None:
    if request.client is None or not _is_loopback(request.client.host):
        raise HTTPException(status_code=403, detail="loopback access required")


def _build_pipeline(settings: RuntimeSettings) -> VoicePipeline:
    paths = RuntimePaths.from_settings(settings)
    manifest_path = paths.voice_root / "manifest.json"
    manifest = VoiceArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    return VoicePipeline(
        XttsEngine(XttsSettings.from_runtime_paths(paths, manifest)),
        RvcEngine.from_runtime_paths(paths, manifest),
        paths.cache_root,
        low_vram_voice=settings.low_vram_voice,
        keep_intermediate_audio=os.environ.get("YOUTUBER_KEEP_INTERMEDIATE_AUDIO") == "1",
    )


def _validate_manifest(paths: RuntimePaths) -> VoiceArtifactManifest:
    manifest_path = paths.voice_root / "manifest.json"
    manifest = VoiceArtifactManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest.files:
        manifest.validate_file(paths.voice_root, entry.path)
    XttsSettings.from_runtime_paths(paths, manifest)
    # Selecting RVC settings validates its fixed epoch/index identity without loading Applio.
    from voice.rvc_engine import RvcSettings

    RvcSettings.from_runtime_paths(paths, manifest)
    return manifest


def _probe_runtime_imports(paths: RuntimePaths) -> None:
    """Import runtime APIs only; model construction and torch.load are forbidden here."""
    torch = importlib.import_module("torch")
    if not bool(torch.cuda.is_available()):
        raise RuntimeError("CUDA runtime is unavailable")
    importlib.import_module("TTS.tts.configs.xtts_config")
    probe_applio_runtime(paths.code_root / "vendor" / "applio")


def run_readiness_check(
    settings: RuntimeSettings, *, runtime_probe: RuntimeImportProbe | None = None
) -> None:
    """Check manifest identities and runtime imports without loading any model weights."""
    paths = RuntimePaths.from_settings(settings)
    _validate_manifest(paths)
    (runtime_probe or _probe_runtime_imports)(paths)


def run_package_healthcheck(
    settings: RuntimeSettings, *, runtime_probe: RuntimeImportProbe | None = None
) -> None:
    """Verify frozen runtime imports without reading or loading model artifacts."""
    paths = RuntimePaths.from_settings(settings)
    (runtime_probe or _probe_runtime_imports)(paths)


def create_app(
    settings: RuntimeSettings,
    *,
    pipeline: VoicePipeline | object | None = None,
    readiness_probe: ReadinessProbe | None = None,
    maintenance_lease_s: float = 30.0,
) -> FastAPI:
    """Create an import-light service; models are instantiated only on synthesis."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    active_pipeline: VoicePipeline | object | None = pipeline
    pipeline_gate = threading.Lock()
    synth_gate = threading.Lock()
    maintenance_lock = threading.Lock()
    maintenance_token: str | None = None
    maintenance_timer: threading.Timer | None = None
    maintenance_generation = 0
    if not 0 < maintenance_lease_s <= 300:
        raise ValueError("voice maintenance lease duration is invalid")
    probe = readiness_probe or (lambda: run_readiness_check(settings))

    def expire_maintenance(token: str, generation: int) -> None:
        nonlocal maintenance_token, maintenance_timer
        with maintenance_lock:
            if maintenance_token is None or generation != maintenance_generation or not secrets.compare_digest(maintenance_token, token):
                return
            maintenance_token = None
            maintenance_timer = None
            synth_gate.release()

    def arm_maintenance_timer(token: str) -> None:
        nonlocal maintenance_timer, maintenance_generation
        maintenance_generation += 1
        timer = threading.Timer(maintenance_lease_s, expire_maintenance, args=(token, maintenance_generation))
        timer.daemon = True
        maintenance_timer = timer
        timer.start()

    def get_pipeline() -> VoicePipeline | object:
        nonlocal active_pipeline
        with pipeline_gate:
            if active_pipeline is None:
                active_pipeline = _build_pipeline(settings)
            return active_pipeline

    def require_auth(request: Request) -> None:
        secret = settings.session_secret
        header = request.headers.get("authorization")
        expected = f"Bearer {secret}" if secret else ""
        if not secret or not header or not hmac.compare_digest(header, expected):
            raise HTTPException(
                status_code=401,
                detail="authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.get("/health/live", dependencies=[Depends(_loopback_only)])
    def health_live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready", dependencies=[Depends(_loopback_only)])
    def health_ready() -> ReadinessResponse:
        try:
            probe()
        except Exception as error:
            raise HTTPException(status_code=503, detail="voice worker is not ready") from error
        return ReadinessResponse(status="ready")

    @app.get("/v1/status", dependencies=[Depends(_loopback_only), Depends(require_auth)])
    def status() -> StatusResponse:
        with maintenance_lock:
            if maintenance_token is not None:
                return StatusResponse(status="maintenance")
        worker = get_pipeline()
        return StatusResponse(status=getattr(worker, "status"))

    @app.post("/v1/maintenance/acquire", dependencies=[Depends(_loopback_only), Depends(require_auth)])
    def acquire_maintenance() -> dict[str, object]:
        nonlocal maintenance_token
        if not synth_gate.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="voice worker is busy")
        try:
            with maintenance_lock:
                if maintenance_token is not None:
                    raise RuntimeError("voice maintenance state is inconsistent")
                maintenance_token = secrets.token_urlsafe(32)
                arm_maintenance_timer(maintenance_token)
                return {"lease_token": maintenance_token, "expires_in_seconds": maintenance_lease_s}
        except BaseException:
            synth_gate.release()
            raise

    @app.post("/v1/maintenance/release", dependencies=[Depends(_loopback_only), Depends(require_auth)])
    def release_maintenance(payload: MaintenanceReleaseRequest) -> dict[str, str]:
        nonlocal maintenance_token, maintenance_timer, maintenance_generation
        with maintenance_lock:
            if maintenance_token is None or not secrets.compare_digest(maintenance_token, payload.lease_token):
                raise HTTPException(status_code=409, detail="maintenance lease does not match")
            if maintenance_timer is not None:
                maintenance_timer.cancel()
                maintenance_timer = None
            maintenance_generation += 1
            maintenance_token = None
            synth_gate.release()
        return {"status": "released"}

    @app.post("/v1/maintenance/renew", dependencies=[Depends(_loopback_only), Depends(require_auth)])
    def renew_maintenance(payload: MaintenanceReleaseRequest) -> dict[str, object]:
        nonlocal maintenance_timer
        with maintenance_lock:
            if maintenance_token is None or not secrets.compare_digest(maintenance_token, payload.lease_token):
                raise HTTPException(status_code=409, detail="maintenance lease does not match")
            if maintenance_timer is not None:
                maintenance_timer.cancel()
            arm_maintenance_timer(maintenance_token)
        return {"status": "renewed", "expires_in_seconds": maintenance_lease_s}

    @app.post("/v1/unload", dependencies=[Depends(_loopback_only), Depends(require_auth)])
    def unload() -> StatusResponse:
        if not synth_gate.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="voice worker is busy")
        try:
            worker = get_pipeline()
            try:
                worker.unload()
            except PipelineError as error:
                raise HTTPException(status_code=500, detail="voice worker unload failed") from error
            return StatusResponse(status=getattr(worker, "status"))
        finally:
            synth_gate.release()

    @app.post("/v1/synthesize", dependencies=[Depends(_loopback_only), Depends(require_auth)])
    def synthesize(payload: SynthesisRequest) -> Response:
        if not synth_gate.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="voice worker is busy")
        try:
            worker = get_pipeline()
            try:
                result = worker.synthesize(payload.text, payload.request_id)
            except PipelineBusyError as error:
                raise HTTPException(status_code=409, detail="voice worker is busy") from error
            except PipelineError as error:
                raise HTTPException(status_code=500, detail="voice synthesis failed") from error
            except Exception as error:
                raise HTTPException(status_code=500, detail="voice synthesis failed") from error
            try:
                # Hold immutable bytes before emitting the response; never race a later cleanup.
                body = Path(result.final_path).read_bytes()
                digest = hashlib.sha256(body).hexdigest()
            except OSError as error:
                raise HTTPException(status_code=500, detail="voice synthesis failed") from error
            if not hmac.compare_digest(digest, result.sha256):
                raise HTTPException(status_code=500, detail="voice synthesis failed")
            return Response(
                content=body,
                media_type="audio/wav",
                headers={
                    "X-YouTuber-SHA256": digest,
                    "X-YouTuber-Sample-Rate": str(result.sample_rate),
                    "X-YouTuber-Duration-Seconds": f"{result.duration_seconds:.6f}",
                    "X-YouTuber-RVC-Epoch": str(result.rvc_epoch),
                    "X-YouTuber-Index-Rate": str(result.index_rate),
                    "X-YouTuber-Request-ID": payload.request_id,
                },
            )
        finally:
            synth_gate.release()

    return app
