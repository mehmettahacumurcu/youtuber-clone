"""Fail-closed helpers for enforcing Ollama's one-model residency policy."""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping

from runtime.ollama_http import _validate_ollama_url, ollama_json
from runtime.settings import RuntimeSettings


UNLOAD_TIMEOUT_S = 15.0
UNLOAD_POLL_INTERVAL_S = 0.25


class ModelResidencyError(RuntimeError):
    """Ollama could not prove that a requested model is absent from memory."""


def ensure_absent(model: str) -> None:
    """Unload one local model and fail closed until Ollama confirms its absence."""
    ensure_model_absent(
        model,
        base_url=RuntimeSettings.from_environment().ollama_origin,
        timeout_s=UNLOAD_TIMEOUT_S,
        poll_interval_s=UNLOAD_POLL_INTERVAL_S,
    )


def ensure_model_absent(
    model: str,
    *,
    base_url: str,
    timeout_s: float,
    poll_interval_s: float,
    request_json: Callable[..., Mapping[str, object]] = ollama_json,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Request an unload and return only after ``/api/ps`` confirms absence.

    Callers inject transport and timing dependencies so the barrier is fully
    deterministic in tests and does not hide a second model behind stale state.
    """
    if not _is_positive_finite(timeout_s):
        raise ValueError("timeout_s must be finite and positive")
    if not _is_positive_finite(poll_interval_s):
        raise ValueError("poll_interval_s must be finite and positive")
    _validate_ollama_origin(base_url)

    url = base_url.rstrip("/")
    deadline = clock() + timeout_s
    _post_unload(request_json, f"{url}/api/generate", model, _remaining_timeout(deadline, clock, model))
    _remaining_timeout(deadline, clock, model)

    while True:
        models = _get_loaded_models(
            request_json,
            f"{url}/api/ps",
            _remaining_timeout(deadline, clock, model),
        )
        remaining = _remaining_timeout(deadline, clock, model)
        if not _contains_model(models, model):
            return
        sleep(min(poll_interval_s, remaining))


def _is_positive_finite(value: float) -> bool:
    return math.isfinite(value) and value > 0


def _remaining_timeout(deadline: float, clock: Callable[[], float], model: str) -> float:
    remaining = deadline - clock()
    if remaining <= 0:
        raise ModelResidencyError(f"timed out waiting for Ollama model {model!r} to unload")
    return remaining


def _post_unload(request_json: Callable[..., Mapping[str, object]], url: str, model: str, timeout_s: float) -> None:
    try:
        request_json("POST", url, payload={"model": model, "keep_alive": 0}, timeout_s=timeout_s)
    except Exception as exc:
        raise ModelResidencyError(f"Ollama unload request failed: {exc}") from exc


def _get_loaded_models(request_json: Callable[..., Mapping[str, object]], url: str, timeout_s: float) -> list[Mapping[str, object]]:
    try:
        payload = request_json("GET", url, timeout_s=timeout_s)
    except Exception as exc:
        raise ModelResidencyError(f"Ollama residency status request failed: {exc}") from exc

    try:
        return _parse_loaded_models(payload)
    except Exception as exc:
        raise ModelResidencyError(f"invalid Ollama residency status: {exc}") from exc


def _parse_loaded_models(payload: object) -> list[Mapping[str, object]]:
    if type(payload) is not dict:
        raise TypeError("top-level status must be an object")
    models = payload.get("models")
    if type(models) is not list:
        raise TypeError("status models must be an array")

    parsed: list[Mapping[str, object]] = []
    for entry in models:
        if type(entry) is not dict:
            raise TypeError("each loaded model must be an object")
        name = entry.get("name")
        if type(name) is not str or not name.strip():
            raise TypeError("each loaded model must have a non-empty string name")
        if "model" in entry and (type(entry["model"]) is not str or not entry["model"].strip()):
            raise TypeError("loaded model identifier must be a non-empty string when present")
        parsed.append(entry)
    return parsed


def _contains_model(models: list[Mapping[str, object]], target: str) -> bool:
    return any(entry["name"] == target or entry.get("model") == target for entry in models)


def _validate_ollama_origin(url: str) -> None:
    _validate_ollama_url(url.rstrip("/"), expected_origin=url.rstrip("/"))
