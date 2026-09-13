"""Generate from the local Ollama model via the raw completion endpoint.

raw=true bypasses Ollama's chat templating — required because this is a base/transcript
completion model, not an instruct chat model.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

from runtime.ollama_http import ollama_json
from runtime.settings import RuntimeSettings

def generate_ollama(prompt: str, model: str, temperature: float, top_p: float,
                    repeat_penalty: float, num_predict: int, timeout: int = 180,
                    post_json: Callable[..., Mapping[str, object]] = ollama_json) -> str:
    body = {
        "model": model,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "num_predict": num_predict,
        },
    }
    url = RuntimeSettings.from_environment().ollama_origin + "/api/generate"
    payload = post_json("POST", url, payload=body, timeout_s=float(timeout))
    response = payload.get("response", "")
    return response.strip() if isinstance(response, str) else ""
