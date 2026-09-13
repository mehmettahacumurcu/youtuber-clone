"""Small redirect-denying JSON client for launcher-assigned loopback services."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from urllib.parse import urlsplit


_MAX_JSON_BYTES = 1_000_000


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


def _open_no_redirect(request: urllib.request.Request, *, timeout: float):
    # Explicitly ignore HTTP(S)_PROXY, NO_PROXY, and lowercase variants.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect()).open(request, timeout=timeout)


def _validate_url(url: str) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or
        parsed.port is None or not 1024 <= parsed.port <= 65535 or
        parsed.username is not None or parsed.password is not None or parsed.fragment
    ):
        raise ValueError("URL must be an exact http://127.0.0.1:<launcher-port> origin")


def loopback_json(
    url: str,
    *,
    bearer: str | None = None,
    timeout_s: float = 3.0,
    max_bytes: int = _MAX_JSON_BYTES,
    opener: Callable[..., object] = _open_no_redirect,
) -> dict[str, object]:
    """GET a bounded JSON body without following redirects or trusting caller origins."""
    _validate_url(url)
    headers = {"Accept": "application/json"}
    if bearer:
        headers["Authorization"] = "Bearer " + bearer
    request = urllib.request.Request(url, headers=headers)
    try:
        response = opener(request, timeout=timeout_s)
    except urllib.error.HTTPError as error:
        if 300 <= error.code < 400:
            raise RuntimeError("loopback service attempted a redirect") from error
        response = error
    with response:
        final_url = getattr(response, "geturl", lambda: url)()
        if final_url != url:
            raise RuntimeError("loopback service changed the final URL")
        length = getattr(response, "headers", {}).get("Content-Length")
        if length is not None and (not str(length).isdigit() or int(length) > max_bytes):
            raise RuntimeError("loopback JSON body is too large")
        try:
            body = response.read(max_bytes + 1)
        except TypeError:
            body = response.read()
    if len(body) > max_bytes:
        raise RuntimeError("loopback JSON body is too large")
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeError, ValueError) as error:
        raise RuntimeError("loopback response is not valid JSON") from error
    if type(value) is not dict:
        raise RuntimeError("loopback response must be a JSON object")
    return value
