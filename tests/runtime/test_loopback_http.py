from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from runtime import loopback_http
from runtime.loopback_http import loopback_json


def test_loopback_json_rejects_a_redirect_without_sending_bearer_to_the_target():
    seen = []

    def opener(request, *, timeout):
        seen.append((request.full_url, request.get_header("Authorization"), timeout))
        raise urllib.error.HTTPError(
            request.full_url, 302, "redirect", {"Location": "http://127.0.0.1:49999/steal"}, None,
        )

    with pytest.raises(RuntimeError, match="redirect"):
        loopback_json("http://127.0.0.1:49153/retrieve", bearer="s" * 64, opener=opener)

    assert seen == [("http://127.0.0.1:49153/retrieve", "Bearer " + "s" * 64, 3.0)]


def test_loopback_json_rejects_non_exact_loopback_origin_before_making_a_request():
    with pytest.raises(ValueError, match="127.0.0.1"):
        loopback_json("http://localhost:49153/retrieve", opener=lambda *_args, **_kwargs: None)


def test_default_opener_explicitly_disables_proxy_discovery(monkeypatch):
    captured = {}

    class Opener:
        def open(self, _request, *, timeout):
            captured["timeout"] = timeout
            return None

    def build_opener(*handlers):
        captured["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    loopback_http._open_no_redirect(
        urllib.request.Request("http://127.0.0.1:49153/health"), timeout=2.0,
    )

    proxy = next(handler for handler in captured["handlers"] if isinstance(handler, urllib.request.ProxyHandler))
    assert proxy.proxies == {}
    assert captured["timeout"] == 2.0
