from __future__ import annotations

from types import SimpleNamespace

from rag.retrieve_cli import retrieve_packaged


def test_packaged_retrieve_cli_uses_only_configured_loopback_rag_with_bearer():
    seen = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"schema_version": 2, "status": "unsupported"}

    def get(url, *, params, headers, timeout, allow_redirects):
        seen.update(url=url, params=params, headers=headers, timeout=timeout, redirects=allow_redirects)
        return Response()

    result = retrieve_packaged(
        "soru", "clean", SimpleNamespace(rag_port=49153, session_secret="s" * 64), get=get,
    )

    assert result == {"schema_version": 2, "status": "unsupported"}
    assert seen == {
        "url": "http://127.0.0.1:49153/retrieve",
        "params": {"q": "soru", "mode": "clean"},
        "headers": {"Authorization": "Bearer " + "s" * 64},
        "timeout": 660,
        "redirects": False,
    }
