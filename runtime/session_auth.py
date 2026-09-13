"""Ephemeral, per-launch authentication for loopback desktop services."""
from __future__ import annotations

import hmac
import secrets
import threading
import time
from collections.abc import Callable
from urllib.parse import urlencode


COOKIE_NAME = "youtuber_session"
_NONCE_TTL_SECONDS = 30.0
_MAX_NONCES = 128


class SessionAuth:
    """Keep bearer authentication separate from the browser session cookie."""

    def __init__(
        self,
        secret: str,
        *,
        clock: Callable[[], float] = time.monotonic,
        token_factory: Callable[[int], str] = secrets.token_urlsafe,
    ) -> None:
        if not secret:
            raise ValueError("session secret is required")
        self._secret = secret
        self._clock = clock
        self._token_factory = token_factory
        # This cookie token is intentionally unrelated to the launcher bearer secret.
        self._cookie_token = token_factory(32)
        if hmac.compare_digest(self._cookie_token, secret):
            self._cookie_token = token_factory(32)
        if hmac.compare_digest(self._cookie_token, secret):
            raise RuntimeError("session cookie token must differ from the bearer secret")
        self._nonces: dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def cookie_name(self) -> str:
        return COOKIE_NAME

    @property
    def cookie_token(self) -> str:
        return self._cookie_token

    def valid_bearer(self, header: str | None) -> bool:
        expected = "Bearer " + self._secret
        return isinstance(header, str) and hmac.compare_digest(header, expected)

    def valid_cookie(self, token: str | None) -> bool:
        return isinstance(token, str) and hmac.compare_digest(token, self._cookie_token)

    def create_browser_exchange_url(self, studio_port: int) -> str:
        """Create a one-use, 128-bit nonce URL for the explicit browser escape hatch."""
        nonce = self._token_factory(16)
        now = self._clock()
        with self._lock:
            self._discard_expired(now)
            if len(self._nonces) >= _MAX_NONCES:
                oldest = min(self._nonces, key=self._nonces.__getitem__)
                del self._nonces[oldest]
            self._nonces[nonce] = now + _NONCE_TTL_SECONDS
        return "http://127.0.0.1:%d/v1/browser-exchange?%s" % (
            studio_port,
            urlencode({"nonce": nonce}),
        )

    def consume_browser_nonce(self, supplied: str | None) -> bool:
        """Atomically consume a valid browser nonce, including under concurrent exchanges."""
        if not isinstance(supplied, str):
            return False
        now = self._clock()
        with self._lock:
            self._discard_expired(now)
            match = next(
                (nonce for nonce in self._nonces if hmac.compare_digest(nonce, supplied)),
                None,
            )
            if match is None:
                return False
            # Consume before returning so every race after this point fails closed.
            expires_at = self._nonces.pop(match)
            return now < expires_at

    def _discard_expired(self, now: float) -> None:
        for nonce, expires_at in tuple(self._nonces.items()):
            if now >= expires_at:
                del self._nonces[nonce]
