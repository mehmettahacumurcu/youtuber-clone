from __future__ import annotations

import threading

from runtime.session_auth import SessionAuth


def test_cookie_session_is_distinct_from_bearer_and_exactly_checks_bearer():
    auth = SessionAuth("s" * 64)

    assert auth.cookie_token != "s" * 64
    assert auth.valid_bearer("Bearer " + "s" * 64) is True
    assert auth.valid_bearer("Bearer wrong") is False
    assert auth.valid_bearer(None) is False
    assert auth.valid_cookie(auth.cookie_token) is True
    assert auth.valid_cookie("wrong") is False


def test_cookie_generation_retries_if_a_token_would_equal_the_bearer_secret():
    tokens = iter(["s" * 64, "independent-cookie-token"])
    auth = SessionAuth("s" * 64, token_factory=lambda _bytes: next(tokens))

    assert auth.cookie_token == "independent-cookie-token"


def test_browser_nonce_is_single_use_and_bound_to_its_launch_session():
    first = SessionAuth("s" * 64)
    second = SessionAuth("s" * 64)
    url = first.create_browser_exchange_url(49152)
    nonce = url.rsplit("nonce=", 1)[1]

    assert url.startswith("http://127.0.0.1:49152/v1/browser-exchange?nonce=")
    assert first.consume_browser_nonce(nonce) is True
    assert first.consume_browser_nonce(nonce) is False
    assert second.consume_browser_nonce(nonce) is False


def test_browser_nonce_expires_after_thirty_seconds_using_injected_monotonic_clock():
    now = [100.0]
    auth = SessionAuth("s" * 64, clock=lambda: now[0])
    nonce = auth.create_browser_exchange_url(49152).rsplit("nonce=", 1)[1]

    now[0] = 130.0

    assert auth.consume_browser_nonce(nonce) is False


def test_browser_nonce_exchange_consumes_once_under_racing_requests():
    auth = SessionAuth("s" * 64)
    nonce = auth.create_browser_exchange_url(49152).rsplit("nonce=", 1)[1]
    results: list[bool] = []

    threads = [threading.Thread(target=lambda: results.append(auth.consume_browser_nonce(nonce))) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count(True) == 1
    assert results.count(False) == 7
