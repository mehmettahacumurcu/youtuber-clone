"""Dependency-light schema-v2 retrieval client and Studio answer policy."""
from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

from pipeline.config import load_settings
from rag.ollama_residency import ensure_model_absent
from rag.prompt import REFUSAL_TR, SYS, build_grounded_user
from runtime.settings import RuntimeSettings
from runtime.loopback_http import loopback_json
from ui.answer_modes import AnswerMode, validate_answer_mode
from runtime.ollama_http import ollama_json


_RUNTIME_SETTINGS = RuntimeSettings.from_environment()
RETR_URL = f"http://127.0.0.1:{_RUNTIME_SETTINGS.rag_port}/retrieve"
OLLAMA_BASE_URL = _RUNTIME_SETTINGS.ollama_origin
OLLAMA_CHAT = OLLAMA_BASE_URL + "/api/chat"
_UNLOAD_TIMEOUT_S = 15.0
_UNLOAD_POLL_INTERVAL_S = 0.25
MIN_ANSWER_TOKEN_CAP = 64
MAX_ANSWER_TOKEN_CAP = 4096
DEFAULT_ANSWER_TOKEN_CAP = 640
_ERROR_TR = "RAG kanıt denetimi şu anda çalışmadığı için cevap üretmedim."
_EMPTY_OUTPUT_TR = "Boş model çıktısı alındı; yanıt üretilemedi."
_PARTIAL_PREFIX = "Kanıtı bulunamayan kısım: {claims}\n\n"
_UNVERIFIED_UNSUPPORTED = (
    "Unverified model answer — not supported by verified transcripts."
)
_UNVERIFIED_ERROR = (
    "Unverified model answer — transcript verification was unavailable."
)
_GROUNDED_STATUSES = {"answerable", "partial"}
_FAILURE_REASONS = {
    "retrieval_error",
    "source_error",
    "verifier_transport_error",
    "verifier_invalid_output",
    "verifier_error",
    "evidence_budget_error",
}
_VERDICTS = {"entailed", "contradicted", "explicitly_unresolved", "not_found"}
_SPAN_ID = re.compile(r"span::([^:]+)::(\d+)::(\d+)")
GenerationStatus: TypeAlias = Literal[
    "generated",
    "empty_output",
    "model_error",
    "policy_refusal",
    "retrieval_error",
    "no_input",
]


@dataclass(frozen=True)
class ChatTurnResult:
    history: list[dict[str, str]]
    context: str
    metadata: str
    trace: dict[str, object]
    generation_status: GenerationStatus


def _ensure_absent(model: str) -> None:
    """Fail closed until Ollama confirms a model has released GPU residency."""
    ensure_model_absent(
        model,
        base_url=OLLAMA_BASE_URL,
        timeout_s=_UNLOAD_TIMEOUT_S,
        poll_interval_s=_UNLOAD_POLL_INTERVAL_S,
    )


def ensure_answer_model_absent(model: str) -> None:
    """Voice handoff uses the same fail-closed Ollama residency barrier as RAG."""
    _ensure_absent(model)


def retrieve_decision(
    query: str,
    mode: str,
    *,
    url: str = RETR_URL,
    timeout_s: int = 660,
    opener: Callable[..., object] | None = None,
) -> dict[str, object]:
    """Fetch and validate the retrieval service's only supported response schema."""
    request_url = url + "?" + urllib.parse.urlencode({"q": query, "mode": mode})
    kwargs: dict[str, object] = {
        "bearer": _RUNTIME_SETTINGS.session_secret if _RUNTIME_SETTINGS.packaged else None,
        "timeout_s": float(timeout_s),
    }
    if opener is not None:
        kwargs["opener"] = opener
    payload = loopback_json(request_url, **kwargs)
    return validate_v2_response(payload)


def validate_v2_response(payload: object) -> dict[str, object]:
    """Reject every payload that could make a RAG-on generation unsafe."""
    if type(payload) is not dict or payload.get("schema_version") != 2:
        raise ValueError("retrieval response is not schema version 2")
    status = payload.get("status")
    if type(status) is not str or status not in {"answerable", "partial", "unsupported", "error"}:
        raise ValueError("retrieval response has an unknown status")
    if type(payload.get("grounded")) is not bool or payload["grounded"] != (status in _GROUNDED_STATUSES):
        raise ValueError("retrieval grounded flag disagrees with status")
    failure_reason = payload.get("failure_reason")
    if status == "error":
        if type(failure_reason) is not str or failure_reason not in _FAILURE_REASONS:
            raise ValueError("error response has an invalid failure_reason")
    elif failure_reason is not None:
        raise ValueError("non-error response has a failure_reason")
    for field in ("claims", "evidence_hits", "hint_hits", "unsupported_claims"):
        if type(payload.get(field)) is not list:
            raise ValueError(f"retrieval field {field} must be an array")
    evidence = payload["evidence_hits"]
    if status in _GROUNDED_STATUSES:
        if type(payload.get("answer_question")) is not str or not payload["answer_question"].strip():
            raise ValueError("grounded response has no answer_question")
        if not evidence:
            raise ValueError("grounded response has no evidence")
    elif evidence or payload.get("answer_question") is not None:
        raise ValueError("fail-closed response contains generation input")

    unsupported = payload["unsupported_claims"]
    if any(type(claim) is not str or not claim.strip() for claim in unsupported):
        raise ValueError("unsupported claims must be non-empty strings")
    if status == "partial" and not unsupported:
        raise ValueError("partial response has no unsupported claims")
    if status in {"answerable", "error"} and unsupported:
        raise ValueError("only partial and unsupported responses may contain unsupported claims")

    hint_takes = set()
    for hint in payload["hint_hits"]:
        _validate_hint(hint)
        hint_takes.add(hint["take"])

    total = 0
    evidence_ids = set()
    for item in evidence:
        _validate_evidence(item)
        if item["id"] in evidence_ids:
            raise ValueError("evidence IDs must be unique")
        evidence_ids.add(item["id"])
        if item["text"] in hint_takes:
            raise ValueError("card take cannot appear in evidence")
        total += len(item["text"])
    if len(evidence) > 5 or total > 9000:
        raise ValueError("answer evidence exceeds the locked budget")
    claims = payload["claims"]
    _validate_claims(claims, evidence_ids, allow_diagnostic_citations=status == "unsupported")
    if status in _GROUNDED_STATUSES:
        central_claims = [claim for claim in claims if claim["central"]]
        if not central_claims:
            raise ValueError("grounded response has no central claims")
        if status == "answerable" and any(claim["verdict"] == "not_found" for claim in central_claims):
            raise ValueError("answerable response has an unsupported central claim")
        if status == "partial" and not any(
            claim["verdict"] != "not_found" for claim in central_claims
        ):
            raise ValueError("partial response has no supported central claims")
        if status == "partial":
            expected_unsupported = [
                claim["claim"] for claim in central_claims
                if claim["verdict"] == "not_found"
            ]
            if not expected_unsupported:
                raise ValueError("partial response has no unsupported central claims")
            if unsupported != expected_unsupported:
                raise ValueError("partial unsupported claims disagree with verifier claims")
    return payload


def _validate_claims(
    claims: list[object],
    evidence_ids: set[str],
    *,
    allow_diagnostic_citations: bool = False,
) -> None:
    claim_ids = set()
    for claim in claims:
        if type(claim) is not dict:
            raise ValueError("claim is malformed")
        for field in ("claim_id", "claim", "reason"):
            if type(claim.get(field)) is not str or not claim[field].strip():
                raise ValueError(f"claim {field} must be non-empty text")
        if claim["claim_id"] in claim_ids:
            raise ValueError("claim IDs must be unique")
        claim_ids.add(claim["claim_id"])
        if type(claim.get("central")) is not bool:
            raise ValueError("claim central must be a boolean")
        verdict = claim.get("verdict")
        if verdict not in _VERDICTS:
            raise ValueError("claim has an invalid verdict")
        cited = claim.get("evidence_ids")
        if type(cited) is not list or any(type(value) is not str for value in cited):
            raise ValueError("claim evidence_ids must be an array of strings")
        if len(cited) != len(set(cited)) or len(cited) > 2:
            raise ValueError("claim evidence_ids must be unique and bounded")
        if verdict == "not_found":
            if cited:
                raise ValueError("not_found claim cannot cite evidence")
        elif not cited:
            raise ValueError("evidence-bearing claim has no evidence")
        if allow_diagnostic_citations:
            if any(not _SPAN_ID.fullmatch(value) for value in cited):
                raise ValueError("unsupported diagnostic claim has an invalid diagnostic span ID")
        elif unknown := set(cited) - evidence_ids:
            raise ValueError(f"claim cites unknown evidence IDs {sorted(unknown)}")


def _validate_evidence(item: object) -> None:
    if type(item) is not dict:
        raise ValueError("evidence hit is malformed")
    for field in ("id", "video_id", "title", "text"):
        if type(item.get(field)) is not str or not item[field].strip():
            raise ValueError(f"evidence {field} must be non-empty text")
    match = _SPAN_ID.fullmatch(item["id"])
    if not match or match.group(1) != item["video_id"]:
        raise ValueError("evidence has an invalid span ID")
    start, end = _finite_number(item.get("start"), "evidence start"), _finite_number(item.get("end"), "evidence end")
    if start < 0 or start >= end:
        raise ValueError("evidence has invalid timing")
    if (int(round(start * 1000)), int(round(end * 1000))) != (int(match.group(2)), int(match.group(3))):
        raise ValueError("evidence span ID timing does not match metadata")
    _finite_number(item.get("retrieval_score"), "evidence retrieval_score")
    _finite_number(item.get("score"), "evidence score")
    if len(item["text"]) > 1800:
        raise ValueError("evidence text exceeds 1800 characters")
    source_ids = item.get("source_chunk_ids")
    if type(source_ids) is not list or not source_ids:
        raise ValueError("evidence source_chunk_ids must be a non-empty array")
    if any(type(value) is not str or not value or value.startswith("card::") for value in source_ids):
        raise ValueError("card chunk cannot appear in evidence")
    origins = item.get("query_origins")
    if type(origins) is not list or not origins or any(type(value) is not str or not value for value in origins):
        raise ValueError("evidence query_origins must be a non-empty array of strings")


def _validate_hint(hint: object) -> None:
    if type(hint) is not dict:
        raise ValueError("hint hit is malformed")
    for field in ("card_id", "question", "take"):
        if type(hint.get(field)) is not str or not hint[field].strip():
            raise ValueError(f"hint {field} must be non-empty text")
    if not hint["card_id"].startswith("card::"):
        raise ValueError("hint card_id must identify a card")
    _finite_number(hint.get("retrieval_score"), "hint retrieval_score")


def _finite_number(value: object, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric metadata")
    return float(value)


def validate_answer_token_cap(value: object) -> int:
    if type(value) is not int:
        raise ValueError(f"answer token cap must be an integer, got {value!r}")
    if not MIN_ANSWER_TOKEN_CAP <= value <= MAX_ANSWER_TOKEN_CAP:
        raise ValueError(
            "answer token cap must be between "
            f"{MIN_ANSWER_TOKEN_CAP} and {MAX_ANSWER_TOKEN_CAP}, got {value}"
        )
    return value


def render_context_markdown(payload: Mapping[str, object]) -> str:
    """Show verifier inputs and hints without ever treating hints as evidence."""
    lines = [f"**Verifier:** `{payload['status']}`"]
    claims = payload.get("claims", [])
    if claims:
        lines.append("\n**Claims**")
        for claim in claims:
            lines.append(f"- `{claim['claim_id']}` `{claim['verdict']}` — {claim['claim']}")
    evidence = payload.get("evidence_hits", [])
    if evidence:
        lines.append("\n**Injected clean evidence**")
        for span in evidence:
            second = int(span["start"])
            title = str(span.get("title") or span["video_id"]).replace("[", "(").replace("]", ")")
            url = f"https://www.youtube.com/watch?v={span['video_id']}&t={second}s"
            lines.append(f"- [{title} @ {_timestamp(second)}]({url}) — {span['text'][:180]}")
    hints = payload.get("hint_hits", [])
    if hints:
        lines.append("\n**search hints — not evidence**")
        for hint in hints:
            lines.append(f"- `{hint['card_id']}` {hint['question']}")
    unsupported = payload.get("unsupported_claims", [])
    if unsupported:
        lines.append("\n**Unsupported portions**")
        lines.extend(f"- {claim}" for claim in unsupported)
    diagnostics = payload.get("diagnostics")
    if payload["status"] == "error" and diagnostics:
        lines.append("\n**Failure diagnostics**")
        lines.append(f"```json\n{json.dumps(diagnostics, ensure_ascii=False, indent=2)}\n```")
    return "\n".join(lines)


def chat_turn_with_trace(
    message: str,
    history: list[dict[str, str]] | None,
    model: str,
    answer_mode: AnswerMode,
    retrieval_mode: str,
    *,
    answer_token_cap: int = DEFAULT_ANSWER_TOKEN_CAP,
    retrieve_fn: Callable[[str, str], dict[str, object]] = retrieve_decision,
    answer_post: Callable[..., Mapping[str, object]] = ollama_json,
    clock: Callable[[], float] = time.monotonic,
    ensure_absent: Callable[[str], None] = _ensure_absent,
) -> ChatTurnResult:
    """Generate and expose the exact structured retrieval used for this answer."""
    answer_mode = validate_answer_mode(answer_mode)
    answer_token_cap = validate_answer_token_cap(answer_token_cap)
    message = (message or "").strip()
    if not message:
        return ChatTurnResult(
            history or [],
            "",
            "",
            {"status": "error", "evidence_count": 0},
            "no_input",
        )
    prefix = ""
    fallback_warning = ""
    trace: dict[str, object] = {"status": "error", "evidence_count": 0}
    if answer_mode == "free":
        trace = {"status": "free", "evidence_count": 0}
        user_message = message
        context = "_RAG kapalı: serbest üretim._"
        mode_label = "serbest (RAG kapalı)"
    else:
        try:
            ensure_absent(model)
        except Exception as exc:
            context = f"**Verifier:** `error`\n\n`{type(exc).__name__}: {exc}`"
            return ChatTurnResult(
                _append(history, message, _ERROR_TR),
                context,
                "",
                trace,
                "retrieval_error",
            )
        try:
            payload = validate_v2_response(retrieve_fn(message, retrieval_mode))
        except Exception as exc:
            context = f"**Verifier:** `error`\n\n`{type(exc).__name__}: {exc}`"
            if answer_mode == "grounded_strict":
                return ChatTurnResult(
                    _append(history, message, _ERROR_TR),
                    context,
                    "",
                    trace,
                    "retrieval_error",
                )
            user_message = message
            mode_label = "unverified fallback"
            fallback_warning = _UNVERIFIED_ERROR
        else:
            context = render_context_markdown(payload)
            status = payload["status"]
            trace = {"status": status, "evidence_count": 0}
            if status == "unsupported":
                if answer_mode == "grounded_strict":
                    return ChatTurnResult(
                        _append(history, message, REFUSAL_TR),
                        context,
                        "",
                        trace,
                        "policy_refusal",
                    )
                user_message = message
                mode_label = "unverified fallback"
                fallback_warning = _UNVERIFIED_UNSUPPORTED
            elif status == "error":
                if answer_mode == "grounded_strict":
                    return ChatTurnResult(
                        _append(history, message, _ERROR_TR),
                        context,
                        "",
                        trace,
                        "retrieval_error",
                    )
                user_message = message
                mode_label = "unverified fallback"
                fallback_warning = _UNVERIFIED_ERROR
            else:
                evidence_spans = [
                    span["text"] for span in payload["evidence_hits"][:5]
                ]
                trace["evidence_count"] = len(evidence_spans)
                if status == "partial":
                    prefix = _PARTIAL_PREFIX.format(
                        claims="; ".join(payload["unsupported_claims"])
                    )
                user_message = build_grounded_user(
                    payload["answer_question"],
                    evidence_spans,
                    max_spans=5,
                    span_char_cap=1800,
                )
                mode_label = "kanıtlı" if status == "answerable" else "kısmi kanıtlı"
        try:
            verifier_model = (
                _RUNTIME_SETTINGS.verifier_model
                if _RUNTIME_SETTINGS.packaged
                else load_settings().rag.verifier_model
            )
            ensure_absent(verifier_model)
        except Exception as exc:
            context = f"**Verifier:** `error`\n\n`{type(exc).__name__}: {exc}`"
            return ChatTurnResult(
                _append(history, message, _ERROR_TR),
                context,
                "",
                {"status": "error", "evidence_count": 0},
                "retrieval_error",
            )
    body = {
        "model": model,
        "stream": False,
        "think": False,
        "keep_alive": -1,
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": user_message},
        ],
        "options": {
            "temperature": 0.7,
            "top_p": 0.85,
            "repeat_penalty": 1.2,
            "num_predict": answer_token_cap,
            "num_ctx": 8192,
        },
    }
    try:
        started = clock()
        result = answer_post("POST", OLLAMA_CHAT, payload=body, timeout_s=600.0)
        generated = result["message"]["content"].strip()
        if not generated:
            result = answer_post("POST", OLLAMA_CHAT, payload=body, timeout_s=600.0)
            generated = result["message"]["content"].strip()
        if generated:
            generation_status: GenerationStatus = "generated"
            wall = clock() - started
            warning = (
                f'<div class="fallback-warning">⚠ {fallback_warning}</div>'
                if fallback_warning
                else ""
            )
            meta = warning + (
                f'<div class="meta"><b>{model.split(":")[0]}</b> · {mode_label} · '
                f'{result.get("eval_count", "?")} token · {wall:.0f}s</div>'
            )
        else:
            generated = _EMPTY_OUTPUT_TR
            meta = ""
            generation_status = "empty_output"
    except Exception as exc:
        generated = f"[LLM error: {exc}]"
        meta = ""
        generation_status = "model_error"
    if prefix:
        meta = (
            f'<div class="partial-warning">⚠ {prefix.strip()}</div>' + meta
        )
    return ChatTurnResult(
        _append(history, message, generated),
        context,
        meta,
        trace,
        generation_status,
    )


def chat_turn(
    message: str,
    history: list[dict[str, str]] | None,
    model: str,
    answer_mode: AnswerMode,
    retrieval_mode: str,
    *,
    answer_token_cap: int = DEFAULT_ANSWER_TOKEN_CAP,
    retrieve_fn: Callable[[str, str], dict[str, object]] = retrieve_decision,
    answer_post: Callable[..., Mapping[str, object]] = ollama_json,
    clock: Callable[[], float] = time.monotonic,
    ensure_absent: Callable[[str], None] = _ensure_absent,
) -> tuple[list[dict[str, str]], str, str]:
    """Keep the stable public UI tuple while the server consumes the trace."""
    result = chat_turn_with_trace(
        message,
        history,
        model,
        answer_mode,
        retrieval_mode,
        answer_token_cap=answer_token_cap,
        retrieve_fn=retrieve_fn,
        answer_post=answer_post,
        clock=clock,
        ensure_absent=ensure_absent,
    )
    return result.history, result.context, result.metadata


def _append(history, question, answer):
    return (history or []) + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def _timestamp(seconds: int) -> str:
    if seconds >= 3600:
        return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60:02d}:{seconds % 60:02d}"
