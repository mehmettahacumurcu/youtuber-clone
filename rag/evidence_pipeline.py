"""Production evidence orchestration and deterministic answer authorization."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from pipeline.config import Settings
from rag.candidates import collect_candidates
from rag.card_manifest import CardHintStatus, validate_card_hint_manifest
from rag.context import resolve_evidence_spans
from rag.embed import get_reranker
from rag.retrieve import retrieve
from rag.types import CandidatePacket, CardHint, EvidenceSpan, Hit
from rag.verifier import (
    ClaimVerdict,
    EvidenceVerifier,
    VerifierClaim,
    VerifierInvalidOutput,
    VerifierTransportError,
)


DecisionStatus = Literal["answerable", "partial", "unsupported", "error"]
FailureReason = Literal[
    "retrieval_error",
    "source_error",
    "verifier_transport_error",
    "verifier_invalid_output",
    "verifier_error",
    "evidence_budget_error",
]


@dataclass(frozen=True)
class EvidenceDecision:
    status: DecisionStatus
    claims: tuple[VerifierClaim, ...]
    selected_spans: tuple[EvidenceSpan, ...]
    hint_hits: tuple[CardHint, ...]
    answer_question: str | None
    unsupported_claims: tuple[str, ...]
    failure_reason: FailureReason | None
    diagnostics: Mapping[str, object]


class EvidenceBudgetError(RuntimeError):
    """Verifier citations cannot fit the locked answer evidence budget."""


def derive_status(claims: Sequence[VerifierClaim]) -> DecisionStatus:
    central = [claim for claim in claims if claim.central]
    if not central:
        raise ValueError("a verifier result must contain a central claim")
    found = [claim for claim in central if claim.verdict is not ClaimVerdict.NOT_FOUND]
    if not found:
        return "unsupported"
    if len(found) != len(central):
        return "partial"
    return "answerable"


def select_evidence(
    claims: Sequence[VerifierClaim],
    spans: Sequence[EvidenceSpan],
    *,
    max_spans: int,
) -> tuple[EvidenceSpan, ...]:
    rank = {span.id: index for index, span in enumerate(spans)}
    cited = list(dict.fromkeys(
        evidence_id for claim in claims for evidence_id in claim.evidence_ids
    ))
    if len(cited) > max_spans:
        raise EvidenceBudgetError(
            f"verifier cited {len(cited)} unique spans but answer budget is {max_spans}"
        )
    selected: list[str] = []
    for claim in claims:
        if not claim.central or claim.verdict is ClaimVerdict.NOT_FOUND:
            continue
        best = min(claim.evidence_ids, key=lambda evidence_id: rank[evidence_id])
        if best not in selected:
            selected.append(best)
    for evidence_id in cited:
        if evidence_id not in selected:
            selected.append(evidence_id)
    selected.sort(key=rank.__getitem__)
    return tuple(spans[rank[evidence_id]] for evidence_id in selected)


def build_partial_question(claims: Sequence[VerifierClaim]) -> str:
    supported = [
        claim.claim.strip() for claim in claims
        if claim.central and claim.verdict is not ClaimVerdict.NOT_FOUND
    ]
    if not supported:
        raise ValueError("partial question requires a supported central claim")
    numbered = " ".join(f"{index}) {claim}" for index, claim in enumerate(supported, 1))
    return f"Şu doğrulanabilen kısımları yanıtla: {numbered}"


def assess_question(
    question: str,
    settings: Settings,
    *,
    verifier: EvidenceVerifier | None = None,
    retrieval_mode: Literal["clean", "clean_with_card_hints"] | None = None,
    retrieve_fn: Callable[..., list[Hit]] = retrieve,
    collect_fn: Callable[..., CandidatePacket] = collect_candidates,
    resolve_fn: Callable[..., tuple[EvidenceSpan, ...]] = resolve_evidence_spans,
    card_manifest_validator: Callable[..., CardHintStatus] = validate_card_hint_manifest,
    score_segments: Callable[[str, list[str]], Sequence[float]] | None = None,
) -> EvidenceDecision:
    rag = settings.rag.model_copy(
        update={"retrieval_mode": retrieval_mode or settings.rag.retrieval_mode},
    )
    diagnostics: dict[str, object] = {"retrieval_mode": rag.retrieval_mode}
    hints: tuple[CardHint, ...] = ()
    try:
        packet = collect_fn(
            question, rag, retrieve_fn=retrieve_fn,
            card_manifest_validator=card_manifest_validator,
        )
        hints = packet.hint_hits
        diagnostics["candidates"] = dict(packet.diagnostics)
    except Exception as exc:
        return _error("retrieval_error", exc, hints, diagnostics)
    if not packet.candidates:
        return _unsupported(hints, diagnostics)

    try:
        scorer = score_segments
        if scorer is None:
            scorer = get_reranker(rag.reranker).score
        spans = resolve_fn(
            question,
            packet.candidates,
            settings.paths.clean_dir,
            score_segments=scorer,
            max_spans=min(rag.verifier_candidate_max, rag.answer_span_max),
            max_chars_per_span=rag.verifier_span_char_cap,
            max_seconds_per_span=rag.verifier_span_seconds_cap,
        )
    except Exception as exc:
        return _error("source_error", exc, hints, diagnostics)
    diagnostics["resolved_span_count"] = len(spans)
    if not spans:
        return _unsupported(hints, diagnostics)

    try:
        active_verifier = verifier or EvidenceVerifier.from_config(rag)
        result = active_verifier.verify(question, spans)
    except VerifierTransportError as exc:
        return _error("verifier_transport_error", exc, hints, diagnostics)
    except VerifierInvalidOutput as exc:
        return _error("verifier_invalid_output", exc, hints, diagnostics)
    except Exception as exc:
        return _error("verifier_error", exc, hints, diagnostics)
    try:
        diagnostics["verifier"] = dict(result.diagnostics)
        claims = tuple(result.output.claims)
        status = derive_status(claims)
        if status == "unsupported":
            return EvidenceDecision(
                "unsupported", claims, (), hints, None,
                tuple(claim.claim for claim in claims if claim.central),
                None, diagnostics,
            )
        try:
            selected = select_evidence(claims, spans, max_spans=rag.answer_span_max)
        except (EvidenceBudgetError, KeyError, ValueError) as exc:
            return _error("evidence_budget_error", exc, hints, diagnostics, claims)
        if any(len(span.text) > rag.answer_span_char_cap for span in selected):
            return _error(
                "evidence_budget_error", ValueError("selected span exceeds answer char cap"),
                hints, diagnostics, claims,
            )
        if sum(len(span.text) for span in selected) > rag.answer_context_char_cap:
            return _error(
                "evidence_budget_error", ValueError("selected context exceeds total char cap"),
                hints, diagnostics, claims,
            )
        unsupported = tuple(
            claim.claim for claim in claims
            if claim.central and claim.verdict is ClaimVerdict.NOT_FOUND
        )
        answer_question = question.strip() if status == "answerable" else build_partial_question(claims)
        return EvidenceDecision(
            status, claims, selected, hints, answer_question, unsupported, None, diagnostics,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        return _error("verifier_invalid_output", exc, hints, diagnostics)
    except Exception as exc:
        return _error("verifier_error", exc, hints, diagnostics)


def _unsupported(hints, diagnostics) -> EvidenceDecision:
    return EvidenceDecision("unsupported", (), (), hints, None, (), None, diagnostics)


def _error(reason, exc, hints, diagnostics, claims=()) -> EvidenceDecision:
    merged = dict(diagnostics)
    merged["error"] = f"{type(exc).__name__}: {exc}"
    return EvidenceDecision("error", tuple(claims), (), hints, None, (), reason, merged)
