import hashlib

from rag.prompt import (
    SYS,
    _GROUNDED_HEADER,
    Decision,
    REFUSAL_TR,
    build_grounded_user,
    build_prompt,
    format_citations,
    format_evidence_citations,
    gate,
)
from rag.types import Chunk, EvidenceSpan, Hit


def _hit(text, score, vid="v1", start=83.0, title="Bir Video"):
    return Hit(
        chunk=Chunk(
            id=f"{vid}::0", video_id=vid, start=start, end=start + 5,
            title=title, text=text,
        ),
        rerank_score=score,
    )


def test_gate_grounds_above_threshold():
    decision = gate([_hit("content", 2.0)], threshold=0.0, ungrounded=False)
    assert isinstance(decision, Decision)
    assert decision.grounded and not decision.flagged and len(decision.hits) == 1


def test_gate_refuses_below_threshold_by_default():
    decision = gate([_hit("content", -3.0)], threshold=0.0, ungrounded=False)
    assert not decision.grounded and not decision.flagged and decision.hits == []


def test_gate_refuses_on_empty_hits():
    assert not gate([], threshold=0.0, ungrounded=False).grounded


def test_gate_ungrounded_flag_allows_below_threshold_flagged():
    decision = gate([_hit("content", -3.0)], threshold=0.0, ungrounded=True)
    assert decision.grounded and decision.flagged and decision.hits == []


def test_build_prompt_has_chunks_then_question_no_reference_marker():
    prompt = build_prompt("What happened?", [_hit("first", 2.0), _hit("second", 1.5)])
    assert "first" in prompt and "second" in prompt and "What happened?" in prompt
    assert "[Reference" not in prompt and "Reference:" not in prompt
    assert prompt.index("first") < prompt.index("What happened?")


def test_format_citations_renders_timestamp_and_url():
    citations = format_citations([_hit("x", 2.0, vid="abc123", start=83.0)])
    assert "Bir Video" in citations
    assert "01:23" in citations
    assert "watch?v=abc123&t=83s" in citations


def test_format_citations_dedupes_same_video_timestamp():
    hit = _hit("x", 2.0, vid="abc", start=10.0)
    assert format_citations([hit, hit]).count("watch?v=abc") == 1


def test_refusal_message_is_present():
    assert REFUSAL_TR


def test_canonical_prompt_bytes_are_frozen():
    assert hashlib.sha256(SYS.encode("utf-8")).hexdigest() == (
        "a9cae890ec6f08b37da95dd287fd6bfea2ab06c63a8a13909a4630aa7920170b"
    )
    assert hashlib.sha256(_GROUNDED_HEADER.encode("utf-8")).hexdigest() == (
        "eb9b6ac9eece6e0faae80bbeee8f66d9bbe44f032f5032187f0d025a705ad0e0"
    )
    assert hashlib.sha256(build_grounded_user("Q", ["S"]).encode("utf-8")).hexdigest() == (
        "005014e546751a14e28f4cca31a5ff3fbeac98fce39ab5632a9bbe9ca07cd0cc"
    )


def test_evidence_citations_use_exact_span_start_and_dedupe():
    span = EvidenceSpan(
        "span::abc::83000::90000", "abc", "Bir Video", 83.0, 90.0, "metin",
        2.0, ("abc::0",), ("original",),
    )
    citations = format_evidence_citations((span, span))
    assert citations.count("watch?v=abc&t=83s") == 1
    assert "01:23" in citations and "Bir Video" in citations
