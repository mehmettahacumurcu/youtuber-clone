import json

import pytest
import requests

from pipeline.config import RagConfig
from rag.types import EvidenceSpan
from rag.verifier import (
    ClaimVerdict,
    EvidenceVerifier,
    VerifierInvalidOutput,
    VerifierTransportError,
)


def _span(span_id="span::v1::1000::2000"):
    return EvidenceSpan(
        span_id, "v1", "Başlık", 1.0, 2.0, "temiz kanıt", 2.0,
        ("v1::0",), ("original",),
    )


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload

    @property
    def headers(self):
        return {"Content-Length": str(len(json.dumps(self.payload).encode("utf-8")))}

    def iter_content(self, chunk_size):
        assert chunk_size == 8192
        yield json.dumps(self.payload).encode("utf-8")

    def close(self):
        pass


class BrokenJsonResponse(FakeResponse):
    def json(self):
        raise ValueError("invalid top-level JSON")

    def iter_content(self, chunk_size):
        assert chunk_size == 8192
        yield b"not-json"


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def request(self, method, url, *, json, timeout, **kwargs):
        self.calls.append((url, json, timeout, kwargs))
        if self.error is not None:
            raise self.error
        self.response.url = url
        return self.response


def _assistant(content, **metrics):
    return FakeResponse({"message": {"role": "assistant", "content": content}, **metrics})


def _valid_content(evidence_id="S1", verdict="entailed"):
    evidence = [] if verdict == "not_found" else [evidence_id]
    return json.dumps({
        "claims": [{
            "claim_id": "C1",
            "claim": "Merkez iddia",
            "central": True,
            "verdict": verdict,
            "evidence_ids": evidence,
            "reason": "Kanıt bu iddiayı doğrudan ele alıyor.",
        }],
    })


def test_exact_native_ollama_request_and_diagnostics():
    session = FakeSession(_assistant(
        _valid_content("S1"), load_duration=3, prompt_eval_count=40, eval_count=12,
    ))
    clock_values = iter([10.0, 12.5])
    verifier = EvidenceVerifier.from_config(
        RagConfig(), session=session, clock=lambda: next(clock_values),
    )
    result = verifier.verify("Soru?", (_span(),))
    assert result.output.claims[0].verdict is ClaimVerdict.ENTAILED
    assert len(session.calls) == 1
    url, body, timeout, options = session.calls[0]
    assert url == "http://127.0.0.1:11434/api/chat"
    assert timeout == 120
    assert options == {"allow_redirects": False, "stream": True}
    assert body["model"] == "qwen3:4b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["keep_alive"] == 0
    assert isinstance(body["format"], dict)
    assert body["options"] == {
        "temperature": 0.0,
        "seed": 42,
        "num_ctx": 8192,
        "num_predict": 1200,
    }
    serialized_messages = json.dumps(body["messages"], ensure_ascii=False)
    assert "temiz kanıt" in serialized_messages
    packet = json.loads(body["messages"][1]["content"].rsplit("\n\n", 1)[1])
    assert packet["evidence"][0]["id"] == "S1"
    assert "span::v1::1000::2000" not in body["messages"][1]["content"]
    assert "outside knowledge" in serialized_messages
    assert "one to six claims and mark one to six as central" in serialized_messages
    assert (
        "For a pure information-seeking question with no asserted answer, emit exactly one "
        "central claim describing the requested information. Do not invent or enumerate "
        "possible answers. If the evidence does not supply the requested information, mark "
        "that central claim not_found."
        in serialized_messages
    )
    assert result.diagnostics["wall_s"] == 2.5
    assert result.diagnostics["prompt_eval_count"] == 40
    assert result.diagnostics["eval_count"] == 12


def test_redirect_response_is_reported_as_a_verifier_transport_error():
    verifier = EvidenceVerifier.from_config(RagConfig(), session=FakeSession(FakeResponse({}, status=307)))

    with pytest.raises(VerifierTransportError, match="redirect"):
        verifier.verify("Soru?", (_span(),))


def test_verifier_uses_injected_bounded_json_transport_without_a_session():
    calls = []

    def request_json(method, url, *, payload, timeout_s):
        calls.append((method, url, payload, timeout_s))
        return _assistant(_valid_content()).json()

    verifier = EvidenceVerifier.from_config(RagConfig(), request_json=request_json)
    verifier.verify("Soru?", (_span(),))

    assert calls[0][0:2] == ("POST", "http://127.0.0.1:11434/api/chat")
    assert calls[0][3] == 120


def test_system_prompt_centralizes_direct_corrections_without_promoting_related_facts():
    session = FakeSession(_assistant(_valid_content("S1")))
    verifier = EvidenceVerifier.from_config(RagConfig(), session=session)

    verifier.verify(
        "Did Merkel make a secret admission, and what is the real explanation?",
        (_span(),),
    )

    system_message = session.calls[0][1]["messages"][0]["content"]
    assert (
        "If the question asks whether a premise is true and requests the real explanation, mark "
        "as central both the claim that evaluates that premise and any evidence-backed corrective "
        "or explanatory claim that directly answers the request"
        in system_message
    )
    assert "Do not mark merely related facts as central" in system_message


def test_alias_citation_is_mapped_back_to_exact_source_span_id():
    first = _span("span::v1::1000::2000")
    second = _span("span::v2::3000::4000")
    session = FakeSession(_assistant(_valid_content("S2")))

    result = EvidenceVerifier.from_config(RagConfig(), session=session).verify(
        "Soru?", (first, second),
    )

    body = session.calls[0][1]
    evidence = json.loads(body["messages"][1]["content"].rsplit("\n\n", 1)[1])["evidence"]
    assert [item["id"] for item in evidence] == ["S1", "S2"]
    assert result.output.claims[0].evidence_ids == [second.id]
    assert result.diagnostics["verifier_output"]["claims"][0]["evidence_ids"] == [second.id]


def test_not_found_evidence_alias_is_discarded_before_strict_validation():
    content = json.dumps({"claims": [{
        "claim_id": "C1", "claim": "Eksik iddia", "central": True,
        "verdict": "not_found", "evidence_ids": ["S1"], "reason": "Kanıt yok.",
    }]})

    result = EvidenceVerifier.from_config(
        RagConfig(), session=FakeSession(_assistant(content)),
    ).verify("Soru?", (_span(),))

    assert result.output.claims[0].evidence_ids == []


def test_not_found_unknown_alias_is_rejected_before_recovery():
    content = json.dumps({"claims": [{
        "claim_id": "C1", "claim": "Eksik iddia", "central": True,
        "verdict": "not_found", "evidence_ids": ["S6"], "reason": "Kanıt yok.",
    }]})

    with pytest.raises(VerifierInvalidOutput, match="unknown evidence aliases"):
        EvidenceVerifier.from_config(
            RagConfig(), session=FakeSession(_assistant(content)),
        ).verify("Soru?", (_span(),))


def test_not_found_duplicate_aliases_are_rejected_before_recovery():
    content = json.dumps({"claims": [{
        "claim_id": "C1", "claim": "Eksik iddia", "central": True,
        "verdict": "not_found", "evidence_ids": ["S1", "S1"], "reason": "Kanıt yok.",
    }]})

    with pytest.raises(VerifierInvalidOutput, match="evidence IDs must be unique within a claim"):
        EvidenceVerifier.from_config(
            RagConfig(), session=FakeSession(_assistant(content)),
        ).verify("Soru?", (_span(),))


def test_not_found_recovery_preserves_entailment_alias_mapping():
    first = _span("span::v1::1000::2000")
    second = _span("span::v2::3000::4000")
    content = json.dumps({"claims": [
        {
            "claim_id": "C1", "claim": "Eksik iddia", "central": True,
            "verdict": "not_found", "evidence_ids": ["S1"], "reason": "Kanıt yok.",
        },
        {
            "claim_id": "C2", "claim": "Desteklenen iddia", "central": True,
            "verdict": "entailed", "evidence_ids": ["S2"], "reason": "Kanıt var.",
        },
    ]})

    result = EvidenceVerifier.from_config(
        RagConfig(), session=FakeSession(_assistant(content)),
    ).verify("Soru?", (first, second))

    assert result.output.claims[0].evidence_ids == []
    assert result.output.claims[1].evidence_ids == [second.id]


def test_unknown_evidence_alias_is_rejected_before_real_id_validation():
    session = FakeSession(_assistant(_valid_content("S6")))

    with pytest.raises(VerifierInvalidOutput, match="unknown evidence aliases"):
        EvidenceVerifier.from_config(RagConfig(), session=session).verify("Soru?", (_span(),))


@pytest.mark.parametrize("verdict", [
    "entailed", "contradicted", "explicitly_unresolved", "not_found",
])
def test_all_verdicts_parse_with_their_valid_evidence_cardinality(verdict):
    session = FakeSession(_assistant(_valid_content(verdict=verdict)))
    result = EvidenceVerifier.from_config(RagConfig(), session=session).verify("Soru?", (_span(),))
    assert result.output.claims[0].verdict.value == verdict


def test_arbitrary_nonblank_claim_labels_are_renumbered_without_changing_claims():
    labels = ["analysis", "step-two", "42", "C0", "not-an-evidence-id"]
    claims = [
        {
            "claim_id": label,
            "claim": f"claim {index}",
            "central": index == 1,
            "verdict": "entailed",
            "evidence_ids": ["S1"],
            "reason": f"reason {index}",
        }
        for index, label in enumerate(labels, start=1)
    ]
    result = EvidenceVerifier.from_config(
        RagConfig(), session=FakeSession(_assistant(json.dumps({"claims": claims}))),
    ).verify("Soru?", (_span(),))

    normalized = result.output.model_dump(mode="json")["claims"]
    assert [claim.pop("claim_id") for claim in normalized] == ["C1", "C2", "C3", "C4", "C5"]
    expected = [{key: value for key, value in claim.items() if key != "claim_id"} for claim in claims]
    for claim in expected:
        claim["evidence_ids"] = ["span::v1::1000::2000"]
    assert normalized == expected


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"claims": []}),
        json.dumps({"claims": [{
            "claim_id": "C1", "claim": "x", "central": True, "verdict": "entailed",
            "evidence_ids": ["unknown"], "reason": "r",
        }]}),
    ],
)
def test_malformed_or_semantically_invalid_output_is_rejected(content):
    session = FakeSession(_assistant(content))
    with pytest.raises(VerifierInvalidOutput):
        EvidenceVerifier.from_config(RagConfig(), session=session).verify("Soru?", (_span(),))
    assert len(session.calls) == 1


def test_duplicate_evidence_ids_are_rejected():
    duplicate = json.dumps({"claims": [{
        "claim_id": "C1", "claim": "x", "central": True, "verdict": "entailed",
        "evidence_ids": ["span::v1::1000::2000", "span::v1::1000::2000"], "reason": "r",
    }]})
    with pytest.raises(VerifierInvalidOutput):
        EvidenceVerifier.from_config(
            RagConfig(), session=FakeSession(_assistant(duplicate)),
        ).verify("Soru?", (_span(),))


def test_six_central_claims_are_accepted():
    claims = [
        {
            "claim_id": f"C{i}", "claim": f"claim {i}", "central": True,
            "verdict": "entailed", "evidence_ids": ["S1"], "reason": "supported",
        }
        for i in range(1, 7)
    ]

    result = EvidenceVerifier.from_config(
        RagConfig(), session=FakeSession(_assistant(json.dumps({"claims": claims}))),
    ).verify("Soru?", (_span(),))

    assert len(result.output.claims) == 6
    assert all(claim.central for claim in result.output.claims)


def test_zero_central_not_found_claims_recover_requested_information_as_unsupported():
    question = "  Hangi sicaklikta servis edilmeli?  "
    claims = [
        {
            "claim_id": f"candidate-{i}", "claim": f"invented candidate {i}",
            "central": False, "verdict": "not_found", "evidence_ids": [],
            "reason": "not present",
        }
        for i in range(1, 3)
    ]
    content = json.dumps({"claims": claims})

    result = EvidenceVerifier.from_config(
        RagConfig(), session=FakeSession(_assistant(content)),
    ).verify(question, (_span(),))

    assert len(result.output.claims) == 1
    recovered = result.output.claims[0]
    assert recovered.claim_id == "C1"
    assert recovered.claim == question.strip()
    assert recovered.central is True
    assert recovered.verdict is ClaimVerdict.NOT_FOUND
    assert recovered.evidence_ids == []
    assert result.diagnostics["raw_content"] == content


def test_zero_central_evidence_bearing_claim_is_rejected():
    claims = [{
        "claim_id": "C1", "claim": "claim 1", "central": False,
        "verdict": "entailed", "evidence_ids": ["S1"], "reason": "supported",
    }]

    with pytest.raises(VerifierInvalidOutput):
        EvidenceVerifier.from_config(
            RagConfig(), session=FakeSession(_assistant(json.dumps({"claims": claims}))),
        ).verify("Soru?", (_span(),))


def test_evidence_packet_above_five_aliases_is_rejected_before_http_request():
    spans = tuple(_span(f"span::v1::{index}::0") for index in range(1, 7))
    session = FakeSession(_assistant(_valid_content()))
    with pytest.raises(VerifierInvalidOutput, match="five-span alias limit"):
        EvidenceVerifier.from_config(
            RagConfig(), session=session,
        ).verify("Soru?", spans)
    assert session.calls == []


@pytest.mark.parametrize("central", [1, "true", "yes"])
def test_coercible_non_boolean_central_values_are_rejected(central):
    content = json.dumps({"claims": [{
        "claim_id": "C1", "claim": "x", "central": central, "verdict": "entailed",
        "evidence_ids": ["span::v1::1000::2000"], "reason": "r",
    }]})
    with pytest.raises(VerifierInvalidOutput):
        EvidenceVerifier.from_config(
            RagConfig(), session=FakeSession(_assistant(content)),
        ).verify("Soru?", (_span(),))


@pytest.mark.parametrize("error", [requests.Timeout("slow"), requests.ConnectionError("down")])
def test_transport_failure_is_typed_and_never_retried(error):
    session = FakeSession(error=error)
    with pytest.raises(VerifierTransportError):
        EvidenceVerifier.from_config(RagConfig(), session=session).verify("Soru?", (_span(),))
    assert len(session.calls) == 1


def test_http_error_is_transport_failure_and_never_retried():
    session = FakeSession(FakeResponse({}, status=503))
    with pytest.raises(VerifierTransportError):
        EvidenceVerifier.from_config(RagConfig(), session=session).verify("Soru?", (_span(),))
    assert len(session.calls) == 1


def test_non_json_is_transport_failure_and_non_assistant_protocol_is_invalid_output():
    broken = FakeSession(BrokenJsonResponse(None))
    with pytest.raises(VerifierTransportError):
        EvidenceVerifier.from_config(RagConfig(), session=broken).verify("Soru?", (_span(),))
    wrong_role = FakeSession(FakeResponse({
        "message": {"role": "user", "content": _valid_content()},
    }))
    with pytest.raises(VerifierInvalidOutput):
        EvidenceVerifier.from_config(RagConfig(), session=wrong_role).verify("Soru?", (_span(),))


def test_empty_packet_makes_zero_http_calls():
    session = FakeSession(_assistant(_valid_content()))
    with pytest.raises(VerifierInvalidOutput, match="empty evidence packet"):
        EvidenceVerifier.from_config(RagConfig(), session=session).verify("Soru?", ())
    assert session.calls == []
