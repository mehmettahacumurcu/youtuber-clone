from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from eval.card_eval import (
    _build_unit_spec,
    _ensure_run_manifest,
    _execute_unit,
    expand_stage,
    main,
    result_path,
    run_fixed_case,
    write_atomic_json,
)
from eval.card_eval_ollama import (
    EmptyContentError,
    GenerationResult,
    GenerationSettings,
    ModelIdentity,
    OllamaTransportError,
    ThinkingLeakError,
)
from eval.card_eval_schema import load_suite


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_PATH = REPO_ROOT / "eval" / "fixtures" / "verified_card_eval_v1.json"
DIGEST = "bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8"


@pytest.fixture(scope="module")
def suite():
    return load_suite(SUITE_PATH)


@pytest.fixture(scope="module")
def g01(suite):
    return next(case for case in suite.cases if case.id == "G01")


def _identity(*, digest=DIGEST):
    return ModelIdentity(
        name="qwen3:14b",
        digest=digest,
        ollama_version="0.31.2",
        format="gguf",
        family="qwen3",
        parameter_size="14.8B",
        quantization_level="Q4_K_M",
        file_type=15,
        quantization_version=2,
    )


def _diagnostic_settings(*, seed=20260711, num_predict=384):
    return GenerationSettings(
        temperature=0,
        seed=seed,
        top_p=1.0,
        top_k=1,
        repeat_penalty=1.05,
        num_ctx=4096,
        num_predict=num_predict,
    )


def _result(*, done_reason="stop", eval_count=20):
    return GenerationResult(
        content="Mutlak butlan dönüşü kabul ettirilemez; suç örgütü kısmı bir hikâye.",
        model="qwen3:14b",
        done=True,
        done_reason=done_reason,
        prompt_eval_count=120,
        eval_count=eval_count,
    )


class FakeClient:
    def __init__(self, identity, outcomes=None):
        self.identity = identity
        self.outcomes = list(outcomes or [_result()])
        self.calls = []

    def generate(self, messages, settings):
        self.calls.append((messages, settings))
        if not self.outcomes:
            raise AssertionError("unexpected generation")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_atomic_json_round_trips_turkish_and_leaves_no_temp(tmp_path):
    path = tmp_path / "sonuç.json"
    write_atomic_json(path, {"cevap": "güya değil; düşünüyorum"})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "cevap": "güya değil; düşünüyorum"
    }
    assert path.read_bytes().endswith(b"\n")
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_replace_failure_preserves_existing_destination(tmp_path, monkeypatch):
    path = tmp_path / "result.json"
    path.write_text('{"old":true}\n', encoding="utf-8")
    original = path.read_bytes()
    real_replace = Path.replace

    def fail_for_temp(self, target):
        if self.suffix == ".tmp":
            raise OSError("replace failed")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_for_temp)
    with pytest.raises(OSError, match="replace failed"):
        write_atomic_json(path, {"new": True})

    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_stage_expansion_and_paths_are_exact(suite, tmp_path):
    diagnostic = expand_stage(suite, "fixed-diagnostic")
    production = expand_stage(suite, "fixed-production")

    assert len(diagnostic) == 30
    assert [(unit.case.id, unit.arm, unit.seed) for unit in diagnostic[:4]] == [
        ("G01", "semantic_only", 20260711),
        ("G01", "semantic_plus_style", 20260711),
        ("G02", "semantic_only", 20260711),
        ("G02", "semantic_plus_style", 20260711),
    ]
    assert len(production) == 45
    assert {unit.arm for unit in production} == {"semantic_plus_style"}
    assert [unit.seed for unit in production[:3]] == [
        20260711,
        20260712,
        20260713,
    ]
    assert result_path(
        tmp_path, "fixed-diagnostic", "G01", "semantic_only", 20260711
    ) == (
        tmp_path / "fixed" / "diagnostic" / "G01" / "semantic_only" / "20260711.json"
    )
    assert result_path(
        tmp_path, "fixed-production", "G01", "semantic_plus_style", 20260712
    ) == (
        tmp_path
        / "fixed"
        / "production"
        / "G01"
        / "semantic_plus_style"
        / "20260712.json"
    )


def test_run_fixed_case_builds_prompt_once_and_diagnoses_natural_stop(g01):
    identity = _identity()
    client = FakeClient(identity)
    result = run_fixed_case(
        g01,
        "semantic_plus_style",
        20260711,
        client,
        identity,
        _diagnostic_settings(),
    )

    assert result.status == "complete"
    assert result.question_id == "G01"
    assert result.machine_diagnostics.natural_stop
    assert result.generation_result is not None
    assert len(client.calls) == 1
    assert client.calls[0][0] == list(result.messages)


@pytest.mark.parametrize(
    ("failure", "expected_flag"),
    [
        (EmptyContentError("empty"), "empty_answer"),
        (
            ThinkingLeakError(
                "leak", content="yanıt", thinking="gizli muhakeme"
            ),
            "think_leak",
        ),
    ],
)
def test_terminal_output_failures_become_cacheable_records(
    g01, failure, expected_flag
):
    identity = _identity()
    result = run_fixed_case(
        g01,
        "semantic_only",
        20260711,
        FakeClient(identity, [failure]),
        identity,
        _diagnostic_settings(),
    )
    assert result.status == "terminal_output_failure"
    assert expected_flag in result.machine_diagnostics.blocking_reasons
    assert result.terminal_error["type"] == type(failure).__name__


def _spec(suite, g01, tmp_path, *, identity=None, settings=None):
    return _build_unit_spec(
        suite=suite,
        case=g01,
        stage="fixed-diagnostic",
        arm="semantic_only",
        seed=(settings or _diagnostic_settings()).seed,
        identity=identity or _identity(),
        settings=settings or _diagnostic_settings(),
        destination=result_path(
            tmp_path,
            "fixed-diagnostic",
            "G01",
            "semantic_only",
            (settings or _diagnostic_settings()).seed,
        ),
    )


def test_matching_cache_skips_generation_and_identity_changes_rerun(
    suite, g01, tmp_path
):
    spec = _spec(suite, g01, tmp_path)
    client = FakeClient(spec.identity, [_result()])
    first = _execute_unit(spec, client, resume=True)
    assert first.disposition == "ran"
    assert len(client.calls) == 1

    skip_client = FakeClient(spec.identity, [])
    skipped = _execute_unit(spec, skip_client, resume=True)
    assert skipped.disposition == "skipped"
    assert not skip_client.calls

    changed_identity = _identity(digest="a" * 64)
    stale = _spec(suite, g01, tmp_path, identity=changed_identity)
    rerun_client = FakeClient(changed_identity, [_result()])
    rerun = _execute_unit(stale, rerun_client, resume=True)
    assert rerun.disposition == "ran"
    assert len(rerun_client.calls) == 1
    assert json.loads(stale.destination.read_text(encoding="utf-8"))[
        "model_identity"
    ]["digest"] == "a" * 64


@pytest.mark.parametrize(
    "change",
    ["seed", "settings", "prompt", "evidence"],
)
def test_seed_settings_prompt_and_evidence_changes_invalidate_cache(
    suite, g01, tmp_path, change
):
    base = _spec(suite, g01, tmp_path)
    _execute_unit(base, FakeClient(base.identity, [_result()]), resume=True)

    case = g01
    settings = _diagnostic_settings()
    seed = settings.seed
    if change == "seed":
        settings = _diagnostic_settings(seed=20260712)
        seed = settings.seed
    elif change == "settings":
        settings = _diagnostic_settings(num_predict=385)
    elif change in {"prompt", "evidence"}:
        first = g01.cards[0]
        changed_card = replace(
            first,
            take=first.take + (
                " Ek prompt." if change == "prompt" else " Ek kanıt."
            ),
        )
        case = replace(g01, cards=(changed_card, *g01.cards[1:]))

    changed = _build_unit_spec(
        suite=suite,
        case=case,
        stage="fixed-diagnostic",
        arm="semantic_only",
        seed=seed,
        identity=base.identity,
        settings=settings,
        destination=base.destination,
    )
    client = FakeClient(base.identity, [_result()])
    assert _execute_unit(changed, client, resume=True).disposition == "ran"
    assert len(client.calls) == 1


def test_corrupt_cache_is_quarantined_and_current_unit_reruns(suite, g01, tmp_path):
    spec = _spec(suite, g01, tmp_path)
    spec.destination.parent.mkdir(parents=True, exist_ok=True)
    spec.destination.write_bytes(b'{"broken":')
    client = FakeClient(spec.identity, [_result()])

    outcome = _execute_unit(spec, client, resume=True)

    assert outcome.disposition == "ran"
    assert len(client.calls) == 1
    quarantined = list(spec.destination.parent.glob("20260711.json.invalid-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b'{"broken":'
    assert json.loads(spec.destination.read_text(encoding="utf-8"))["cache_key"] == (
        spec.cache_key
    )


def test_structurally_invalid_cache_is_quarantined_not_skipped(suite, g01, tmp_path):
    spec = _spec(suite, g01, tmp_path)
    _execute_unit(spec, FakeClient(spec.identity, [_result()]), resume=True)
    raw = json.loads(spec.destination.read_text(encoding="utf-8"))
    raw["generation_result"] = "not-an-object"
    write_atomic_json(spec.destination, raw)
    client = FakeClient(spec.identity, [_result()])

    assert _execute_unit(spec, client, resume=True).disposition == "ran"
    assert len(client.calls) == 1
    assert len(list(spec.destination.parent.glob("20260711.json.invalid-*"))) == 1


def test_cached_diagnostics_must_match_exact_answer_and_completion_metadata(
    suite, g01, tmp_path
):
    spec = _spec(suite, g01, tmp_path)
    _execute_unit(spec, FakeClient(spec.identity, [_result()]), resume=True)
    raw = json.loads(spec.destination.read_text(encoding="utf-8"))
    raw["generation_result"]["done_reason"] = "length"
    raw["generation_result"]["eval_count"] = spec.settings.num_predict
    assert raw["machine_diagnostics"]["natural_stop"] is True
    write_atomic_json(spec.destination, raw)
    client = FakeClient(spec.identity, [_result()])

    assert _execute_unit(spec, client, resume=True).disposition == "ran"
    assert len(client.calls) == 1
    assert len(list(spec.destination.parent.glob("20260711.json.invalid-*"))) == 1


def test_terminal_and_length_failures_are_cached_without_resampling(
    suite, g01, tmp_path
):
    terminal = _spec(suite, g01, tmp_path)
    _execute_unit(
        terminal,
        FakeClient(terminal.identity, [EmptyContentError("empty")]),
        resume=True,
    )
    assert _execute_unit(
        terminal, FakeClient(terminal.identity, []), resume=True
    ).disposition == "skipped"

    length_settings = _diagnostic_settings(num_predict=384)
    length_spec = _build_unit_spec(
        suite=suite,
        case=g01,
        stage="fixed-diagnostic",
        arm="semantic_plus_style",
        seed=length_settings.seed,
        identity=terminal.identity,
        settings=length_settings,
        destination=result_path(
            tmp_path,
            "fixed-diagnostic",
            "G01",
            "semantic_plus_style",
            length_settings.seed,
        ),
    )
    _execute_unit(
        length_spec,
        FakeClient(
            length_spec.identity,
            [_result(done_reason="length", eval_count=384)],
        ),
        resume=True,
    )
    raw = json.loads(length_spec.destination.read_text(encoding="utf-8"))
    assert raw["machine_diagnostics"]["token_cap_stop"] is True
    assert _execute_unit(
        length_spec, FakeClient(length_spec.identity, []), resume=True
    ).disposition == "skipped"


def test_cache_cannot_relabel_infrastructure_failure_as_terminal(
    suite, g01, tmp_path
):
    spec = _spec(suite, g01, tmp_path)
    _execute_unit(
        spec,
        FakeClient(spec.identity, [EmptyContentError("empty")]),
        resume=True,
    )
    raw = json.loads(spec.destination.read_text(encoding="utf-8"))
    raw["terminal_error"]["type"] = "OllamaTransportError"
    write_atomic_json(spec.destination, raw)
    client = FakeClient(spec.identity, [_result()])

    assert _execute_unit(spec, client, resume=True).disposition == "ran"
    assert len(client.calls) == 1
    assert len(list(spec.destination.parent.glob("20260711.json.invalid-*"))) == 1


def test_thinking_leak_cache_recomputes_diagnostics_from_visible_content(
    suite, g01, tmp_path
):
    spec = _spec(suite, g01, tmp_path)
    leak = ThinkingLeakError(
        "leak", content="Hakan Fidan 777 dedi", thinking="private"
    )
    _execute_unit(spec, FakeClient(spec.identity, [leak]), resume=True)
    raw = json.loads(spec.destination.read_text(encoding="utf-8"))
    assert raw["terminal_content"] == "Hakan Fidan 777 dedi"
    raw["machine_diagnostics"]["unexpected_numbers_dates"] = ["forged"]
    raw["machine_diagnostics"]["token_cap_stop"] = True
    write_atomic_json(spec.destination, raw)
    client = FakeClient(spec.identity, [_result()])

    assert _execute_unit(spec, client, resume=True).disposition == "ran"
    assert len(client.calls) == 1
    assert len(list(spec.destination.parent.glob("20260711.json.invalid-*"))) == 1


def test_cache_read_io_error_aborts_without_quarantining_valid_result(
    suite, g01, tmp_path, monkeypatch
):
    spec = _spec(suite, g01, tmp_path)
    _execute_unit(spec, FakeClient(spec.identity, [_result()]), resume=True)
    original = spec.destination.read_bytes()
    real_read_bytes = Path.read_bytes

    def deny_result(self):
        if self == spec.destination:
            raise PermissionError("sharing violation")
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", deny_result)
    with pytest.raises(PermissionError, match="sharing violation"):
        _execute_unit(spec, FakeClient(spec.identity, []), resume=True)

    assert spec.destination.exists()
    assert real_read_bytes(spec.destination) == original
    assert not list(spec.destination.parent.glob("20260711.json.invalid-*"))


def test_failed_unit_does_not_modify_completed_sibling(suite, g01, tmp_path):
    first = _spec(suite, g01, tmp_path)
    _execute_unit(first, FakeClient(first.identity, [_result()]), resume=True)
    sibling_bytes = first.destination.read_bytes()

    second_case = suite.cases[1]
    second = _build_unit_spec(
        suite=suite,
        case=second_case,
        stage="fixed-diagnostic",
        arm="semantic_only",
        seed=20260711,
        identity=first.identity,
        settings=_diagnostic_settings(),
        destination=result_path(
            tmp_path,
            "fixed-diagnostic",
            second_case.id,
            "semantic_only",
            20260711,
        ),
    )
    with pytest.raises(OllamaTransportError):
        _execute_unit(
            second,
            FakeClient(first.identity, [OllamaTransportError("offline")]),
            resume=True,
        )

    assert first.destination.read_bytes() == sibling_bytes
    assert not second.destination.exists()


def test_offline_validate_never_constructs_ollama_client(monkeypatch, capsys):
    class ForbiddenClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("offline validation contacted Ollama")

    monkeypatch.setattr("eval.card_eval.OllamaClient", ForbiddenClient)
    exit_code = main(
        ["validate", "--suite", str(SUITE_PATH), "--offline"]
    )
    assert exit_code == 0
    assert "15 cases; 0 provenance errors" in capsys.readouterr().out


def test_incompatible_existing_run_manifest_is_refused_without_replacement(
    suite, tmp_path
):
    run_path = tmp_path / "run.json"
    stale = {
        "run_schema": 1,
        "suite_path": str(SUITE_PATH.resolve()),
        "suite_id": suite.suite_id,
        "suite_sha256": "a" * 64,
        "model_identity": _identity().to_json(),
    }
    write_atomic_json(run_path, stale)
    original = run_path.read_bytes()

    with pytest.raises(ValueError, match="incompatible"):
        _ensure_run_manifest(run_path, suite, SUITE_PATH, _identity())
    assert run_path.read_bytes() == original
