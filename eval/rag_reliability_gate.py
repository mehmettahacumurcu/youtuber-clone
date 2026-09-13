"""Acceptance scoring for production RAG reliability records."""
from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from eval.rag_reliability_schema import ReliabilityCase, ReliabilitySuite


@dataclass(frozen=True)
class CaseGateResult:
    case_id: str
    passed: bool
    failures: tuple[str, ...]
    status: str
    covered_groups: tuple[str, ...]


@dataclass(frozen=True)
class GateSummary:
    passed: bool
    results: tuple[CaseGateResult, ...]
    repeat_failures: tuple[str, ...]


def evaluate_case_record(
    case: ReliabilityCase,
    record: dict,
    suite: ReliabilitySuite,
    repo_root: Path,
) -> CaseGateResult:
    """Apply the frozen acceptance contract to one persisted live decision."""
    failures: list[str] = []
    decision = record.get("decision", {})
    status = decision.get("status")
    if status != case.expected_status:
        failures.append(f"expected {case.expected_status}, got {status}")
    central_verdicts = {
        claim.get("verdict") for claim in decision.get("claims", []) if claim.get("central") is True
    }
    for verdict in case.required_central_verdicts:
        if verdict not in central_verdicts:
            failures.append(f"missing required central verdict {verdict}")
    spans = decision.get("selected_spans", [])
    hints = decision.get("hint_hits", [])
    hint_takes = {hint.get("take") for hint in hints if type(hint) is dict}
    for span in spans:
        if any(str(source).startswith("card::") for source in span.get("source_chunk_ids", [])):
            failures.append(f"card chunk entered evidence in {case.id}")
        if span.get("text") in hint_takes:
            failures.append(f"card take entered evidence in {case.id}")
        reconstruction_error = _verify_span_reconstruction(span, repo_root)
        if reconstruction_error:
            failures.append(reconstruction_error)
    groups = {group.id: group for group in suite.evidence_groups}
    covered: list[str] = []
    for group_id in case.required_evidence_group_ids:
        group = groups[group_id]
        if any(_span_covers_window(span, window) for span in spans for window in group.windows):
            covered.append(group_id)
        else:
            failures.append(f"required evidence group not covered: {group_id}")
    if case.expected_status == "unsupported" and spans:
        failures.append("unsupported case selected evidence")
    return CaseGateResult(case.id, not failures, tuple(failures), str(status), tuple(covered))


def evaluate_repeat(case_id: str, baseline: dict, repeated: dict) -> tuple[str, ...]:
    """Ensure a repeat preserves its answer status and required evidence coverage."""
    failures = []
    if baseline.get("status") != repeated.get("status"):
        failures.append(f"{case_id} repeat status changed")
    if set(baseline.get("covered_groups", [])) != set(repeated.get("covered_groups", [])):
        failures.append(f"{case_id} repeat evidence coverage changed")
    return tuple(failures)


def evaluate_run(suite, run_dir: Path, repo_root: Path) -> GateSummary:
    """Evaluate all case and repeat records without rerunning live retrieval."""
    run_dir = Path(run_dir)
    results = []
    summaries = {}
    by_id = {case.id: case for case in suite.cases}
    for case in suite.cases:
        path = run_dir / "cases" / f"{case.id}.json"
        if not path.is_file():
            result = CaseGateResult(case.id, False, ("missing case result",), "missing", ())
        else:
            result = evaluate_case_record(
                case, json.loads(path.read_text(encoding="utf-8")), suite, repo_root,
            )
        results.append(result)
        summaries[case.id] = {"status": result.status, "covered_groups": list(result.covered_groups)}
    repeat_failures = []
    for case_id in suite.repeat_case_ids:
        path = run_dir / "repeats" / f"{case_id}.json"
        if not path.is_file():
            repeat_failures.append(f"missing repeat result {case_id}")
            continue
        repeated_result = evaluate_case_record(
            by_id[case_id], json.loads(path.read_text(encoding="utf-8")), suite, repo_root,
        )
        if not repeated_result.passed:
            repeat_failures.extend(
                f"{case_id} repeat gate: {failure}" for failure in repeated_result.failures
            )
        repeat_failures.extend(evaluate_repeat(
            case_id,
            summaries[case_id],
            {"status": repeated_result.status, "covered_groups": list(repeated_result.covered_groups)},
        ))
    return GateSummary(
        all(result.passed for result in results) and not repeat_failures,
        tuple(results),
        tuple(repeat_failures),
    )


def write_report(summary: GateSummary, output: Path) -> None:
    """Write a compact human-readable acceptance report."""
    lines = [
        "# RAG Reliability Gate", "",
        f"Overall: **{'PASS' if summary.passed else 'FAIL'}**", "",
        "| Case | Status | Gate | Failures |", "|---|---|---|---|",
    ]
    for result in summary.results:
        lines.append(
            f"| {result.case_id} | {result.status} | {'PASS' if result.passed else 'FAIL'} | "
            f"{'<br>'.join(result.failures) or '—'} |"
        )
    if summary.repeat_failures:
        lines.extend(["", "## Repeat failures", ""])
        lines.extend(f"- {failure}" for failure in summary.repeat_failures)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _span_covers_window(span, window) -> bool:
    if span.get("video_id") != window.video_id:
        return False
    overlap = min(float(span.get("end", -math.inf)), window.end) - max(
        float(span.get("start", math.inf)), window.start,
    )
    return overlap > 0


def _verify_span_reconstruction(span, repo_root: Path) -> str | None:
    path = Path(repo_root) / "data/clean" / f"{span.get('video_id')}.json"
    if not path.is_file():
        return f"selected span source missing: {path.name}"
    document = json.loads(path.read_text(encoding="utf-8"))
    start, end = float(span.get("start")), float(span.get("end"))
    selected = [
        segment for segment in document.get("segments", [])
        if float(segment["start"]) >= start - 1e-6 and float(segment["end"]) <= end + 1e-6
    ]
    if (
        not selected
        or abs(float(selected[0]["start"]) - start) > 1e-6
        or abs(float(selected[-1]["end"]) - end) > 1e-6
    ):
        return f"selected span timestamps are not segment-aligned: {span.get('id')}"
    text = unicodedata.normalize(
        "NFC",
        " ".join(
            (segment.get("text") or "").strip()
            for segment in selected
            if (segment.get("text") or "").strip()
        ),
    )
    if text != span.get("text"):
        return f"selected span text does not reconstruct: {span.get('id')}"
    return None
