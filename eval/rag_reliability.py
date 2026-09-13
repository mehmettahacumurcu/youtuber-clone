"""Validate, run, report, and smoke-test the production RAG reliability suite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.rag_reliability_gate import evaluate_run, write_report
from eval.rag_reliability_runner import run_suite
from eval.rag_reliability_schema import load_reliability_suite, validate_repository_sources
from pipeline.config import load_settings
from ui.chat_runtime import chat_turn, retrieve_decision


ROOT = Path(__file__).resolve().parents[1]


class _CountingAnswer:
    """In-memory answer client that makes server smoke side effects observable."""

    def __init__(self) -> None:
        self.bodies: list[object] = []

    def __call__(self, method, url, *, payload, timeout_s) -> dict[str, object]:
        self.bodies.append(payload)
        return {"message": {"content": "smoke answer"}, "eval_count": 1}


def smoke_server(suite, base_url: str, output: Path) -> bool:
    """Check server response policy without sending any generation to Ollama."""
    counter = _CountingAnswer()
    rows = []
    for case in suite.smoke_cases:
        payload = retrieve_decision(
            case.question,
            case.retrieval_mode,
            url=base_url.rstrip("/") + "/retrieve",
        )
        before = len(counter.bodies)
        chat_turn(
            case.question,
            [],
            "counting-answer-model",
            "grounded_strict",
            case.retrieval_mode,
            retrieve_fn=lambda query, mode, payload=payload: payload,
            answer_post=counter,
            ensure_absent=lambda model: None,
        )
        after = len(counter.bodies)
        expected_call = case.expected_status in {"answerable", "partial"}
        row_pass = payload["status"] == case.expected_status and (after > before) == expected_call
        if case.expected_status == "partial" and after > before:
            body_text = json.dumps(counter.bodies[-1], ensure_ascii=False)
            row_pass = row_pass and all(
                claim not in body_text for claim in payload["unsupported_claims"]
            )
        rows.append({
            "id": case.id, "expected": case.expected_status, "actual": payload["status"],
            "answer_calls": after - before, "passed": row_pass,
        })
    result = {"schema_version": 1, "passed": all(row["passed"] for row in rows), "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result["passed"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "run", "report", "smoke-server"])
    parser.add_argument("--suite", type=Path, default=Path("eval/fixtures/rag_reliability_v1.json"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("results/rag_reliability/current"))
    parser.add_argument("--base-url", default="http://127.0.0.1:7670")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    suite = load_reliability_suite(args.suite)
    if args.command == "validate":
        errors = validate_repository_sources(suite, ROOT)
        if errors:
            print("\n".join(errors))
            return 1
        print(f"VALID: {len(suite.cases)} cases, {len(suite.evidence_groups)} evidence groups")
        return 0
    if args.command == "run":
        run_suite(
            suite=suite, suite_path=args.suite, settings=load_settings(args.config),
            run_dir=args.run_dir, resume=args.resume,
        )
        return 0
    if args.command == "report":
        summary = evaluate_run(suite, args.run_dir, ROOT)
        output = args.output or args.run_dir / "report.md"
        write_report(summary, output)
        print(f"{'PASS' if summary.passed else 'FAIL'}: {output}")
        return 0 if summary.passed else 1
    output = args.output or args.run_dir / "server-smoke.json"
    return 0 if smoke_server(suite, args.base_url, output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
