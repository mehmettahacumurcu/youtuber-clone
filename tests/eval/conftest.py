"""Real-corpus evaluation requires separately supplied private fixtures."""
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
collect_ignore = ['test_card_eval_prompt.py', 'test_card_eval_report.py', 'test_card_eval_resume.py', 'test_card_eval_retrieval.py', 'test_card_eval_score.py', 'test_rag_reliability_fixture.py', 'test_rag_reliability_gate.py', 'test_rag_reliability_runner.py', 'test_rag_reliability_schema.py', 'test_verified_card_fixture.py'] if not all(
    (_ROOT / "eval" / "fixtures" / name).is_file()
    for name in ("verified_card_eval_v1.json", "rag_reliability_v1.json")
) else []
