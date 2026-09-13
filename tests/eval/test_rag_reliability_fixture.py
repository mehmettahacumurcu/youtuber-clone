from pathlib import Path

from eval.build_rag_reliability_fixture import build_fixture


ROOT = Path(__file__).resolve().parents[2]


def test_committed_fixture_is_reproducible_byte_for_byte(tmp_path):
    generated = tmp_path / "rag_reliability_v1.json"
    build_fixture(
        ROOT / "eval/fixtures/verified_card_eval_v1.json",
        generated,
    )
    committed = ROOT / "eval/fixtures/rag_reliability_v1.json"
    assert generated.read_bytes() == committed.read_bytes()
