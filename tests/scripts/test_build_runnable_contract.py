from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "scripts/build_runnable.ps1").read_text(encoding="utf-8")


def test_runnable_copies_complete_v2_runtime_graph():
    for path in (
        "candidates.py", "context.py", "verifier.py", "evidence_pipeline.py",
        "card_manifest.py", "chat_runtime.py",
    ):
        assert path in SCRIPT


def test_runnable_stages_before_atomic_swap_and_keeps_rollback():
    assert "$Stage" in SCRIPT
    assert "Move-Item" in SCRIPT
    assert "$Backup" in SCRIPT
    assert "PlanOnly" in SCRIPT


def test_runnable_treats_card_catalog_and_manifest_as_one_pair():
    assert "speaker_cards.json" in SCRIPT
    assert "speaker_cards.index_manifest.json" in SCRIPT
    assert "relocate_card_manifest" in SCRIPT


def test_runnable_restores_backup_when_stage_transition_fails():
    transition = SCRIPT[SCRIPT.index("if (Test-Path -LiteralPath $Destination)"):]

    assert "try {" in transition
    assert "Move-Item -LiteralPath $Stage -Destination $Destination" in transition
    assert "catch {" in transition
    assert "Move-Item -LiteralPath $Backup -Destination $Destination" in transition
    assert "previous runnable restored" in transition
