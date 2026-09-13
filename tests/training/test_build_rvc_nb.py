from __future__ import annotations

import json
from pathlib import Path

from training.build_rvc_nb import build_notebook, write_notebook


def _cell_source(cell: dict) -> str:
    source = cell.get("source", [])
    return source if isinstance(source, str) else "".join(source)


def _notebook_source(notebook: dict) -> str:
    return "\n".join(_cell_source(cell) for cell in notebook["cells"])


def test_notebook_pins_applio_and_never_launches_ui() -> None:
    source = _notebook_source(build_notebook())

    assert "3e5e248c2bd003f65f3e128dd12677533def27be" in source
    assert "git rev-parse HEAD" in source
    assert "core.py\", \"prerequisites" in source
    assert "app.py" not in source
    assert "gradio" not in source.casefold()
    assert "share=True" not in source


def test_notebook_defaults_to_approved_poc_training_contract() -> None:
    source = _notebook_source(build_notebook())

    for expected in (
        "RUN_FULL_DATASET = False",
        "SAMPLE_RATE = 32000",
        "TOTAL_EPOCHS = 200",
        "BATCH_SIZE = 8",
        "SAVE_EVERY_EPOCH = 25",
        'F0_METHOD = "rmvpe"',
        'EMBEDDER = "contentvec"',
        'VOCODER = "HiFi-GAN"',
        '"--cut_preprocess", "Skip"',
        '"--process_effects", "False"',
        '"--noise_reduction", "False"',
        '"--normalization_mode", "none"',
        '"--save_only_latest", "False"',
        '"--save_every_weights", "True"',
        '"--pretrained", "True"',
    ):
        assert expected in source


def test_notebook_persists_state_and_builds_controlled_comparisons() -> None:
    source = _notebook_source(build_notebook())

    assert "/content/drive/MyDrive/speaker_rvc" in source
    assert "os.symlink" in source
    assert "CHOSEN_EPOCH = 150" in source
    assert "AUDITION_EPOCHS = (100, 150, 200)" in source
    assert "INDEX_RATES = (0.25, 0.50, 0.75)" in source
    assert '"--pitch", "0"' in source
    assert '"--protect", "0.5"' in source
    assert '"--volume_envelope", "0.8"' in source
    assert '"--f0_autotune", "False"' in source
    assert "conversion_manifest.json" in source


def test_notebook_has_ordered_stages_and_gpu_metadata() -> None:
    notebook = build_notebook()
    source_by_cell = [_cell_source(cell) for cell in notebook["cells"]]
    stage_markers = [
        "1. Configuration",
        "2. Mount Drive and verify GPU",
        "3. Install pinned Applio",
        "4. Extract and verify experiment bundle",
        "5. Bind resumable logs to Drive",
        "6. Preprocess curated audio",
        "7. Extract RMVPE and ContentVec features",
        "8. Train RVC v2 and build index",
        "9. Verify and export artifacts",
        "10. Audition epochs 100, 150, and 200",
        "11. Choose epoch and sweep index rates",
        "12. Compute metrics and listening sheet",
        "13. Export results ZIP",
    ]
    positions = [
        next(index for index, source in enumerate(source_by_cell) if marker in source)
        for marker in stage_markers
    ]
    assert positions == sorted(positions)
    assert notebook["metadata"]["accelerator"] == "GPU"
    assert notebook["nbformat"] == 4


def test_notebook_fails_closed_on_runtime_bundle_and_artifact_errors() -> None:
    source = _notebook_source(build_notebook())

    assert "CUDA GPU is required" in source
    assert "manifest hash mismatch" in source
    assert "archive.testzip()" in source
    assert "No RVC inference weights found" in source
    assert "No FAISS index found" in source
    assert "Requested epoch" in source
    assert "available epochs" in source
    assert "speaker_rvc_poc_v1_results.zip" in source


def test_notebook_excludes_generator_and_discriminator_training_checkpoints() -> None:
    source = _notebook_source(build_notebook())

    assert 'not path.name.startswith(("G_", "D_"))' in source
    assert "Training checkpoints remain in Drive logs for resume" in source


def test_notebook_isolates_model_metrics_and_keeps_local_fallback() -> None:
    source = _notebook_source(build_notebook())

    assert "/content/rvc-eval-env" in source
    assert "--system-site-packages" in source
    assert "speechbrain<1.0" in source
    assert "faster-whisper>=1.2.1" in source
    assert "--allow-model-failure" in source
    assert "training.rvc_eval" in source
    assert "LOCAL EVALUATION FALLBACK" in source


def test_write_notebook_round_trips_json(tmp_path: Path) -> None:
    output = write_notebook(tmp_path / "nested" / "rvc.ipynb")
    parsed = json.loads(output.read_text(encoding="utf-8"))

    assert parsed == build_notebook()
    assert len(parsed["cells"]) >= 20
