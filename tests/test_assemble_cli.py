import json

from pipeline.assemble.__main__ import main
from pipeline.assemble.runner import dataset_path_for


def test_cli_assembles_corpus(settings):
    (settings.paths.filter_dir / "v1.json").write_text(
        json.dumps({"video_id": "v1", "segments": [
            {"start": 0.0, "end": 2.0, "duration_s": 2.0, "text": "merhaba"}
        ]}), encoding="utf-8")
    cfg = settings.paths.data_dir.parent / "config.yaml"
    rc = main(["--config", str(cfg)])
    assert rc == 0
    lines = dataset_path_for(settings).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["text"] == "merhaba"


def test_cli_errors_when_no_filter_outputs(settings):
    cfg = settings.paths.data_dir.parent / "config.yaml"
    assert main(["--config", str(cfg)]) == 2
