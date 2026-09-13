import json

from pipeline.filter.__main__ import main
from pipeline.filter.runner import filter_path_for


def test_cli_filters_one_video(settings):
    vid = "vidcli"
    (settings.paths.transcribe_dir / f"{vid}.json").write_text(
        json.dumps({"video_id": vid, "segments": [
            {"start": 0.0, "end": 5.0, "duration_s": 5.0, "text": "selam", "avg_logprob": -0.2}
        ]}), encoding="utf-8")
    cfg = settings.paths.data_dir.parent / "config.yaml"
    rc = main(["--config", str(cfg), "--video-id", vid])
    assert rc == 0
    assert filter_path_for(vid, settings).exists()


def test_cli_errors_when_nothing_to_filter(settings):
    cfg = settings.paths.data_dir.parent / "config.yaml"
    assert main(["--config", str(cfg), "--all"]) == 2
