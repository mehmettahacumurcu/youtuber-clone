import sys

from colab.run_batch import STAGES, commands_for, select_batch


def test_select_batch_picks_first_n_untranscribed(tmp_path):
    trn = tmp_path / "transcribe"
    trn.mkdir()
    (trn / "b.json").write_text("{}", encoding="utf-8")
    assert select_batch(["a", "b", "c", "d"], trn, 2) == ["a", "c"]


def test_commands_for_covers_all_stages():
    cmds = commands_for("vid1", config_path=None)
    assert len(cmds) == len(STAGES)
    # each command runs under the SAME interpreter as the driver (not a bare "python")
    assert all(c[0] == sys.executable for c in cmds)
    # download is first, filter is last, each targets the right module + video id
    assert cmds[0][1:3] == ["-m", "pipeline.download"]
    assert cmds[-1][1:3] == ["-m", "pipeline.filter"]
    assert all(c[-2:] == ["--video-id", "vid1"] for c in cmds)


def test_commands_for_includes_config_when_given():
    cmds = commands_for("vid1", config_path="config.yaml")
    assert ["--config", "config.yaml"] == cmds[0][3:5]
