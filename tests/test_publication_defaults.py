"""Public source workflows must not publish generated datasets by default."""
from colab.run_corpus import _build_parser as corpus_parser
from colab.run_stages import _build_parser as stages_parser


def test_corpus_git_checkpoint_is_explicit_opt_in():
    assert corpus_parser().parse_args([]).no_checkpoint is True
    assert corpus_parser().parse_args(["--checkpoint"]).no_checkpoint is False


def test_stages_git_checkpoint_is_explicit_opt_in():
    assert stages_parser().parse_args([]).no_checkpoint is True
    assert stages_parser().parse_args(["--checkpoint"]).no_checkpoint is False
