from pipeline.download.runner import _id_from_url


def test_id_from_watch_url():
    assert _id_from_url("https://www.youtube.com/watch?v=sample00059") == "sample00059"


def test_id_from_url_with_extra_params():
    assert _id_from_url("https://www.youtube.com/watch?v=abc123XYZ_-&t=42s") == "abc123XYZ_-"


def test_id_from_url_none_when_missing():
    assert _id_from_url("https://www.youtube.com/playlist?list=PL123") is None
