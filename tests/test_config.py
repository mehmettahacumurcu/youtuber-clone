def test_new_config_fields(settings):
    assert settings.paths.filter_dir.name == "filter"
    assert settings.paths.dataset_dir.name == "dataset"
    assert settings.assemble.skip_gap_s == 1.0
