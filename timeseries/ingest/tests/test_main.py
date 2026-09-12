from types import SimpleNamespace

import pytest

from cog_stac_pipeline import main
from cog_stac_pipeline.manifest import ManifestVariable


class FakeDataset:
    def __init__(self, band_count):
        self.RasterCount = band_count

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def test_read_input_band_counts_uses_raster_headers(monkeypatch):
    datasets = {
        "/data/ppt.tif": FakeDataset(1423),
        "/data/tmax.tif": FakeDataset(1420),
    }
    monkeypatch.setattr(main.gdal, "Open", datasets.get)
    monkeypatch.setattr(main.fs_utils, "to_vsi", lambda path: path)

    counts = main.read_input_band_counts(
        [
            ManifestVariable("ppt", "/data/ppt.tif"),
            ManifestVariable("tmax", "/data/tmax.tif"),
        ],
        "prism",
    )

    assert counts == {"ppt": 1423, "tmax": 1420}


def test_read_input_band_counts_reports_unopenable_variable(monkeypatch):
    monkeypatch.setattr(main.gdal, "Open", lambda _path: None)
    monkeypatch.setattr(main.fs_utils, "to_vsi", lambda path: path)

    with pytest.raises(ValueError) as error:
        main.read_input_band_counts(
            [ManifestVariable("tmin", "/data/tmin.tif")], "prism"
        )

    message = str(error.value)
    assert "dataset 'prism'" in message
    assert "variable 'tmin'" in message
    assert "/data/tmin.tif" in message


def test_read_input_band_counts_rejects_zero_band_raster(monkeypatch):
    monkeypatch.setattr(main.gdal, "Open", lambda _path: FakeDataset(0))
    monkeypatch.setattr(main.fs_utils, "to_vsi", lambda path: path)

    with pytest.raises(ValueError, match="invalid band count 0"):
        main.read_input_band_counts(
            [ManifestVariable("empty", "/data/empty.tif")], "test-dataset"
        )


def test_pipeline_validates_temporal_coverage_before_creating_output(monkeypatch):
    config = SimpleNamespace(
        dataset_time_delta={"months": 1},
        metadata_file_path="metadata.yml",
        dataset_name="prism",
        dataset_start_datetime=None,
        input_manifest_path="prism.yml",
        preflight_only=False,
    )
    dataset_metadata = {
        "timespan": {
            "resolution": {"months": 1},
            "period": {"gte": "1895-01", "lte": "2013-07"},
        }
    }
    variables = [ManifestVariable("tmax", "/data/tmax.tif")]

    monkeypatch.setattr(main, "configure_gdal", lambda: None)
    monkeypatch.setattr(
        main.metadata,
        "load_and_verify_metadata",
        lambda *_args: ([dataset_metadata], dataset_metadata),
    )
    monkeypatch.setattr(main.metadata, "validate_else_add_timespan", lambda *_args: False)
    monkeypatch.setattr(main, "resolve_input_variables", lambda _config: variables)
    monkeypatch.setattr(main, "validate_manifest_metadata", lambda *_args: None)
    monkeypatch.setattr(main, "read_input_band_counts", lambda *_args: {"tmax": 1420})
    monkeypatch.setattr(
        main.metadata,
        "validate_else_add_temporal_end",
        lambda *_args: (_ for _ in ()).throw(ValueError("temporal mismatch")),
    )
    monkeypatch.setattr(
        main.fs_utils,
        "makedirs",
        lambda _path: pytest.fail("output directory created before preflight"),
    )

    with pytest.raises(ValueError, match="temporal mismatch"):
        main.run_pipeline(config)


def test_preflight_only_returns_without_creating_output(monkeypatch):
    config = SimpleNamespace(
        dataset_time_delta={"months": 1},
        metadata_file_path="metadata.yml",
        dataset_name="prism",
        dataset_start_datetime=None,
        input_manifest_path="prism.yml",
        preflight_only=True,
    )
    dataset_metadata = {
        "timespan": {
            "resolution": {"months": 1},
            "period": {"gte": "1895-01", "lte": "2013-07"},
        }
    }
    variables = [ManifestVariable("ppt", "/data/ppt.tif")]

    monkeypatch.setattr(main, "configure_gdal", lambda: None)
    monkeypatch.setattr(
        main.metadata,
        "load_and_verify_metadata",
        lambda *_args: ([dataset_metadata], dataset_metadata),
    )
    monkeypatch.setattr(main.metadata, "validate_else_add_timespan", lambda *_args: False)
    monkeypatch.setattr(main, "resolve_input_variables", lambda _config: variables)
    monkeypatch.setattr(main, "validate_manifest_metadata", lambda *_args: None)
    monkeypatch.setattr(main, "read_input_band_counts", lambda *_args: {"ppt": 1423})
    monkeypatch.setattr(
        main.metadata, "validate_else_add_temporal_end", lambda *_args: False
    )
    monkeypatch.setattr(
        main.fs_utils,
        "makedirs",
        lambda _path: pytest.fail("preflight-only run created output"),
    )

    main.run_pipeline(config)
