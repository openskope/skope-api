from datetime import datetime, timezone

from cog_stac_pipeline.config import PipelineConfig


def test_pipeline_config_from_env(monkeypatch):
    monkeypatch.setenv("INPUT_DIR", "/data/input")
    monkeypatch.setenv("INPUT_MANIFEST_PATH", "/manifests/input.yml")
    monkeypatch.setenv("OUTPUT_DIR", "/tmp/output")
    monkeypatch.setenv("DATASET_NAME", "test-dataset")
    monkeypatch.setenv("METADATA_FILE_PATH", "/metadata.yml")
    monkeypatch.setenv("TRUNC_TO_UINT16", "false")
    monkeypatch.setenv("PREFLIGHT_ONLY", "true")
    monkeypatch.setenv("MAX_BANDS_PER_SLICE", "25")
    monkeypatch.setenv("DATASET_START_DATETIME", "0001-01-01T00:00:00Z")
    monkeypatch.setenv("DATASET_TIME_DELTA", '{"months": 1}')

    config = PipelineConfig.from_env()

    assert config.input_dir == "/data/input"
    assert config.input_manifest_path == "/manifests/input.yml"
    assert config.output_dir == "/tmp/output"
    assert config.resolved_output_dir == "/tmp/output"
    assert config.dataset_name == "test-dataset"
    assert config.metadata_file_path == "/metadata.yml"
    assert config.trunc_to_uint16 is False
    assert config.preflight_only is True
    assert config.max_bands_per_slice == 25
    assert config.dataset_start_datetime == datetime(1, 1, 1, tzinfo=timezone.utc)
    assert config.dataset_time_delta == {"months": 1}


def test_pipeline_config_defaults_output_to_dataset_under_input_dir():
    config = PipelineConfig(input_dir="/data/input", dataset_name="test-dataset")

    assert config.output_dir is None
    assert config.resolved_output_dir == "/data/input/test-dataset"
