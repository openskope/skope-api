import pytest
import yaml
from unittest.mock import AsyncMock

from app.store.index_loaders import load_registry, resolve_colormaps
from app.core.slice_resolver import resolve_temporal_slice, resolve_uri_single_band

# ---------------------------------------------------------------------------
# load_registry


def test_load_registry_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_registry(tmp_path / "nonexistent.yml")


def test_load_registry_valid_yaml(tmp_path):
    data = [
        {
            "id": "ds-one",
            "crs": "EPSG:4326",
            "transform": [0.00833, 0.0, -115.0, 0.0, -0.00833, 43.0],
            "variables": [{"id": "ppt"}],
        },
        {
            "id": "ds-two",
            "crs": "EPSG:32612",
            "transform": [800.0, 0.0, 200000.0, 0.0, -800.0, 4800000.0],
            "variables": [{"id": "temp"}],
        },
    ]
    registry_file = tmp_path / "metadata.yml"
    registry_file.write_text(yaml.dump(data))
    result = load_registry(registry_file)
    assert set(result.keys()) == {"ds-one", "ds-two"}
    assert result["ds-one"]["crs"] == "EPSG:4326"


def test_load_registry_missing_id_raises(tmp_path):
    data = [{"crs": "EPSG:4326", "transform": [1, 2, 3, 4, 5, 6]}]
    registry_file = tmp_path / "metadata.yml"
    registry_file.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="'id'"):
        load_registry(registry_file)


def test_load_registry_missing_crs_raises(tmp_path):
    data = [{"id": "my-ds", "transform": [1, 2, 3, 4, 5, 6]}]
    registry_file = tmp_path / "metadata.yml"
    registry_file.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="my-ds"):
        load_registry(registry_file)


def test_load_registry_invalid_transform_length_raises(tmp_path):
    data = [{"id": "bad-ds", "crs": "EPSG:4326", "transform": [1, 2, 3, 4]}]
    registry_file = tmp_path / "metadata.yml"
    registry_file.write_text(yaml.dump(data))
    with pytest.raises(ValueError, match="bad-ds"):
        load_registry(registry_file)


def test_load_registry_accepts_9_element_transform(tmp_path):
    data = [
        {
            "id": "nine-ds",
            "crs": "EPSG:4326",
            "transform": [0.00833, 0.0, -115.0, 0.0, -0.00833, 43.0, 0.0, 0.0, 1.0],
        }
    ]
    registry_file = tmp_path / "metadata.yml"
    registry_file.write_text(yaml.dump(data))
    result = load_registry(registry_file)
    assert "nine-ds" in result


def test_load_registry_invalid_yaml_raises(tmp_path):
    registry_file = tmp_path / "bad.yml"
    registry_file.write_text("key: [unclosed")
    with pytest.raises(ValueError, match="Failed to parse registry YAML"):
        load_registry(registry_file)


async def test_resolve_colormaps_assigns_and_resolves_default(tmp_path, monkeypatch):
    registry = {
        "dataset": {
            "variables": [
                {"id": "defaulted"},
                {"id": "custom", "colormap": "skope-precip"},
            ]
        }
    }
    colormaps_path = tmp_path / "colormaps.json"
    colormaps_path.write_text("{}")
    fetch = AsyncMock(side_effect=lambda _client, _url, name: [f"#{name}"])
    monkeypatch.setattr("app.store.index_loaders._fetch_colormap_from_titiler", fetch)

    await resolve_colormaps(registry, colormaps_path, object(), "http://titiler")

    defaulted, custom = registry["dataset"]["variables"]
    assert defaulted["colormap"] == "viridis"
    assert defaulted["colormap_stops"] == ["#viridis"]
    assert custom["colormap"] == "skope-precip"
    assert custom["colormap_stops"] == ["#skope-precip"]


# ---------------------------------------------------------------------------
# resolve_temporal_slice


def test_resolve_temporal_slice_normal_range(minimal_lookup_data):
    file_mapping, timestep_list = resolve_temporal_slice(
        minimal_lookup_data, "ppt", "0101", "0103", base_url="s3://bucket"
    )
    assert timestep_list == ["0101", "0102", "0103"]
    assert file_mapping["s3://bucket/file_a.tif"] == [2]
    assert file_mapping["s3://bucket/file_b.tif"] == [1, 2]


def test_resolve_temporal_slice_full_range(minimal_lookup_data):
    file_mapping, timestep_list = resolve_temporal_slice(
        minimal_lookup_data, "ppt", "0100", "0105", base_url="s3://bucket"
    )
    assert len(timestep_list) == 6
    assert "s3://bucket/file_a.tif" in file_mapping
    assert "s3://bucket/file_b.tif" in file_mapping
    assert "s3://bucket/file_c.tif" in file_mapping


def test_resolve_temporal_slice_single_step(minimal_lookup_data):
    file_mapping, timestep_list = resolve_temporal_slice(
        minimal_lookup_data, "ppt", "0102", "0102", base_url="s3://bucket"
    )
    assert timestep_list == ["0102"]
    assert list(file_mapping.values()) == [[1]]


def test_resolve_temporal_slice_bands_sorted():
    # file_b.tif covers 0102 (bidx 1), 0103 (bidx 2), 0104 (bidx 3) — already in order
    # Build a lookup where the same URI gets bands in non-insertion order across steps
    lookup = {
        "ppt": {
            "0100": {"file": "chunk.tif", "bidx": 3},
            "0101": {"file": "chunk.tif", "bidx": 1},
            "0102": {"file": "chunk.tif", "bidx": 2},
        }
    }
    file_mapping, _ = resolve_temporal_slice(
        lookup, "ppt", "0100", "0102", base_url="s3://bucket"
    )
    assert file_mapping["s3://bucket/chunk.tif"] == [1, 2, 3]


def test_resolve_temporal_slice_unknown_variable(minimal_lookup_data):
    with pytest.raises(ValueError, match="nonexistent"):
        resolve_temporal_slice(
            minimal_lookup_data, "nonexistent", "0100", "0105", base_url="s3://bucket"
        )


def test_resolve_temporal_slice_no_data_in_range(minimal_lookup_data):
    with pytest.raises(ValueError):
        resolve_temporal_slice(
            minimal_lookup_data, "ppt", "0200", "0300", base_url="s3://bucket"
        )


# ---------------------------------------------------------------------------
# resolve_uri_single_band


def test_resolve_uri_single_band_found(minimal_lookup_data):
    uri, band = resolve_uri_single_band(
        minimal_lookup_data, "ppt", "0103", base_url="s3://bucket"
    )
    assert uri == "s3://bucket/file_b.tif"
    assert band == 2


def test_resolve_uri_single_band_missing_timestep(minimal_lookup_data):
    with pytest.raises(ValueError):
        resolve_uri_single_band(
            minimal_lookup_data, "ppt", "9999", base_url="s3://bucket"
        )


def test_resolve_uri_single_band_missing_variable(minimal_lookup_data):
    with pytest.raises(ValueError):
        resolve_uri_single_band(
            minimal_lookup_data, "bogus", "0100", base_url="s3://bucket"
        )
