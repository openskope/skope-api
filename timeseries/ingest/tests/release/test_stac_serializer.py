"""Serializing STAC from the final observation (Sections 9 and 10; STAC-003 to STAC-012)."""

import json

import pytest

from skope_release.plan import StateError
from skope_release.stac_serializer import build_collection, build_items, rfc3339, write_stac

from stage_helpers import observed


@pytest.fixture
def temporal(temporal_dataset, tmp_path):
    return observed(temporal_dataset(), tmp_path / "staging")


def test_collection(temporal):
    _, obs = temporal
    c = build_collection(obs)
    assert c["type"] == "Collection" and c["stac_version"] == "1.1.0" and c["version"] == "1"
    assert c["skope:region_name"] == "Testland"
    assert c["skope:variables"] == {"alpha": {"category": "precipitation"}, "beta": {"category": "precipitation"}}
    assert c["skope:temporal"]["timestep_meaning"] == "aggregation_period"
    assert c["cube:dimensions"]["time"] == {
        "type": "temporal", "extent": ["0100-01-01T00:00:00Z", "0104-12-31T23:59:59Z"],
        "values": ["0100", "0101", "0102", "0103", "0104"], "step": "P1Y",
    }
    assert set(c["item_assets"]) == set(c["cube:variables"]) == {"alpha", "beta"}
    assert c["extent"]["temporal"]["interval"] == [["0100-01-01T00:00:00Z", "0104-12-31T23:59:59Z"]]
    assert all(link["rel"] != "self" for link in c["links"])
    assert [l["href"] for l in c["links"] if l["rel"] == "item"] == [
        "./items/synth--0100--0101.json", "./items/synth--0102--0103.json", "./items/synth--0104--0104.json",
    ]
    processor = c["providers"][0]  # Processing 1.2.0 fields sit on a provider
    assert processor["processing:lineage"] == "Made up for tests."
    assert "GDAL" in processor["processing:software"]
    assert "https://stac-extensions.github.io/datacube/v2.3.0/schema.json" in c["stac_extensions"]
    assert not any("api.openskope.org" in url for url in c["stac_extensions"])  # SKOPE-005


def test_items(temporal):
    _, obs = temporal
    items = build_items(obs)
    assert [i["id"] for i in items] == ["synth--0100--0101", "synth--0102--0103", "synth--0104--0104"]
    item = items[1]
    assert item["properties"] == {
        "datetime": None, "start_datetime": "0102-01-01T00:00:00Z", "end_datetime": "0103-12-31T23:59:59Z",
    }
    asset = item["assets"]["alpha"]
    assert asset["href"] == "../cogs/alpha/synth--0102--0103.tif"
    assert asset["type"] == "image/tiff; application=geotiff; profile=cloud-optimized" and asset["roles"] == ["data"]
    assert asset["proj:code"] == "EPSG:4269" and "proj:epsg" not in asset
    assert asset["data_type"] == "uint32" and asset["nodata"] == 4294967295 and asset["unit"] == "mm"
    assert [b["name"] for b in asset["bands"]] == ["0102", "0103"]
    assert set(asset["bands"][0]) == {"name", "statistics"}  # STAC-008


def test_static_collection(static_dataset, tmp_path):
    _, obs = observed(static_dataset(), tmp_path / "staging", "elev-r-2026.10.07")
    c = build_collection(obs)
    assert "item_assets" not in c and "cube:dimensions" not in c and "skope:temporal" not in c
    assert c["assets"]["elevation"]["href"] == "./cogs/elevation.tif"
    assert c["assets"]["elevation"]["bands"][0]["name"] == "elevation"
    assert c["extent"]["temporal"]["interval"] == [["2000-02-11T00:00:00Z", "2000-02-22T23:59:59Z"]]


def test_write_stac(temporal, tmp_path):
    _, obs = temporal
    staging = tmp_path / "staging"
    write_stac(obs, staging)
    text = (staging / "collection.json").read_text()
    assert text.endswith("}\n") and json.loads(text)["id"] == "synth"
    assert sorted(p.name for p in (staging / "items").iterdir()) == [
        "synth--0100--0101.json", "synth--0102--0103.json", "synth--0104--0104.json",
    ]


def test_years_before_1000_keep_four_digits():
    from datetime import datetime, timezone

    assert rfc3339(datetime(103, 1, 1, tzinfo=timezone.utc)) == "0103-01-01T00:00:00Z"


def test_the_serializer_takes_only_an_observation(temporal, tmp_path):
    plan, _ = temporal
    with pytest.raises(StateError):
        write_stac(plan, tmp_path / "staging")
