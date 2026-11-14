"""Offline STAC validation (AT-013, AT-021; VAL-005, REL-002, STAC-002 to STAC-011)."""

import json

import pytest

from skope_release.findings import Report
from skope_release.stac_serializer import write_stac
from skope_release.stac_validate import validate_stac

from stage_helpers import observed


@pytest.fixture
def staging(temporal_dataset, tmp_path):
    staging = tmp_path / "staging"
    _, obs = observed(temporal_dataset(), staging)
    write_stac(obs, staging)
    return staging


def errors(staging):
    report = Report()
    validate_stac(staging, report)
    return {f.requirement for f in report.errors}


def edit(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_generated_stac_is_valid(staging):
    report = Report()
    assert validate_stac(staging, report) is not None
    assert report.ok, report.to_text()


def test_static_stac_is_valid(static_dataset, tmp_path):
    staging = tmp_path / "static"
    _, obs = observed(static_dataset(), staging, "elev-r-2026.10.07")
    write_stac(obs, staging)
    assert errors(staging) == set()


@pytest.mark.parametrize(
    "change,requirement",
    [
        (lambda c: c["links"].append({"rel": "self", "href": "./collection.json"}), "REL-002"),
        (lambda c: c["links"].append({"rel": "via", "href": "http://example.org"}), "REL-002"),
        (lambda c: c["links"].append({"rel": "related", "href": "./missing.json"}), "OBS-010"),
        (lambda c: c["stac_extensions"].append("https://stac-extensions.github.io/projection/v1.1.0/schema.json"), "STAC-002"),
        (lambda c: c.pop("license"), "VAL-005"),
        (lambda c: c.update({"skope:style": "red"}), "SKOPE-001"),
        (lambda c: c["skope:variables"].update({"gamma": {"category": "x"}}), "SKOPE-002"),
        (lambda c: c["cube:dimensions"]["time"]["values"].reverse(), "STAC-011"),
        (lambda c: c["extent"]["spatial"]["bbox"][0].__setitem__(0, -120.0), "STAC-004"),
    ],
)
def test_collection_problems_are_detected(staging, change, requirement):
    edit(staging / "collection.json", change)
    assert requirement in errors(staging)


def test_item_problems_are_detected(staging):
    item = staging / "items" / "synth--0102--0103.json"
    edit(item, lambda i: i["assets"].pop("beta"))
    assert "ORG-005" in errors(staging)


def test_proj_epsg_is_rejected(staging):
    item = staging / "items" / "synth--0102--0103.json"
    edit(item, lambda i: i["assets"]["alpha"].update({"proj:epsg": 4269}))
    assert "STAC-007" in errors(staging)
