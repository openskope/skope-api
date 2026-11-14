"""The models accept the documents the release build writes and reject the rest."""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.registry.models import Overview, Pin, ReleaseManifest

REAL = Path(__file__).parent / "data" / "paleocar_v3-r-2026.10.07"


def _overview_doc():
    return yaml.safe_load((REAL / "overview.yml").read_text())


def test_real_overview_parses():
    overview = Overview.model_validate(_overview_doc())

    assert overview.release_id == "paleocar_v3-r-2026.10.07"
    assert overview.dataset.time.origin == "0103"
    assert overview.dataset.time.count == 1898
    assert sorted(overview.variables) == ["gdd_cotton_annual", "ppt_annual"]


def test_real_manifest_parses():
    manifest = ReleaseManifest.model_validate(
        json.loads((REAL / "release-manifest.json").read_text())
    )

    assert manifest.dataset.id == "paleocar_v3"
    assert any(f.path == "overview.yml" for f in manifest.files)


PREPARE = {
    "script": "scripts/prepare_lbda_v2.py",
    "script_sha256": "c" * 64,
    "inputs": [{"href": "https://example.org/lbda.nc", "checksum": "1220" + "d" * 64}],
}


def _manifest_with_source(source):
    doc = json.loads((REAL / "release-manifest.json").read_text())
    doc["sources"] = [{"id": "pmdi", "checksum": "1220" + "e" * 64, **source}]
    return doc


def test_prepared_source_parses():
    manifest = ReleaseManifest.model_validate(
        _manifest_with_source({"prepare": PREPARE})
    )

    assert manifest.sources[0].href is None
    assert manifest.sources[0].prepare.inputs[0].href == "https://example.org/lbda.nc"


@pytest.mark.parametrize(
    "source",
    [{}, {"href": "s3://test/pmdi.tif", "prepare": PREPARE}],
    ids=["neither", "both"],
)
def test_source_needs_exactly_one_origin(source):
    with pytest.raises(ValidationError, match="MAN-002"):
        ReleaseManifest.model_validate(_manifest_with_source(source))


def test_unknown_overview_field_is_rejected():
    doc = _overview_doc()
    doc["dataset"]["colormap"] = "viridis"

    with pytest.raises(ValidationError, match="colormap"):
        Overview.model_validate(doc)


def test_temporal_overview_needs_time():
    doc = _overview_doc()
    del doc["dataset"]["time"]

    with pytest.raises(ValidationError, match="time"):
        Overview.model_validate(doc)


def test_unquoted_timestep_is_rejected():
    doc = _overview_doc()
    doc["dataset"]["time"]["origin"] = 103  # what YAML makes of an unquoted 0103

    with pytest.raises(ValidationError):
        Overview.model_validate(doc)


PIN = {
    "release_root": "/srv/datasets/releases",
    "releases": [
        {
            "dataset": "paleocar_v3",
            "release_id": "paleocar_v3-r-2026.10.07",
            "declaration_digest": "a" * 64,
            "manifest_sha256": "b" * 64,
        }
    ],
}


def test_pin_parses():
    pin = Pin.model_validate(PIN)

    assert pin.releases[0].release_id == "paleocar_v3-r-2026.10.07"


@pytest.mark.parametrize(
    "change",
    [
        {"releases": []},
        {"release_root": ""},
        {"releases": [{**PIN["releases"][0], "release_id": "paleocar_v3"}]},
        {"releases": [{**PIN["releases"][0], "manifest_sha256": "B" * 64}]},
        {"name": "dev"},
    ],
)
def test_bad_pin_is_rejected(change):
    with pytest.raises(ValidationError):
        Pin.model_validate({**PIN, **change})
