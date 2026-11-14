"""Authoring and document models (AT-001, AT-003, AT-022; META-001 to META-005, META-010, META-013, VAL-003)."""

import json

import pytest
import yaml

from skope_release.documents import load_document, load_yaml, validate
from skope_release.findings import Report
from skope_release.models import Curated, Ledger, Overview, SourceManifest

from release_fixtures import curated_doc, manifest_doc


def check(model, data):
    report = Report()
    return validate(model, data, requirement="TEST", path="doc", report=report), report


def test_curated_fixture_is_valid():
    doc, report = check(Curated, curated_doc("synth", ["alpha"]))
    assert doc is not None, report.to_text()


@pytest.mark.parametrize("field", ["title", "description", "license", "providers", "region_name", "variables"])
def test_required_dataset_fields(field):
    data = curated_doc("synth", ["alpha"])
    del data[field]
    doc, report = check(Curated, data)
    assert doc is None and report.errors


@pytest.mark.parametrize("field", ["title", "description", "unit"])
def test_required_variable_fields(field):
    data = curated_doc("synth", ["alpha"])
    del data["variables"][0][field]
    assert check(Curated, data)[0] is None


@pytest.mark.parametrize("field", ["calendar", "precision", "origin", "end", "endpoint_inclusion", "timestep_meaning", "description"])
def test_required_temporal_semantics(field):
    data = curated_doc("synth", ["alpha"])
    del data["temporal"][field]
    assert check(Curated, data)[0] is None


@pytest.mark.parametrize(
    "extra",
    [
        {"crs": "EPSG:4269"},  # observed (META-003)
        {"transform": [1, 0, 0, 0, -1, 0]},
        {"ordering": 1},  # presentation (META-002)
        {"colormap": "viridis"},
        {"status": "Published"},  # dropped (Section 20.3)
    ],
)
def test_observed_and_presentation_fields_are_rejected(extra):
    data = curated_doc("synth", ["alpha"]) | extra
    assert check(Curated, data)[0] is None


@pytest.mark.parametrize("extra", [{"min": 0}, {"max": 10}, {"nodata": 0}, {"colormap": "viridis"}])
def test_variable_observed_and_presentation_fields_are_rejected(extra):
    data = curated_doc("synth", ["alpha"])
    data["variables"][0].update(extra)
    assert check(Curated, data)[0] is None


def test_unquoted_timestep_is_rejected(tmp_path):
    """META-005: `0103` unquoted is a YAML integer, not a timestep string."""
    data = curated_doc("synth", ["alpha"])
    text = yaml.safe_dump(data, sort_keys=False).replace("origin: '0100'", "origin: 0100")
    path = tmp_path / "curated.yml"
    path.write_text(text)
    report = Report()
    assert load_document(Curated, path, requirement="META-005", report=report) is None
    assert any("origin" in f.message for f in report.errors)


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    path = tmp_path / "x.yml"
    path.write_text("id: a\nid: b\n")
    with pytest.raises(yaml.YAMLError):
        load_yaml(path)


def test_duplicate_variable_ids_are_rejected():
    data = curated_doc("synth", ["alpha", "alpha"])
    assert check(Curated, data)[0] is None


def test_identifiers_follow_org_007():
    assert check(Curated, curated_doc("Synth", ["alpha"]))[0] is None
    assert check(Curated, curated_doc("synth", ["Alpha"]))[0] is None


def test_profile_rules():
    static = curated_doc("srtm", ["elevation"], profile="StaticRasterDataset")
    assert check(Curated, static)[0] is not None
    static["temporal"] = curated_doc("x", ["y"])["temporal"]
    assert check(Curated, static)[0] is None  # API-006: no temporal axis


def test_external_links_must_be_https():
    data = curated_doc("synth", ["alpha"], links=[{"rel": "via", "href": "http://example.org"}])
    assert check(Curated, data)[0] is None


def test_orcid_and_ror_are_validated():
    good = curated_doc("synth", ["alpha"], people=[{"name": "Josiah Carberry", "orcid": "https://orcid.org/0000-0002-1825-0097"}])
    assert check(Curated, good)[0] is not None
    bad_checksum = curated_doc("synth", ["alpha"], people=[{"name": "X", "orcid": "https://orcid.org/0000-0002-1825-0098"}])
    assert check(Curated, bad_checksum)[0] is None
    conflicting = curated_doc(
        "synth",
        ["alpha"],
        people=[
            {"name": "Josiah Carberry", "orcid": "https://orcid.org/0000-0002-1825-0097"},
            {"name": "Someone Else", "orcid": "https://orcid.org/0000-0002-1825-0097"},
        ],
    )
    assert check(Curated, conflicting)[0] is None
    ror = curated_doc("synth", ["alpha"], providers=[{"name": "ASU", "roles": ["producer"], "ror": "https://ror.org/03efmqc40"}])
    assert check(Curated, ror)[0] is not None
    bad_ror = curated_doc("synth", ["alpha"], providers=[{"name": "ASU", "roles": ["producer"], "ror": "https://ror.org/xyz"}])
    assert check(Curated, bad_ror)[0] is None


def test_categories_are_curated_meanings():
    data = curated_doc("synth", ["niche"])
    data["variables"][0]["categories"] = {0: "Outside", 1: "Inside"}
    doc, report = check(Curated, data)
    assert doc is not None and doc.variables[0].categories == {0: "Outside", 1: "Inside"}


def test_source_manifest_rules():
    good = manifest_doc("synth", {"alpha": "/a.tif"})
    assert check(SourceManifest, good)[0] is not None
    no_release = {k: v for k, v in good.items() if k != "release"}
    assert check(SourceManifest, no_release)[0] is None  # META-004
    bad_created = manifest_doc("synth", {"alpha": "/a.tif"})
    bad_created["release"]["created"] = "2026-10-07"
    assert check(SourceManifest, bad_created)[0] is None
    sparse = manifest_doc("synth", {"alpha": "/a.tif"})
    sparse["release"]["cog"]["sparse_ok"] = True
    assert check(SourceManifest, sparse)[0] is None  # COG-010
    semantics = manifest_doc("synth", {"alpha": "/a.tif"})
    semantics["calendar"] = "proleptic_gregorian"
    assert check(SourceManifest, semantics)[0] is None  # META-004: no temporal semantics


def test_a_source_is_a_uri_or_prepared():
    """META-013: exactly one of `uri` and `prepare`; the script is one file in scripts/."""
    prepared = {"script": "scripts/prepare_synth.py", "inputs": [{"uri": "https://example.org/a.nc"}]}

    def with_source(**source):
        doc = manifest_doc("synth", {"alpha": "/a.tif"})
        doc["variables"][0]["source"] = {"bands": "descriptions", **source}
        return check(SourceManifest, doc)[0]

    assert with_source(prepare=prepared) is not None
    assert with_source() is None
    assert with_source(uri="/a.tif", prepare=prepared) is None
    assert with_source(prepare={**prepared, "inputs": []}) is None
    for script in ("prepare_synth.py", "scripts/sub/prepare_synth.py", "scripts/../prepare_synth.py", "scripts/prepare.sh"):
        assert with_source(prepare={**prepared, "script": script}) is None, script


def test_ledger_shape():
    assert check(Ledger, {"releases": [{"release_id": "x-r-2026.10.07", "declaration_digest": "0" * 64}]})[0] is not None
    assert check(Ledger, {"releases": [{"release_id": "x-r-2026.10.07", "declaration_digest": "abc"}]})[0] is None


def test_overview_requires_profile_consistency():
    base = {
        "schema_version": "1.0.0",
        "release_id": "srtm-r-2026.10.07",
        "declaration_digest": "0" * 64,
        "dataset": {
            "id": "srtm", "version": "4.1", "title": "t", "description": "d", "profile": "StaticRasterDataset",
            "license": "CC-BY-4.0", "region_name": "r", "extent": {"bbox": [0, 0, 1, 1]},
            "grid": {"code": "EPSG:4326", "shape": [1, 1], "transform": [1, 0, 0, 0, -1, 1]}, "providers": [],
        },
        "variables": {"srtm_elevation": {"title": "t", "description": "d", "unit": "m", "data_type": "int16",
                                          "nodata": -32768, "scale": 1, "offset": 0,
                                          "asset_href": "cogs/srtm_elevation.tif", "band_name": "srtm_elevation"}},
    }
    assert check(Overview, base)[0] is not None
    with_time = json.loads(json.dumps(base))
    with_time["dataset"]["time"] = {
        "kind": "regular", "origin": "2000", "step": "P1Y", "count": 1, "chunk_size": 1, "precision": "year",
        "calendar": "proleptic_gregorian", "timestep_meaning": "instant", "endpoint_inclusion": "inclusive",
    }
    assert check(Overview, with_time)[0] is None  # API-006
