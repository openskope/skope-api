"""The release overview (AT-016, AT-017; API-001 to API-010, REL-005)."""

import json

import pytest
import yaml

from skope_release.findings import Report
from skope_release.models import Overview
from skope_release.overview import build_overview, emit, render, verify_rule, write_overview
from skope_release.stac_serializer import write_stac
from skope_release.timeaxis import Axis

from stage_helpers import observed

DIGEST = "d" * 64


@pytest.fixture
def staging(temporal_dataset, tmp_path):
    staging = tmp_path / "staging"
    _, obs = observed(temporal_dataset(), staging)
    write_stac(obs, staging)
    return staging


def test_overview_from_stac(staging):
    report = Report()
    path = write_overview(staging, release_id="synth-r-2026.10.07", declaration_digest=DIGEST, report=report)
    assert path is not None, report.to_text()
    text = path.read_text()
    assert text.startswith("# Generated from validated STAC. Do not edit.\n# Authority: collection.json and items/.")
    overview = Overview.model_validate_json(json.dumps(yaml.safe_load(text)))
    assert overview.release_id == "synth-r-2026.10.07" and overview.declaration_digest == DIGEST
    time = overview.dataset.time
    assert (time.kind, time.origin, time.step, time.count, time.chunk_size) == ("regular", "0100", "P1Y", 5, 2)
    assert overview.dataset.grid.transform == (0.01, 0.0, -115.0, 0.0, -0.01, 43.0)
    assert overview.dataset.lineage == "Made up for tests."
    assert overview.variables["alpha"].category == "precipitation"
    assert "colormap" not in text and "range" not in text  # no presentation (API-007)


def test_overview_reads_only_stac(staging, tmp_path):
    """API-001: the curated file and display files are not inputs."""
    (tmp_path / "datasets").rename(tmp_path / "moved-away")
    report = Report()
    assert build_overview(staging, release_id="synth-r-2026.10.07", declaration_digest=DIGEST, report=report) is not None


def test_overview_is_deterministic(staging):
    first = build_overview(staging, release_id="synth-r-2026.10.07", declaration_digest=DIGEST, report=Report())
    second = build_overview(staging, release_id="synth-r-2026.10.07", declaration_digest=DIGEST, report=Report())
    assert render(first, True) == render(second, True)


def test_rule_disagreeing_with_stac_fails(staging):
    """AT-016: the rule must reproduce Band names and band counts."""
    item = staging / "items" / "synth--0102--0103.json"
    data = json.loads(item.read_text())
    data["assets"]["alpha"]["bands"][1]["name"] = "0104"
    item.write_text(json.dumps(data))
    report = Report()
    write_overview(staging, release_id="synth-r-2026.10.07", declaration_digest=DIGEST, report=report)
    assert "API-004" in {f.requirement for f in report.errors}


def test_rule_checks_cog_band_counts(staging):
    items = [json.loads(p.read_text()) for p in sorted((staging / "items").glob("*.json"))]
    report = Report()
    verify_rule(staging, "synth", Axis.regular("0100", "P1Y", "0104", "year"), 2, items, ["alpha", "beta"], report)
    assert report.ok, report.to_text()
    verify_rule(staging, "synth", Axis.regular("0100", "P1Y", "0104", "year"), 3, items, ["alpha"], report)
    assert not report.ok


def test_enumerated_axis_is_inlined_in_flow_style():
    text = emit({"time": {"kind": "enumerated", "values": [f"{y:04d}" for y in range(100, 160)]}})
    lines = text.splitlines()
    assert lines[2].startswith('  values: ["0100", "0101"') and all(len(l) <= 100 for l in lines)
    assert yaml.safe_load(text)["time"]["values"][-1] == "0159"


def test_emitter_quotes_strings_and_keeps_numbers():
    text = emit({"a": "0103", "b": 1.0, "c": 4294967295, "d": None, "e": {0: "Outside"}})
    assert yaml.safe_load(text) == {"a": "0103", "b": 1.0, "c": 4294967295, "d": None, "e": {0: "Outside"}}
    assert 'a: "0103"' in text
