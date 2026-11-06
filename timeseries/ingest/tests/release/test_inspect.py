"""Inspecting written COGs (AT-010, AT-011; COG-001 to COG-009, OBS-007)."""

import dataclasses
import json

import numpy as np
import pytest
from osgeo import gdal

from skope_release import gdalio, inspect_bytes
from skope_release.cog_writer import _to_cog, cog_creation_options, write_cogs
from skope_release.crosscheck import check_stac_against_bytes
from skope_release.findings import Report
from skope_release.inspect_bytes import inspect_asset, observe
from skope_release.plan import Producer
from skope_release.preflight import DatasetFiles, preflight_dataset
from skope_release.stac_serializer import write_stac

from release_fixtures import CREATED
from stage_helpers import plan_for


@pytest.fixture
def staged(temporal_dataset, tmp_path):
    report = Report()
    plan = preflight_dataset(DatasetFiles(temporal_dataset()), release_id="synth-r-2026.10.07", mirror=None, producer=Producer("p", "0", "abc"), report=report)
    staging = tmp_path / "staging"
    write_cogs(plan, staging, tmp_path / "scratch")
    return plan, staging


def test_observation_of_valid_cogs(staged):
    plan, staging = staged
    report = Report()
    obs = observe(plan, staging, report)
    assert obs is not None, report.to_text()
    assert [a.path for a in obs.assets][:3] == [
        "cogs/alpha/synth--0100--0101.tif", "cogs/alpha/synth--0102--0103.tif", "cogs/alpha/synth--0104--0104.tif",
    ]
    asset = obs.asset("alpha", 1)
    assert [b.name for b in asset.bands] == ["0102", "0103"]
    assert asset.checksum.startswith("1220") and asset.crs_code == "EPSG:4269"
    assert obs.bbox_wgs84 == (-115.0, 42.7, -114.6, 43.0)


def test_statistics_match_the_pixels(staged, tmp_path):
    """AT-011: embedded statistics equal statistics recomputed from the source."""
    plan, staging = staged
    obs = observe(plan, staging, Report())
    source = gdal.Open(str(tmp_path / "sources" / "alpha.tif"))
    for i, band in enumerate(obs.asset("alpha", 0).bands, start=1):
        values = source.GetRasterBand(i).ReadAsArray().astype(np.float64)
        values = values[values != 4294967295]
        assert (band.minimum, band.maximum) == (values.min(), values.max())
        assert band.mean == pytest.approx(values.mean(), rel=1e-10)
        assert band.stddev == pytest.approx(values.std(), rel=1e-10)
        assert band.valid_percent == pytest.approx(100 * values.size / (30 * 40), abs=0.005)


def _staged_with(temporal_dataset, tmp_path, name, interleave):
    """A multi-block staged release in the given interleave."""
    report = Report()
    size = {"width": 300, "height": 200}
    release = {"created": CREATED, "chunk_size": 2,
               "cog": {"blocksize": 128, "interleave": interleave, "compress": "ZSTD", "level": 9, "bigtiff": "IF_SAFER"}}
    directory = temporal_dataset(source_kwargs={"alpha": size, "beta": size}, manifest={"release": release})
    plan = preflight_dataset(DatasetFiles(directory), release_id="synth-r-2026.10.07", mirror=None, producer=Producer("p", "0", "abc"), report=report)
    assert plan is not None, report.to_text()
    staging = tmp_path / name
    write_cogs(plan, staging, tmp_path / f"scratch-{name}")
    return plan, staging


@pytest.mark.parametrize("interleave", ["BAND", "TILE"])
def test_inspects_band_and_tile_interleaved_cogs(temporal_dataset, tmp_path, interleave):
    """The in-memory inspection reads either layout, and frees its copies."""
    plan, staging = _staged_with(temporal_dataset, tmp_path, interleave.lower(), interleave)
    rel = plan.layout.cog_path("alpha", 0)
    report = Report()
    asset = inspect_asset(staging / rel, rel, plan=plan, variable=plan.variables[0], chunk=0,
                          expected_names=list(plan.layout.chunk_keys(0)), report=report)
    assert asset is not None, report.to_text()
    assert asset.interleave == interleave and asset.blocksize == 128
    assert asset.checksum == gdalio.sha256_multihash(staging / rel)
    assert not [p for p in gdal.ReadDir("/vsimem/") or [] if p.startswith("inspect-")]


def test_in_memory_and_in_place_inspection_agree(staged, monkeypatch):
    plan, staging = staged
    rel = "cogs/alpha/synth--0100--0101.tif"

    def inspect():
        report = Report()
        asset = inspect_asset(staging / rel, rel, plan=plan, variable=plan.variables[0], chunk=0,
                              expected_names=["0100", "0101"], report=report)
        assert asset is not None, report.to_text()
        return asset

    in_memory = inspect()
    monkeypatch.setattr(inspect_bytes, "IN_MEMORY_LIMIT", 0)
    assert inspect() == in_memory


def reinspect(plan, staging, rel):
    report = Report()
    inspect_asset(staging / rel, rel, plan=plan, variable=plan.variables[0], chunk=0,
                  expected_names=["0100", "0101"], report=report)
    return {f.requirement for f in report.errors}


def test_missing_statistics_are_detected(staged):
    plan, staging = staged
    rel = "cogs/alpha/synth--0100--0101.tif"
    path = staging / rel
    gdal.Translate(str(path) + ".tmp.tif", str(path), format="COG", creationOptions=["STATISTICS=NO", "COMPRESS=ZSTD", "BLOCKSIZE=256", "INTERLEAVE=BAND"])
    copy = gdal.Open(str(path) + ".tmp.tif")
    has_stats = "STATISTICS_MEAN" in copy.GetRasterBand(1).GetMetadata()
    copy = None
    if has_stats:
        pytest.skip("GDAL copied the source statistics")
    (staging / (rel + ".tmp.tif")).replace(path)
    assert "COG-004" in reinspect(plan, staging, rel)


def test_wrong_band_description_is_detected(staged):
    plan, staging = staged
    rel = "cogs/alpha/synth--0100--0101.tif"
    report = Report()
    inspect_asset(staging / rel, rel, plan=plan, variable=plan.variables[0], chunk=0,
                  expected_names=["0100", "0102"], report=report)
    assert "COG-006" in {f.requirement for f in report.errors}


def test_sidecars_are_rejected(staged):
    plan, staging = staged
    rel = "cogs/alpha/synth--0100--0101.tif"
    (staging / (rel + ".aux.xml")).write_text("<PAMDataset/>")
    assert "COG-005" in reinspect(plan, staging, rel)


def test_non_cog_is_rejected(staged):
    plan, staging = staged
    rel = "cogs/alpha/synth--0100--0101.tif"
    path = staging / rel
    gdal.Translate(str(path) + ".gtiff", str(path), format="GTiff", creationOptions=["TILED=NO", "COMPRESS=NONE"])
    (staging / (rel + ".gtiff")).replace(path)
    assert {"COG-001", "COG-003"} & reinspect(plan, staging, rel)


# OBS-007, COG-004, STAC-008: declared empty timesteps, end to end.


@pytest.fixture
def gap_plan(temporal_dataset):
    sources = {"alpha": {"empty": ["0101"]}, "beta": {"empty": ["0101"]}}
    gap = {"first": "0101", "last": "0101", "reason": "No reconstruction."}
    return plan_for(temporal_dataset(source_kwargs=sources, curated={"empty_timesteps": [gap]}))


def test_declared_empty_bands_round_trip(gap_plan, tmp_path, capfd):
    staging = tmp_path / "staging"
    write_cogs(gap_plan, staging, tmp_path / "scratch")
    assert "no valid pixels" not in capfd.readouterr().err  # GDAL's message is expected and hidden

    report = Report()
    obs = observe(gap_plan, staging, report)
    assert obs is not None, report.to_text()
    full, empty = obs.asset("alpha", 0).bands
    assert set(full.statistics) == {"minimum", "maximum", "mean", "stddev", "valid_percent"}
    assert (empty.name, empty.statistics) == ("0101", {"valid_percent": 0.0})

    write_stac(obs, staging)
    item = json.loads((staging / "items" / "synth--0100--0101.json").read_text())
    assert item["assets"]["alpha"]["bands"][1] == {"name": "0101", "statistics": {"valid_percent": 0.0}}
    check_stac_against_bytes(obs, staging, report)
    assert report.ok, report.to_text()


def test_an_undeclared_empty_band_is_obs_007(gap_plan, tmp_path):
    staging = tmp_path / "staging"
    write_cogs(gap_plan, staging, tmp_path / "scratch")
    undeclared = dataclasses.replace(gap_plan, empty_timesteps=frozenset())
    assert reinspect(undeclared, staging, "cogs/alpha/synth--0100--0101.tif") == {"OBS-007"}


def test_gdal_empty_band_message_is_shown_unless_expected(gap_plan, tmp_path, capfd):
    write_cogs(gap_plan, tmp_path / "staging", tmp_path / "scratch")
    cog = tmp_path / "staging" / "cogs/alpha/synth--0100--0101.tif"
    options = cog_creation_options(gap_plan, gap_plan.variables[0])
    _to_cog(cog, tmp_path / "again.tif", options, 1)
    assert "no valid pixels" in capfd.readouterr().err
    with pytest.raises(RuntimeError):  # real failures still raise
        _to_cog(tmp_path / "missing.tif", tmp_path / "out.tif", options, 1, expect_empty=True)
