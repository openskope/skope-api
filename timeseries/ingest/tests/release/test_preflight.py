"""Source preflight (AT-007, AT-009, AT-022; OBS-001 to OBS-011, COG-007, COG-013 to COG-016, META-004)."""

import numpy as np
import pytest
import yaml
from osgeo import gdal

from skope_release.findings import Report
from skope_release.plan import Producer
from skope_release.preflight import DatasetFiles, preflight_dataset

from release_fixtures import write_source, years

PRODUCER = Producer("p", "0", "abc")


def run(directory, release_id=None):
    report = Report()
    plan = preflight_dataset(DatasetFiles(directory), release_id=release_id, mirror=None, producer=PRODUCER, report=report)
    return plan, report


def requirements(report):
    return {f.requirement for f in report.errors}


def edit(directory, name, change):
    path = directory / name
    data = yaml.safe_load(path.read_text())
    change(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def test_good_dataset_passes(temporal_dataset):
    plan, report = run(temporal_dataset(), "synth-r-2026.10.07")
    assert plan is not None, report.to_text()
    assert plan.layout.chunk_count == 3
    assert all(v.source_checksum.startswith("1220") for v in plan.variables)


def test_variable_sets_must_match(temporal_dataset):
    directory = temporal_dataset()
    edit(directory, "curated.yml", lambda d: d["variables"].pop())
    assert "OBS-001" in requirements(run(directory)[1])


def test_band_descriptions_must_match_the_axis(temporal_dataset, tmp_path):
    directory = temporal_dataset()
    write_source(tmp_path / "sources" / "alpha.tif", ["0100", "0101", "0103", "0102", "0104"])
    assert "OBS-006" in requirements(run(directory)[1])


def test_band_count_must_match_the_axis(temporal_dataset):
    directory = temporal_dataset()
    edit(directory, "curated.yml", lambda d: d["temporal"].update(end="0105"))
    assert "OBS-005" in requirements(run(directory)[1])


def test_axis_must_reach_its_end(temporal_dataset):
    directory = temporal_dataset()
    edit(directory, "curated.yml", lambda d: d["temporal"].update(step="P3Y"))
    assert "OBS-004" in requirements(run(directory)[1])


def test_declared_checksum_must_match(temporal_dataset):
    directory = temporal_dataset()
    edit(directory, "source-manifest.yml", lambda d: d["variables"][0]["source"].update(checksum="1220" + "0" * 64))
    assert "OBS-011" in requirements(run(directory)[1])


def test_encoding_must_preserve_the_source(temporal_dataset):
    directory = temporal_dataset()
    edit(directory, "source-manifest.yml", lambda d: d["variables"][0]["encoding"].update(data_type="uint16", nodata=65535))
    assert "COG-016" in requirements(run(directory)[1])
    directory = temporal_dataset(dataset_id="other")
    edit(directory, "source-manifest.yml", lambda d: d["variables"][0]["encoding"].update(nodata=0))
    assert "COG-016" in requirements(run(directory)[1])


def test_approved_narrowing_is_scanned(temporal_dataset, tmp_path):
    """COG-014: a narrowing passes only when every valid value fits."""
    narrow = {"data_type": "uint16", "nodata": 65535, "narrowing": {"approved_by": "review", "evidence": "EXP-004"}}
    directory = temporal_dataset()
    edit(directory, "source-manifest.yml", lambda d: [v["encoding"].update(narrow) for v in d["variables"]])
    plan, report = run(directory)
    assert plan is not None, report.to_text()

    big = tmp_path / "sources" / "alpha.tif"
    ds = gdal.Open(str(big), gdal.GA_Update)
    data = ds.GetRasterBand(2).ReadAsArray()
    data[10, 10] = 70000
    ds.GetRasterBand(2).WriteArray(data)
    ds = None
    edit(directory, "source-manifest.yml", lambda d: None)
    assert "COG-014" in requirements(run(directory)[1])


@pytest.mark.parametrize(
    "source_type,value,representable",
    [
        (gdal.GDT_Float64, 1.00000001, False),
        (gdal.GDT_Float64, 1e40, False),
        (gdal.GDT_Float64, 1e-50, False),
        (gdal.GDT_UInt32, 16777217, False),
        (gdal.GDT_Float64, 1.5, True),
        (gdal.GDT_UInt32, 16777216, True),
    ],
)
def test_float_narrowing_requires_exact_values(temporal_dataset, source_type, value, representable):
    directory = temporal_dataset(
        variables=("alpha",),
        source_kwargs={"alpha": {"dtype": source_type}},
    )
    edit(directory, "source-manifest.yml", lambda d: d["variables"][0]["encoding"].update(
        data_type="float32", nodata=-9999, predictor=3,
        narrowing={"approved_by": "review", "evidence": "representability scan"},
    ))
    source = yaml.safe_load((directory / "source-manifest.yml").read_text())["variables"][0]["source"]["uri"]
    with gdal.Open(source, gdal.GA_Update) as ds:
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
        data[10, 10] = value
        band.WriteArray(data)

    plan, report = run(directory)

    if representable:
        assert plan is not None, report.to_text()
    else:
        assert plan is None
        assert messages(report, "COG-014") == ["band 1 has values not exactly representable as float32"]


def test_grids_must_align(temporal_dataset, tmp_path):
    """OBS-003 with the Section 20.3 tolerance."""
    keys = years(100, 5)
    directory = temporal_dataset()
    write_source(tmp_path / "sources" / "beta.tif", keys, geotransform=(-115.0 + 0.01 * 0.0005, 0.01, 0.0, 43.0, 0.0, -0.01))
    plan, report = run(directory)
    assert plan is not None, report.to_text()  # within 0.001 pixel
    write_source(tmp_path / "sources" / "beta.tif", keys, geotransform=(-115.0 + 0.01 * 0.5, 0.01, 0.0, 43.0, 0.0, -0.01))
    assert "OBS-003" in requirements(run(directory)[1])
    write_source(tmp_path / "sources" / "beta.tif", keys, epsg=4326)
    assert "OBS-003" in requirements(run(directory)[1])


def test_canonical_crs_is_the_one_the_cogs_carry(temporal_dataset, tmp_path):
    """COG-006: a GeoTIFF records OGC:CRS84 as EPSG:4326, so the grid says EPSG:4326."""
    directory = temporal_dataset()
    for vid in ("alpha", "beta"):
        source = tmp_path / "sources" / f"{vid}.tif"
        gdal.Translate(str(source.with_suffix(".vrt")), str(source), format="VRT", outputSRS="OGC:CRS84")

    def use_vrt(data):
        for variable in data["variables"]:
            variable["source"]["uri"] = variable["source"]["uri"].replace(".tif", ".vrt")

    edit(directory, "source-manifest.yml", use_vrt)
    plan, report = run(directory)
    assert plan is not None, report.to_text()
    assert plan.grid.crs_code == "EPSG:4326"


def test_categorical_variables_need_category_preserving_overviews(temporal_dataset):
    categories = {0: "Outside", 1: "Inside"}
    directory = temporal_dataset(
        variables=("niche",),
        source_kwargs={"niche": {"dtype": gdal.GDT_Byte, "nodata": 255, "categories": categories}},
        manifest={"data_type": "uint8", "nodata": 255},
    )
    edit(directory, "curated.yml", lambda d: d["variables"][0].update(categories=categories))
    assert "COG-007" in requirements(run(directory)[1])
    edit(directory, "source-manifest.yml", lambda d: d["variables"][0]["encoding"].update(overview_resampling="MODE"))
    plan, report = run(directory)
    assert plan is not None, report.to_text()


def test_nodata_must_be_representable(temporal_dataset):
    directory = temporal_dataset()
    edit(directory, "source-manifest.yml", lambda d: d["variables"][0]["encoding"].update(nodata=-1))
    assert "OBS-008" in requirements(run(directory)[1])


def test_unreadable_source_is_reported(temporal_dataset):
    directory = temporal_dataset()
    edit(directory, "source-manifest.yml", lambda d: d["variables"][0]["source"].update(uri="/missing.tif"))
    assert "OBS-002" in requirements(run(directory)[1])


def test_static_rules(static_dataset):
    directory = static_dataset()
    edit(directory, "source-manifest.yml", lambda d: d["release"].update(chunk_size=10))
    assert "META-004" in requirements(run(directory)[1])


# OBS-007: empty bands must be exactly the declared empty timesteps.

GAP = {"first": "0101", "last": "0102", "reason": "No reconstruction."}
EMPTY_SOURCES = {"alpha": {"empty": ["0101", "0102"]}, "beta": {"empty": ["0101", "0102"]}}


def messages(report, requirement):
    return [f.message for f in report.errors if f.requirement == requirement]


def test_undeclared_empty_bands_fail(temporal_dataset):
    plan, report = run(temporal_dataset(source_kwargs=EMPTY_SOURCES))
    assert plan is None
    assert messages(report, "OBS-007")[0].startswith("timesteps 0101..0102 have no valid pixels")


def test_declared_empty_timesteps_pass(temporal_dataset):
    plan, report = run(temporal_dataset(source_kwargs=EMPTY_SOURCES, curated={"empty_timesteps": [GAP]}))
    assert plan is not None, report.to_text()
    assert plan.empty_timesteps == {"0101", "0102"}


def test_a_single_empty_timestep_has_first_equal_to_last(temporal_dataset):
    sources = {"alpha": {"empty": ["0103"]}, "beta": {"empty": ["0103"]}}
    gap = {"first": "0103", "last": "0103", "reason": "Lost."}
    plan, report = run(temporal_dataset(source_kwargs=sources, curated={"empty_timesteps": [gap]}))
    assert plan is not None, report.to_text()
    assert plan.empty_timesteps == {"0103"}


def test_declared_empty_timesteps_must_be_empty_in_every_variable(temporal_dataset):
    sources = {"alpha": {"empty": ["0101", "0102"]}, "beta": {"empty": ["0101"]}}
    plan, report = run(temporal_dataset(source_kwargs=sources, curated={"empty_timesteps": [GAP]}))
    assert plan is None
    found = [f for f in report.errors if f.requirement == "OBS-007"]
    assert [(f.variable, f.message) for f in found] == [("beta", "declared empty timesteps 0102 have valid pixels")]


@pytest.mark.parametrize(
    "runs",
    [
        [{"first": "0101", "last": "0109", "reason": "Off the axis."}],
        [{"first": "0102", "last": "0101", "reason": "Backwards."}],
        [{"first": "0102", "last": "0102", "reason": "Out of order."}, {"first": "0101", "last": "0101", "reason": "x"}],
        [{"first": "0101", "last": "0102", "reason": "Overlap."}, {"first": "0102", "last": "0103", "reason": "x"}],
    ],
)
def test_empty_timesteps_are_ordered_runs_on_the_axis(temporal_dataset, runs):
    plan, report = run(temporal_dataset(curated={"empty_timesteps": runs}))
    assert plan is None and "OBS-007" in requirements(report)


def test_a_static_band_is_never_empty(static_dataset, tmp_path):
    directory = static_dataset()
    write_source(tmp_path / "sources" / "elevation.tif", ["band1"], empty=["band1"])
    plan, report = run(directory)
    assert plan is None and messages(report, "OBS-007") == ["band 1 has no valid pixels"]


def test_a_static_dataset_has_no_empty_timesteps(static_dataset):
    directory = static_dataset()
    edit(directory, "curated.yml", lambda d: d.update(empty_timesteps=[GAP]))
    assert "META-001" in requirements(run(directory)[1])
