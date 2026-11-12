"""Prepared sources from fetch through the release manifest (AT-037; META-013, OBS-016, MAN-002)."""

import json
from urllib.parse import urlparse

import pytest

from skope_release import gdalio, prepare
from skope_release.build import build_and_publish
from skope_release.fetch import fetch_dataset
from skope_release.findings import Report
from skope_release.preflight import DatasetFiles, preflight_dataset

from release_fixtures import curated_doc, manifest_doc, write_dataset, write_source, years
from stage_helpers import PRODUCER

RELEASE_ID = "synth-r-2026.10.07"
TOOLS = {"gdal_image": "gdal@sha256:1", "python_lock_sha256": "a" * 64}
UPSTREAM = "https://example.org/synth/alpha.tif"
# Stands in for a real conversion: the input already is the GeoTIFF the build reads.
SCRIPT = """import shutil, sys
_, variable, output, source = sys.argv
shutil.copyfile(source, output)
"""


@pytest.fixture
def setup(tmp_path):
    """A one-variable dataset whose source is prepared from an upstream file."""
    upstream = tmp_path / "remote" / "example.org" / "synth" / "alpha.tif"
    upstream.parent.mkdir(parents=True)
    write_source(upstream, years(100, 5))
    root = tmp_path / "ingest"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "prepare_synth.py").write_text(SCRIPT)
    manifest = manifest_doc("synth", {"alpha": "unused"})
    manifest["variables"][0]["source"] = {
        "bands": "descriptions",
        "prepare": {"script": "scripts/prepare_synth.py", "inputs": [{"uri": UPSTREAM}]},
    }
    files = DatasetFiles(write_dataset(tmp_path, "synth", curated_doc("synth", ["alpha"]), manifest))
    return files, tmp_path / "mirror", root, upstream


def remote(tmp_path):
    return lambda uri: str(tmp_path / "remote" / urlparse(uri).netloc / urlparse(uri).path.lstrip("/"))


def fetch(files, mirror, root, tmp_path, tools=TOOLS):
    report = Report()
    outcomes = dict(fetch_dataset(files, mirror=mirror, report=report, remote=remote(tmp_path), scripts_root=root, toolchain=tools))
    assert report.ok, report.to_text()
    return outcomes


def preflight(files, mirror, root, tools=TOOLS):
    report = Report()
    plan = preflight_dataset(files, release_id=RELEASE_ID, mirror=mirror, producer=PRODUCER, report=report,
                             scripts_root=root, toolchain=tools)
    return plan, report


def requirements(report):
    return [f.requirement for f in report.errors]


def test_preflight_reads_the_prepared_source_and_resolves_its_input_checksums(setup, tmp_path):
    files, mirror, root, upstream = setup
    fetch(files, mirror, root, tmp_path)
    plan, report = preflight(files, mirror, root)
    assert plan is not None, report.to_text()
    variable = plan.variables[0]
    assert variable.source_uri is None
    assert variable.source_read_path == str(prepare.output_path(mirror, "synth", "alpha"))
    source = plan.resolved_manifest.variables[0].source
    assert source.prepare.inputs[0].checksum == gdalio.sha256_multihash(upstream)
    assert source.checksum == gdalio.sha256_multihash(upstream)  # the stand-in script copies it
    assert plan.prepare_scripts == {"alpha": prepare.sha256_hex(root / "scripts" / "prepare_synth.py")}


def test_a_prepared_source_is_read_only_from_the_mirror(setup, tmp_path):
    files, mirror, root, _ = setup
    _, report = preflight(files, None, root)
    assert requirements(report) == ["OBS-016"]
    _, report = preflight(files, mirror, root)  # not fetched on this host
    assert requirements(report) == ["OBS-016"] and "run fetch" in report.errors[0].message


@pytest.mark.parametrize("change", ["script", "toolchain"])
def test_a_stale_record_is_refused(setup, tmp_path, change):
    files, mirror, root, _ = setup
    fetch(files, mirror, root, tmp_path)
    tools = TOOLS
    if change == "script":
        (root / "scripts" / "prepare_synth.py").write_text("# reworded\n" + SCRIPT)
    else:
        tools = {**TOOLS, "gdal_image": "gdal@sha256:2"}
    _, report = preflight(files, mirror, root, tools)
    assert requirements(report) == ["OBS-016"] and f"the {change} changed" in report.errors[0].message


def test_an_output_changed_after_fetch_is_refused(setup, tmp_path):
    files, mirror, root, _ = setup
    fetch(files, mirror, root, tmp_path)
    output = prepare.output_path(mirror, "synth", "alpha")
    write_source(output, years(100, 5), seed=7)
    _, report = preflight(files, mirror, root)
    assert requirements(report) == ["OBS-016"]


def test_the_release_manifest_records_the_inputs_and_the_script(setup, tmp_path):
    files, mirror, root, upstream = setup
    fetch(files, mirror, root, tmp_path)
    plan, report = preflight(files, mirror, root)
    assert plan is not None, report.to_text()
    (tmp_path / "releases").mkdir()
    published, _ = build_and_publish(plan, tmp_path / "releases")
    manifest = json.loads((published.path / "release-manifest.json").read_text())
    assert manifest["sources"] == [{
        "id": "alpha",
        "checksum": gdalio.sha256_multihash(upstream),
        "prepare": {
            "script": "scripts/prepare_synth.py",
            "script_sha256": prepare.sha256_hex(root / "scripts" / "prepare_synth.py"),
            "inputs": [{"href": UPSTREAM, "checksum": gdalio.sha256_multihash(upstream)}],
        },
    }]


def test_a_script_edit_that_keeps_the_bytes_keeps_the_digest(setup, tmp_path):
    """MAN-010: the script's hash is recorded, not declared."""
    files, mirror, root, _ = setup
    fetch(files, mirror, root, tmp_path)
    before, _ = preflight(files, mirror, root)
    (root / "scripts" / "prepare_synth.py").write_text("# reworded\n" + SCRIPT)
    assert fetch(files, mirror, root, tmp_path) == {"alpha": "prepared"}
    after, report = preflight(files, mirror, root)
    assert after is not None, report.to_text()
    assert after.declaration_digest == before.declaration_digest
    assert after.prepare_scripts != before.prepare_scripts
