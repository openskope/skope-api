"""Fetching sources into the mirror (OBS-011, OBS-014, OBS-015; AT-037)."""

import hashlib
import json
from urllib.parse import urlparse

import pytest

from skope_release import gdalio, prepare
from skope_release.cli import main
from skope_release.fetch import fetch_dataset
from skope_release.findings import Report
from skope_release.preflight import DatasetFiles

from release_fixtures import curated_doc, manifest_doc, write_dataset


@pytest.fixture
def remote(tmp_path):
    """A stand-in for object storage: s3://<bucket>/<key> lives at remote/<bucket>/<key>."""
    root = tmp_path / "remote"

    def put(uri, data=b"source bytes"):
        path = root / urlparse(uri).netloc / urlparse(uri).path.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def read(uri):
        return str(root / urlparse(uri).netloc / urlparse(uri).path.lstrip("/"))

    put.read = read
    return put


def dataset(tmp_path, sources, checksums=None):
    manifest = manifest_doc("synth", sources)
    for var in manifest["variables"]:
        if checksums and var["id"] in checksums:
            var["source"]["checksum"] = checksums[var["id"]]
    return DatasetFiles(write_dataset(tmp_path, "synth", curated_doc("synth", list(sources)), manifest))


def fetch(files, mirror, remote):
    report = Report()
    outcomes = dict(fetch_dataset(files, mirror=mirror, report=report, remote=remote.read))
    return outcomes, report


def test_fetches_and_verifies_a_declared_checksum(tmp_path, remote):
    uri = "s3://bucket/alpha/source.tif"
    checksum = gdalio.sha256_multihash(remote(uri))
    files = dataset(tmp_path, {"alpha": uri}, {"alpha": checksum})
    mirror = tmp_path / "mirror"
    outcomes, report = fetch(files, mirror, remote)
    assert outcomes == {"alpha": "fetched"} and not report.findings
    copy = mirror / "bucket" / "alpha" / "source.tif"
    assert copy.read_bytes() == b"source bytes"
    assert gdalio.read_path(uri, mirror) == str(copy)


def test_reports_the_checksum_of_an_undeclared_source(tmp_path, remote):
    uri = "s3://bucket/alpha/source.tif"
    checksum = gdalio.sha256_multihash(remote(uri))
    outcomes, report = fetch(dataset(tmp_path, {"alpha": uri}), tmp_path / "mirror", remote)
    assert outcomes == {"alpha": "fetched"} and report.ok
    assert [f.requirement for f in report.findings] == ["OBS-011"] and checksum in report.findings[0].message


def test_a_checksum_mismatch_leaves_nothing_in_the_mirror(tmp_path, remote):
    uri = "s3://bucket/alpha/source.tif"
    remote(uri)
    files = dataset(tmp_path, {"alpha": uri}, {"alpha": "1220" + "0" * 64})
    mirror = tmp_path / "mirror"
    outcomes, report = fetch(files, mirror, remote)
    assert outcomes == {"alpha": "mismatch"} and [f.requirement for f in report.errors] == ["OBS-011"]
    assert not any(p.is_file() for p in mirror.rglob("*"))


def test_a_present_copy_is_verified_not_downloaded(tmp_path, remote):
    uri = "s3://bucket/alpha/source.tif"
    checksum = gdalio.sha256_multihash(remote(uri))
    mirror = tmp_path / "mirror"
    copy = mirror / "bucket" / "alpha" / "source.tif"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(b"source bytes")
    remote(uri, b"changed upstream")
    outcomes, report = fetch(dataset(tmp_path, {"alpha": uri}, {"alpha": checksum}), mirror, remote)
    assert outcomes == {"alpha": "present"} and not report.findings
    assert copy.read_bytes() == b"source bytes"


def test_a_present_copy_that_does_not_match_is_an_error(tmp_path, remote):
    uri = "s3://bucket/alpha/source.tif"
    checksum = gdalio.sha256_multihash(remote(uri))
    mirror = tmp_path / "mirror"
    copy = mirror / "bucket" / "alpha" / "source.tif"
    copy.parent.mkdir(parents=True)
    copy.write_bytes(b"corrupted")
    outcomes, report = fetch(dataset(tmp_path, {"alpha": uri}, {"alpha": checksum}), mirror, remote)
    assert outcomes == {"alpha": "mismatch"} and [f.requirement for f in report.errors] == ["OBS-011"]
    assert copy.read_bytes() == b"corrupted"


def test_a_failed_download_leaves_no_copy(tmp_path, remote):
    uri = "s3://bucket/missing/source.tif"
    mirror = tmp_path / "mirror"
    outcomes, report = fetch(dataset(tmp_path, {"alpha": uri}), mirror, remote)
    assert outcomes == {"alpha": "failed"} and [f.requirement for f in report.errors] == ["OBS-014"]
    assert not any(p.is_file() for p in mirror.rglob("*"))


def test_an_interrupted_download_is_replaced(tmp_path, remote):
    uri = "s3://bucket/alpha/source.tif"
    remote(uri)
    mirror = tmp_path / "mirror"
    partial = mirror / "bucket" / "alpha" / "source.tif.part"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"sour")
    assert gdalio.read_path(uri, mirror).startswith("/vsis3/")
    outcomes, _ = fetch(dataset(tmp_path, {"alpha": uri}), mirror, remote)
    assert outcomes == {"alpha": "fetched"}
    assert (mirror / "bucket" / "alpha" / "source.tif").read_bytes() == b"source bytes" and not partial.exists()


def test_a_source_without_a_mirror_location_is_skipped(tmp_path, remote):
    local = tmp_path / "local.tif"
    local.write_bytes(b"local")
    mirror = tmp_path / "mirror"
    outcomes, report = fetch(dataset(tmp_path, {"alpha": str(local)}), mirror, remote)
    assert outcomes == {"alpha": "skipped"} and report.ok
    assert [f.requirement for f in report.findings] == ["OBS-014"]


def test_cli_needs_a_mirror(tmp_path, monkeypatch):
    monkeypatch.delenv("SKOPE_RELEASE_SOURCE_MIRROR", raising=False)
    dataset(tmp_path, {"alpha": "s3://bucket/alpha/source.tif"})
    with pytest.raises(SystemExit, match="OBS-014"):
        main(["--datasets", str(tmp_path / "datasets"), "fetch", "synth"])


# Prepared sources (META-013, OBS-015). The stand-in script writes the variable
# ID followed by its inputs' bytes, so its output is easy to predict.

SCRIPT = """import sys
_, variable, output, *inputs = sys.argv
with open(output, "wb") as out:
    out.write(variable.encode())
    for path in inputs:
        out.write(open(path, "rb").read())
"""
TOOLS = {"gdal_image": "gdal@sha256:1", "python_lock_sha256": "a" * 64}
NC = "https://example.org/data/a.nc"


@pytest.fixture
def scripts(tmp_path):
    root = tmp_path / "ingest"
    (root / "scripts").mkdir(parents=True)

    def write(text=SCRIPT, name="prepare_synth.py"):
        (root / "scripts" / name).write_text(text)
        return root

    write()
    return write


def prepared_dataset(tmp_path, inputs, *, variables=("alpha",), checksum=None, input_checksums=None):
    manifest = manifest_doc("synth", {v: "unused" for v in variables})
    declared = [{"uri": u, **({"checksum": input_checksums[u]} if input_checksums and u in input_checksums else {})} for u in inputs]
    for var in manifest["variables"]:
        var["source"] = {"bands": "descriptions", "prepare": {"script": "scripts/prepare_synth.py", "inputs": declared}}
        if checksum:
            var["source"]["checksum"] = checksum
    return DatasetFiles(write_dataset(tmp_path, "synth", curated_doc("synth", list(variables)), manifest))


def fetch_prepared(files, mirror, remote, root, tools=TOOLS):
    report = Report()
    outcomes = dict(fetch_dataset(files, mirror=mirror, report=report, remote=remote.read, scripts_root=root, toolchain=tools))
    return outcomes, report


def multihash(data):
    return "1220" + hashlib.sha256(data).hexdigest()


def test_prepares_a_source_and_records_how(tmp_path, remote, scripts):
    remote(NC, b"-nc")
    mirror = tmp_path / "mirror"
    outcomes, report = fetch_prepared(prepared_dataset(tmp_path, [NC]), mirror, remote, scripts())
    assert outcomes == {"alpha": "prepared"} and report.ok
    output = mirror / "prepared" / "synth" / "alpha.tif"
    assert output.read_bytes() == b"alpha-nc"
    assert (mirror / "example.org" / "data" / "a.nc").read_bytes() == b"-nc"
    record = json.loads(prepare.record_path(output).read_text())
    assert record == {
        "script": "scripts/prepare_synth.py",
        "script_sha256": prepare.sha256_hex(scripts() / "scripts" / "prepare_synth.py"),
        "toolchain": TOOLS,
        "inputs": [{"uri": NC, "checksum": multihash(b"-nc")}],
        "output_checksum": multihash(b"alpha-nc"),
    }
    # Nothing declared yet: the checksums are printed for the author to commit.
    assert multihash(b"alpha-nc") in " ".join(f.message for f in report.findings)


def test_an_unchanged_preparation_is_not_redone(tmp_path, remote, scripts):
    remote(NC, b"-nc")
    files, mirror = prepared_dataset(tmp_path, [NC], checksum=multihash(b"alpha-nc"), input_checksums={NC: multihash(b"-nc")}), tmp_path / "mirror"
    fetch_prepared(files, mirror, remote, scripts())
    remote(NC, b"-changed upstream")
    outcomes, report = fetch_prepared(files, mirror, remote, scripts())
    assert outcomes == {"alpha": "present"} and not report.findings
    assert (mirror / "prepared" / "synth" / "alpha.tif").read_bytes() == b"alpha-nc"


def test_a_script_edit_that_writes_the_same_bytes_changes_only_the_record(tmp_path, remote, scripts):
    remote(NC, b"-nc")
    files, mirror = prepared_dataset(tmp_path, [NC], checksum=multihash(b"alpha-nc"), input_checksums={NC: multihash(b"-nc")}), tmp_path / "mirror"
    fetch_prepared(files, mirror, remote, scripts())
    root = scripts("# reworded\n" + SCRIPT)
    outcomes, report = fetch_prepared(files, mirror, remote, root)
    assert outcomes == {"alpha": "prepared"} and not report.findings
    record = prepare.read_record(mirror / "prepared" / "synth" / "alpha.tif")
    assert record["script_sha256"] == prepare.sha256_hex(root / "scripts" / "prepare_synth.py")


def test_a_script_edit_that_writes_different_bytes_fails_and_leaves_nothing(tmp_path, remote, scripts):
    remote(NC, b"-nc")
    files, mirror = prepared_dataset(tmp_path, [NC], checksum=multihash(b"alpha-nc"), input_checksums={NC: multihash(b"-nc")}), tmp_path / "mirror"
    fetch_prepared(files, mirror, remote, scripts())
    outcomes, report = fetch_prepared(files, mirror, remote, scripts(SCRIPT + "    out.write(b'!')\n"))
    assert outcomes == {"alpha": "mismatch"} and [f.requirement for f in report.errors] == ["OBS-015"]
    assert not list((mirror / "prepared").rglob("*.*"))


def test_a_new_toolchain_reruns_the_script(tmp_path, remote, scripts):
    remote(NC, b"-nc")
    files, mirror = prepared_dataset(tmp_path, [NC], checksum=multihash(b"alpha-nc"), input_checksums={NC: multihash(b"-nc")}), tmp_path / "mirror"
    fetch_prepared(files, mirror, remote, scripts())
    tools = {**TOOLS, "python_lock_sha256": "b" * 64}
    outcomes, report = fetch_prepared(files, mirror, remote, scripts(), tools)
    assert outcomes == {"alpha": "prepared"} and not report.findings
    assert prepare.read_record(mirror / "prepared" / "synth" / "alpha.tif")["toolchain"] == tools


def test_an_input_that_does_not_match_its_checksum_is_not_prepared(tmp_path, remote, scripts):
    remote(NC, b"-nc")
    files = prepared_dataset(tmp_path, [NC], input_checksums={NC: "1220" + "0" * 64})
    mirror = tmp_path / "mirror"
    outcomes, report = fetch_prepared(files, mirror, remote, scripts())
    assert outcomes == {"alpha": "mismatch"} and [f.requirement for f in report.errors] == ["OBS-011"]
    assert not (mirror / "prepared").exists()


def test_a_failing_script_is_reported_with_its_output(tmp_path, remote, scripts):
    remote(NC, b"-nc")
    root = scripts("import sys\nprint('cannot read the grid', file=sys.stderr)\nsys.exit(3)\n")
    mirror = tmp_path / "mirror"
    outcomes, report = fetch_prepared(prepared_dataset(tmp_path, [NC]), mirror, remote, root)
    assert outcomes == {"alpha": "failed"} and [f.requirement for f in report.errors] == ["OBS-015"]
    assert "exit 3" in report.errors[0].message and "cannot read the grid" in report.errors[0].message
    assert not list((mirror / "prepared").rglob("*.*"))


def test_variables_share_an_input_and_each_gets_its_own_output(tmp_path, remote, scripts):
    other = "s3://bucket/b.tif"
    remote(NC, b"-nc")
    remote(other, b"-b")
    mirror = tmp_path / "mirror"
    outcomes, report = fetch_prepared(prepared_dataset(tmp_path, [NC, other], variables=("alpha", "beta")), mirror, remote, scripts())
    assert outcomes == {"alpha": "prepared", "beta": "prepared"} and report.ok
    assert (mirror / "prepared" / "synth" / "alpha.tif").read_bytes() == b"alpha-nc-b"
    assert (mirror / "prepared" / "synth" / "beta.tif").read_bytes() == b"beta-nc-b"


def test_a_missing_script_is_an_error(tmp_path, remote, scripts):
    remote(NC, b"-nc")
    root = scripts()
    (root / "scripts" / "prepare_synth.py").unlink()
    outcomes, report = fetch_prepared(prepared_dataset(tmp_path, [NC]), tmp_path / "mirror", remote, root)
    assert outcomes == {"alpha": "failed"} and [f.requirement for f in report.errors] == ["META-013"]

