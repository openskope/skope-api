"""Fetching sources into the mirror (OBS-011, OBS-014)."""

from urllib.parse import urlparse

import pytest

from skope_release import gdalio
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
