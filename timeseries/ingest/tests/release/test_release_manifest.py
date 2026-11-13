"""The release manifest and verification (AT-015; MAN-001 to MAN-008, REL-001, TXN-011)."""

import json
from pathlib import Path

import pytest
import yaml

from skope_release.findings import Report
from skope_release.gdalio import sha256_multihash
from skope_release.manifest import build_manifest, roles_for, verify_release, write_manifest
from skope_release.overview import write_overview
from skope_release.stac_serializer import write_stac

from stage_helpers import observed


@pytest.fixture
def release(temporal_dataset, tmp_path):
    """A complete release directory, named like a published one."""
    root = tmp_path / "synth-r-2026.10.07"
    plan, obs = observed(temporal_dataset(), root)
    write_stac(obs, root)
    write_overview(root, release_id=plan.release_id, declaration_digest=plan.declaration_digest, report=Report())
    report = Report()
    manifest = build_manifest(obs, root, report)
    assert manifest is not None, report.to_text()
    write_manifest(manifest, root)
    return root, obs


def test_manifest_content(release):
    root, obs = release
    manifest = json.loads((root / "release-manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["release_id"] == "synth-r-2026.10.07"
    assert manifest["declaration"] == {
        "identity_profile": "openskope-release-declaration-v1", "digest_algorithm": "sha256", "digest": obs.declaration_digest,
    }
    assert [s["id"] for s in manifest["sources"]] == ["alpha", "beta"]
    paths = [f["path"] for f in manifest["files"]]
    assert "release-manifest.json" not in paths  # MAN-006
    assert paths == sorted(paths) and len(paths) == 11
    for forbidden in ('"crs"', '"transform"', '"nodata"', '"unit"', '"statistics"', '"bbox"'):
        assert forbidden not in json.dumps(manifest)  # MAN-004


def test_roles_follow_the_closed_layout():
    assert roles_for("collection.json") == ["stac", "collection"]
    assert roles_for("overview.yml") == ["derived"]
    assert roles_for("items/ds--0100--0101.json") == ["stac", "item"]
    assert roles_for("cogs/v/ds--0100--0101.tif") == ["data"]
    assert roles_for("cogs/v.tif") == ["data"]
    assert roles_for("notes.txt") is None and roles_for("lookup.json") is None


def test_unexpected_files_fail_the_manifest(release):
    root, obs = release
    (root / "lookup.json").write_text("{}")
    report = Report()
    assert build_manifest(obs, root, report) is None
    assert "REL-001" in {f.requirement for f in report.errors}


def test_verification(release):
    root, _ = release
    report = Report()
    assert verify_release(root, report) is not None, report.to_text()


@pytest.mark.parametrize(
    "case", json.loads(Path("tests/contracts/release-identity.json").read_text()),
    ids=lambda case: case["name"],
)
def test_shared_release_identity_contract(release, case):
    root, _ = release
    overview_path = root / "overview.yml"
    manifest_path = root / "release-manifest.json"
    documents = {
        "manifest": json.loads(manifest_path.read_text()),
        "overview": yaml.safe_load(overview_path.read_text()),
    }
    if case["document"]:
        target = documents[case["document"]]
        for key in case["path"][:-1]:
            target = target[key]
        target[case["path"][-1]] = case["value"]
    overview_path.write_text(yaml.safe_dump(documents["overview"]))
    for entry in documents["manifest"]["files"]:
        if entry["path"] == "overview.yml":
            entry.update(size=overview_path.stat().st_size, checksum=sha256_multihash(overview_path))
    manifest_path.write_text(json.dumps(documents["manifest"]))
    report = Report()

    assert (verify_release(root, report) is not None) == case["valid"], report.to_text()
    if not case["valid"]:
        assert {finding.requirement for finding in report.errors} == {"AUTH-002"}


def test_verification_detects_tampering(release):
    root, _ = release
    extra = root / "notes.txt"
    extra.write_text("hello")
    report = Report()
    assert verify_release(root, report) is None and "REL-001" in {f.requirement for f in report.errors}
    extra.unlink()

    cog = root / "cogs/alpha/synth--0100--0101.tif"
    data = bytearray(cog.read_bytes())
    data[-1] ^= 0xFF
    cog.write_bytes(bytes(data))
    assert verify_release(root, Report(), full=False) is not None  # same size: the startup check passes
    full = Report()
    assert verify_release(root, full) is None and "MAN-005" in {f.requirement for f in full.errors}
