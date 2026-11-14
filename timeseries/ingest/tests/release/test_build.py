"""Building and publishing a release locally (AT-006, AT-018, AT-021, AT-024; TXN-001 to TXN-003, TXN-008, TXN-018)."""

import logging
import os
import shutil
import stat
from pathlib import Path

import pytest
import yaml

from skope_release import build as build_module
from skope_release.build import build_and_publish, verify_reproducible
from skope_release.findings import ReleaseError, Report
from skope_release.manifest import verify_release
from skope_release.plan import Producer
from skope_release.stac_validate import validate_stac

from stage_helpers import plan_for

RELEASE_ID = "synth-r-2026.10.07"


def publish(directory, root, **kwargs):
    root.mkdir(parents=True, exist_ok=True)
    published, _ = build_and_publish(plan_for(directory, **kwargs), root)
    return published


def tree(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


@pytest.fixture
def released(temporal_dataset, tmp_path):
    directory = temporal_dataset()
    return publish(directory, tmp_path / "releases"), directory


def test_published_layout(released):
    """REL-001, ORG-008, ORG-009: a closed layout with deterministic names."""
    published, _ = released
    assert published.path.name == RELEASE_ID
    assert tree(published.path) == [
        "cogs/alpha/synth--0100--0101.tif",
        "cogs/alpha/synth--0102--0103.tif",
        "cogs/alpha/synth--0104--0104.tif",
        "cogs/beta/synth--0100--0101.tif",
        "cogs/beta/synth--0102--0103.tif",
        "cogs/beta/synth--0104--0104.tif",
        "collection.json",
        "items/synth--0100--0101.json",
        "items/synth--0102--0103.json",
        "items/synth--0104--0104.json",
        "overview.yml",
        "release-manifest.json",
    ]
    staging = published.path.parent / ".staging"
    assert [p for p in staging.iterdir()] == []  # staging renamed away; lease released
    assert len(published.manifest_sha256) == 64


def test_published_release_is_read_only_for_everyone(released):
    """TXN-018: files 0444, directories 0555, whatever the umask."""
    published, _ = released
    for path in [published.path, *published.path.rglob("*")]:
        assert stat.S_IMODE(path.stat().st_mode) == (0o555 if path.is_dir() else 0o444), path


def test_modes_are_set_around_the_rename(temporal_dataset, tmp_path, monkeypatch):
    """TXN-018: contents read-only before the rename; the release directory stays
    writable until after it, since moving a directory to a new parent needs that."""
    rename = os.rename
    seen = {}

    def checked_rename(source, destination):
        source = Path(source)
        seen["top"] = stat.S_IMODE(source.stat().st_mode)
        seen["manifest"] = stat.S_IMODE((source / "release-manifest.json").stat().st_mode)
        seen["cogs"] = stat.S_IMODE((source / "cogs").stat().st_mode)
        rename(source, destination)

    monkeypatch.setattr(build_module.os, "rename", checked_rename)
    publish(temporal_dataset(), tmp_path / "releases")
    assert seen["top"] & stat.S_IWUSR
    assert (seen["manifest"], seen["cogs"]) == (0o444, 0o555)


def test_long_steps_log_progress(temporal_dataset, tmp_path, caplog):
    """Inspection and the manifest's checksums log each file as it finishes."""
    caplog.set_level(logging.INFO)
    published = publish(temporal_dataset(), tmp_path / "releases")
    messages = [r.getMessage() for r in caplog.records]
    inspected = [m for m in messages if m.startswith("inspected ")]
    assert len(inspected) == 6 and inspected[-1].startswith("inspected 6 of 6 COGs: cogs/")
    checksummed = [m for m in messages if m.startswith("checksummed ")]
    assert len(checksummed) == 11 and checksummed[-1].startswith("checksummed 11 of 11 files")

    caplog.clear()
    assert verify_release(published.path, Report()) is not None
    verified = [r.getMessage() for r in caplog.records if r.getMessage().startswith("verified ")]
    assert len(verified) == 11 and verified[-1].startswith("verified 11 of 11 files")


def test_static_release(static_dataset, tmp_path):
    """AT-024 (synthetic): one Collection asset, no Items, no time."""
    published = publish(static_dataset(), tmp_path / "releases", release_id="elev-r-2026.10.07")
    assert tree(published.path) == ["cogs/elevation.tif", "collection.json", "overview.yml", "release-manifest.json"]
    overview = yaml.safe_load((published.path / "overview.yml").read_text())
    assert "time" not in overview["dataset"]
    assert overview["variables"]["elevation"]["asset_href"] == "cogs/elevation.tif"


def test_release_with_declared_empty_timesteps(temporal_dataset, tmp_path):
    """OBS-007, STAC-008: a declared empty band passes every build check."""
    sources = {"alpha": {"empty": ["0102", "0103"]}, "beta": {"empty": ["0102", "0103"]}}
    gap = {"first": "0102", "last": "0103", "reason": "No reconstruction."}
    published = publish(temporal_dataset(source_kwargs=sources, curated={"empty_timesteps": [gap]}), tmp_path / "releases")
    report = Report()
    assert verify_release(published.path, report) is not None, report.to_text()


def test_builds_are_byte_reproducible(temporal_dataset, tmp_path):
    """AT-006, REL-009, API-007."""
    directory = temporal_dataset()
    first = publish(directory, tmp_path / "a")
    second = publish(directory, tmp_path / "b")
    for rel in tree(first.path):
        assert (first.path / rel).read_bytes() == (second.path / rel).read_bytes(), rel
    report = verify_reproducible(plan_for(directory), first.path)
    assert report.ok, report.to_text()


def test_a_release_relocates_on_its_own(released, tmp_path):
    """REL-002, AT-021: relative links survive moving the directory."""
    published, _ = released
    moved = tmp_path / "elsewhere" / RELEASE_ID
    shutil.copytree(published.path, moved)
    report = Report()
    assert verify_release(moved, report) is not None, report.to_text()
    validate_stac(moved, report)
    assert report.ok, report.to_text()


def test_failure_leaves_no_release(temporal_dataset, tmp_path, monkeypatch):
    """AT-018, TXN-001, TXN-008: no release path, staging kept, earlier releases untouched."""
    root = tmp_path / "releases"
    earlier = publish(temporal_dataset(), root)
    before = {rel: (earlier.path / rel).read_bytes() for rel in tree(earlier.path)}

    directory = temporal_dataset(dataset_id="other")

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected")

    monkeypatch.setattr(build_module, "write_overview", fail)
    with pytest.raises(RuntimeError, match="injected"):
        build_and_publish(plan_for(directory, release_id="other-r-2026.10.07"), root)
    assert not (root / "other-r-2026.10.07").exists()
    staged = [p for p in (root / ".staging").iterdir() if p.is_dir() and p.name.startswith("other-r-")]
    assert staged and not (staged[0] / "release-manifest.json").exists()
    assert not list((root / ".staging").glob("*.lease"))
    assert {rel: (earlier.path / rel).read_bytes() for rel in tree(earlier.path)} == before


def test_publication_preconditions(released, tmp_path):
    """TXN-002 (no overwrite) and REL-009 (clean producer revision)."""
    published, directory = released
    with pytest.raises(ReleaseError) as exc:
        build_and_publish(plan_for(directory), published.path.parent)
    assert exc.value.report.errors[0].requirement == "TXN-002"
    dirty = Producer("p", "0.1.0", "abc123-dirty")
    with pytest.raises(ReleaseError) as exc:
        build_and_publish(plan_for(directory, producer=dirty), tmp_path / "fresh")
    assert exc.value.report.errors[0].requirement == "REL-009"
