"""`skope-release pin`: the ledger, pin, and source checksum edits (spec §20.3, PIN-001, META-004)."""

import json
import re

import pytest
import yaml

from skope_release.build import build_and_publish
from skope_release.cli import main

from stage_helpers import plan_for

OTHER = """\
  # Another dataset, left alone.
  - dataset: other
    release_id: other-r-2026.01.01
    declaration_digest: 1111111111111111111111111111111111111111111111111111111111111111
    manifest_sha256: 2222222222222222222222222222222222222222222222222222222222222222
"""
PIN = f"""\
# The release set staging serves.
release_root: /srv/datasets/releases
releases:
  # Prototype, about to be replaced.
  - dataset: synth
    release_id: synth-r-2026.01.01
    declaration_digest: {"3" * 64}
    manifest_sha256: {"4" * 64}
{OTHER}"""


@pytest.fixture
def release(temporal_dataset, tmp_path):
    """A published release, its dataset's authoring files, and a staging pin file."""
    directory = temporal_dataset()
    root = tmp_path / "releases"
    root.mkdir()
    published, _ = build_and_publish(plan_for(directory), root)
    pins = tmp_path / "pins"
    pins.mkdir()
    (pins / "staging.yml").write_text(PIN)
    return published, directory, pins


def pin(release, tmp_path, *extra):
    published, _, pins = release
    return main(["--datasets", str(tmp_path / "datasets"), "pin", str(published.path), "--env", "staging", "--pins", str(pins), *extra])


def snapshot(directory, pins):
    files = [directory / "source-manifest.yml", directory / "releases.yml", pins / "staging.yml"]
    return {p.name: p.read_text() if p.exists() else None for p in files}


def test_shows_the_edits_and_writes_nothing(release, tmp_path, capsys):
    published, directory, pins = release
    before = snapshot(directory, pins)
    assert pin(release, tmp_path) == 0
    out = capsys.readouterr().out
    assert f"+  - release_id: \"{published.release_id}\"" in out
    assert f"+    manifest_sha256: {published.manifest_sha256}" in out
    assert re.search(r'^\+\s+checksum: "1220', out, re.MULTILINE)
    assert "nothing written" in out
    assert snapshot(directory, pins) == before


def test_writes_the_ledger_pin_and_checksums(release, tmp_path, capsys):
    published, directory, pins = release
    assert pin(release, tmp_path, "--write") == 0

    ledger = yaml.safe_load((directory / "releases.yml").read_text())
    assert ledger["releases"] == [{"release_id": published.release_id, "declaration_digest": published.declaration_digest}]

    text = (pins / "staging.yml").read_text()
    assert text.startswith("# The release set staging serves.") and text.endswith(OTHER)
    assert "# Prototype, about to be replaced." in text
    entries = yaml.safe_load(text)["releases"]
    assert entries[0] == {
        "dataset": "synth",
        "release_id": published.release_id,
        "declaration_digest": published.declaration_digest,
        "manifest_sha256": published.manifest_sha256,
    }
    assert entries[1]["dataset"] == "other"

    manifest = json.loads((published.path / "release-manifest.json").read_text())
    declared = {v["id"]: v["source"]["checksum"] for v in yaml.safe_load((directory / "source-manifest.yml").read_text())["variables"]}
    assert declared == {s["id"]: s["checksum"] for s in manifest["sources"]}
    # Declaring the measured checksums leaves the declaration, and so its digest, unchanged.
    assert plan_for(directory, release_id=None).declaration_digest == published.declaration_digest

    capsys.readouterr()
    assert pin(release, tmp_path) == 0
    assert "nothing to change" in capsys.readouterr().out


def test_a_conflicting_ledger_entry_stops_everything(release, tmp_path, capsys):
    published, directory, pins = release
    (directory / "releases.yml").write_text(f'releases:\n  - release_id: "{published.release_id}"\n    declaration_digest: "{"5" * 64}"\n')
    before = snapshot(directory, pins)
    assert pin(release, tmp_path, "--write") == 1
    assert "REL-006" in capsys.readouterr().out
    assert snapshot(directory, pins) == before


def test_a_declared_checksum_that_differs_stops_everything(release, tmp_path, capsys):
    published, directory, pins = release
    path = directory / "source-manifest.yml"
    doc = yaml.safe_load(path.read_text())
    doc["variables"][0]["source"]["checksum"] = "1220" + "6" * 64
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    before = snapshot(directory, pins)
    assert pin(release, tmp_path, "--write") == 1
    assert "META-004" in capsys.readouterr().out
    assert snapshot(directory, pins) == before


def test_a_dataset_not_pinned_yet_is_added(release, tmp_path):
    published, directory, pins = release
    (pins / "staging.yml").write_text(f"release_root: /srv/datasets/releases\nreleases:\n{OTHER}")
    assert pin(release, tmp_path, "--write") == 0
    entries = yaml.safe_load((pins / "staging.yml").read_text())["releases"]
    assert [e["dataset"] for e in entries] == ["other", "synth"]
