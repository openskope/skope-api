"""Startup composition: every pinned release is checked, and any failure
refuses the whole registry (PIN-004, PIN-005; APP-AT-004)."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pytest
import yaml

from app.registry.compose import (
    ReleaseRefused,
    log_refusals,
    multihash,
    normalize_key,
    verify_releases,
)
from app.registry.models import Pin
from app.tests.release_builder import build_release
from app.vendor.timeaxis import MalformedTimestep, UnknownTimestep


def _cube(count=5, value=None):
    values = np.arange(count, dtype=np.float32) if value is None else [value] * count
    return np.stack([np.full((3, 4), v, dtype=np.float32) for v in values])


@pytest.fixture
def root(tmp_path):
    return tmp_path / "releases"


@pytest.fixture
def annual(root):
    return build_release(root, "annual", {"ppt": _cube(), "tmax": _cube()})


def _pin(root, *releases):
    return Pin(release_root=str(root), releases=list(releases))


def _refusals(root, *releases, full=False):
    with pytest.raises(ReleaseRefused) as exc_info:
        verify_releases(_pin(root, *releases), root, full=full)
    return exc_info.value.refusals


# ---------------------------------------------------------------------------
# Composition


def test_composes_every_pinned_release(root, annual):
    monthly = build_release(
        root,
        "monthly",
        {"ppt": _cube(24)},
        origin="0001-01",
        step="P1M",
        precision="month",
        chunk_size=12,
    )

    registry = verify_releases(_pin(root, annual, monthly), root)

    assert sorted(registry.releases) == ["annual", "monthly"]
    assert registry.get("annual").overview.release_id == annual.release_id
    assert registry.get("missing") is None


def test_full_verification_passes_an_intact_release(root, annual):
    assert verify_releases(_pin(root, annual), root, full=True).get("annual")


def test_resolves_a_timestep_to_its_file_and_band(root, annual):
    release = verify_releases(_pin(root, annual), root).get("annual")

    path, band = release.resolve("ppt", "0004")

    assert path == str(root / annual.release_id / "cogs/ppt/annual--0003--0004.tif")
    assert band == 2


def test_tile_keys_are_exact(root, annual):
    release = verify_releases(_pin(root, annual), root).get("annual")

    with pytest.raises(MalformedTimestep):
        release.resolve("ppt", "0004-01-01")
    with pytest.raises(UnknownTimestep):
        release.resolve("ppt", "0006")


def test_selects_a_range_across_files_in_axis_order(root, annual):
    release = verify_releases(_pin(root, annual), root).get("annual")

    mapping, timesteps = release.select("ppt", "0002", "0005")

    base = root / annual.release_id / "cogs/ppt"
    assert list(mapping.items()) == [
        (str(base / "annual--0001--0002.tif"), [2]),
        (str(base / "annual--0003--0004.tif"), [1, 2]),
        (str(base / "annual--0005--0005.tif"), [1]),
    ]
    assert timesteps == ["0002", "0003", "0004", "0005"]


def test_range_bounds_may_be_finer_than_the_dataset(root, annual):
    release = verify_releases(_pin(root, annual), root).get("annual")

    _, timesteps = release.select("ppt", "0002-01-01", "0003-01-01T00:00:00Z")

    assert timesteps == ["0002", "0003"]


@pytest.mark.parametrize(
    "key, precision, expected",
    [
        ("0103", "year", "0103"),
        ("0103-01", "year", "0103"),
        ("0103-02-01", "month", "0103-02"),
    ],
)
def test_normalize_key(key, precision, expected):
    assert normalize_key(key, precision) == expected


@pytest.mark.parametrize(
    "key, precision", [("0103-06-01", "year"), ("0103", "month"), ("103", "year")]
)
def test_normalize_rejects_other_keys(key, precision):
    with pytest.raises(MalformedTimestep):
        normalize_key(key, precision)


# ---------------------------------------------------------------------------
# Refusals (PIN-004, PIN-005)


def test_wrong_manifest_digest_is_refused(root, annual):
    pinned = annual.model_copy(update={"manifest_sha256": "0" * 64})

    refusals = _refusals(root, pinned)

    assert [(r.requirement, r.check) for r in refusals] == [
        ("PIN-004", "manifest digest")
    ]
    assert refusals[0].expected == "0" * 64
    assert refusals[0].observed == annual.manifest_sha256


def test_wrong_declaration_digest_is_refused(root, annual):
    pinned = annual.model_copy(update={"declaration_digest": "e" * 64})

    assert [r.check for r in _refusals(root, pinned)] == [
        "manifest declaration digest",
        "overview declaration digest",
    ]


def test_edited_overview_is_refused(root, annual):
    overview = root / annual.release_id / "overview.yml"
    overview.write_text(overview.read_text().replace("Test region", "Edited region"))

    checks = [r.check for r in _refusals(root, annual)]

    assert "overview checksum" in checks


def test_missing_or_truncated_cog_is_refused(root, annual):
    cogs = root / annual.release_id / "cogs" / "ppt"
    (cogs / "annual--0001--0002.tif").unlink()
    with (cogs / "annual--0003--0004.tif").open("r+b") as f:
        f.truncate(100)

    checks = sorted(r.check for r in _refusals(root, annual))

    assert checks == [
        "file exists: cogs/ppt/annual--0001--0002.tif",
        "file size: cogs/ppt/annual--0003--0004.tif",
    ]


def test_changed_bytes_of_the_same_size_fail_only_full_verification(root, annual):
    cog = root / annual.release_id / "cogs" / "ppt" / "annual--0001--0002.tif"
    data = bytearray(cog.read_bytes())
    data[-1] ^= 0xFF
    cog.write_bytes(bytes(data))

    assert verify_releases(_pin(root, annual), root).get("annual")
    refusals = _refusals(root, annual, full=True)
    assert [(r.requirement, r.check) for r in refusals] == [
        ("TXN-011", "file checksum: cogs/ppt/annual--0001--0002.tif")
    ]


def test_missing_release_is_refused(root, annual):
    pinned = annual.model_copy(update={"release_id": "annual-r-2026.02.02"})

    assert [r.check for r in _refusals(root, pinned)] == ["release manifest exists"]


def test_two_releases_of_one_dataset_are_refused(root, annual):
    other = build_release(root, "annual", {"ppt": _cube()}, date="2026.01.02")

    refusals = _refusals(root, annual, other)

    assert [(r.requirement, r.release_id) for r in refusals] == [
        ("PIN-005", other.release_id)
    ]


def test_one_bad_release_refuses_the_whole_registry(root, annual):
    good = build_release(root, "good", {"ppt": _cube()})
    bad = annual.model_copy(update={"manifest_sha256": "0" * 64})

    refusals = _refusals(root, good, bad)

    assert {r.dataset for r in refusals} == {"annual"}


def test_non_identity_scale_is_refused(root):
    scaled = build_release(
        root, "scaled", {"ppt": _cube()}, variable_fields={"ppt": {"scale": 0.1}}
    )

    refusals = _refusals(root, scaled)

    assert [r.requirement for r in refusals] == ["PROTO-010"]


def test_refusals_are_logged_one_structured_error_each(root, annual, caplog):
    pinned = annual.model_copy(
        update={"manifest_sha256": "0" * 64, "declaration_digest": "e" * 64}
    )
    refusals = _refusals(root, pinned)

    with caplog.at_level(logging.ERROR):
        log_refusals(refusals)

    records = [json.loads(r.getMessage().split(": ", 1)[1]) for r in caplog.records]
    assert len(records) == 3
    assert {r["check"] for r in records} == {
        "manifest digest",
        "manifest declaration digest",
        "overview declaration digest",
    }
    for record in records:
        assert set(record) == {
            "requirement",
            "dataset",
            "release_id",
            "check",
            "expected",
            "observed",
        }


@pytest.mark.parametrize(
    "case",
    json.loads(Path("tests/contracts/release-identity.json").read_text()),
    ids=lambda case: case["name"],
)
def test_shared_release_identity_contract(root, annual, case):
    path = root / annual.release_id
    overview_path = path / "overview.yml"
    manifest_path = path / "release-manifest.json"
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
            entry.update(
                size=overview_path.stat().st_size, checksum=multihash(overview_path)
            )
    manifest_path.write_text(json.dumps(documents["manifest"]))
    pinned = annual.model_copy(
        update={
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        }
    )

    if case["valid"]:
        assert verify_releases(_pin(root, pinned), root).get(annual.dataset)
    else:
        checks = [r.check for r in _refusals(root, pinned)]
        assert any(
            "dataset" in check or "digest" in check or "release ID" in check
            for check in checks
        )
        assert "overview checksum" not in checks


def test_rule_names_the_real_release_cogs():
    """The paleocar_v3 prototype's overview and manifest agree on every COG path."""
    from app.registry.compose import _layout
    from app.registry.models import Overview, ReleaseManifest

    real = Path(__file__).parent / "data" / "paleocar_v3-r-2026.10.07"
    overview = Overview.model_validate(
        yaml.safe_load((real / "overview.yml").read_text())
    )
    manifest = ReleaseManifest.model_validate_json(
        (real / "release-manifest.json").read_text()
    )
    layout = _layout(overview)

    named = {
        layout.cog_path(v, c)
        for v in overview.variables
        for c in range(layout.chunk_count)
    }
    assert named == {f.path for f in manifest.files if f.path.startswith("cogs/")}
    assert layout.resolve("ppt_annual", "0417") == (
        "cogs/ppt_annual/paleocar_v3--0403--0502.tif",
        15,
    )
