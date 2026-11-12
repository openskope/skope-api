"""Release identity and the declaration digest (AT-006, AT-034; REL-006, MAN-010, MAN-011)."""

import pytest
import yaml

from skope_release.documents import validate
from skope_release.findings import Report
from skope_release.identity import (
    check_release_id,
    declaration_digest,
    declaration_projection,
    parse_release_id,
)
from skope_release.models import Curated, Ledger, SourceManifest

from release_fixtures import curated_doc, manifest_doc

DIGEST = "a" * 64


def resolved(**manifest_overrides):
    report = Report()
    cur = validate(Curated, curated_doc("synth", ["alpha", "beta"]), requirement="T", path="c", report=report)
    man = manifest_doc("synth", {"alpha": "s3://b/a.tif", "beta": "s3://b/b.tif"})
    for variable in man["variables"]:
        variable["source"]["checksum"] = "1220" + "0" * 64
    man.update(manifest_overrides)
    return cur, validate(SourceManifest, man, requirement="T", path="m", report=report)


def digest_of(cur, man):
    return declaration_digest(declaration_projection(cur, man))


def ids(*entries):
    return Ledger.model_validate({"releases": [{"release_id": r, "declaration_digest": d} for r, d in entries]})


@pytest.mark.parametrize(
    "release_id,expected",
    [
        ("paleocar_v3-r-2026.10.07", ("paleocar_v3", "2026.10.07", 1)),
        ("paleocar_v3-r-2026.10.07-2", ("paleocar_v3", "2026.10.07", 2)),
    ],
)
def test_parse_release_id(release_id, expected):
    assert parse_release_id(release_id) == expected


@pytest.mark.parametrize("release_id", ["paleocar_v3-r-2026.10.07-1", "paleocar_v3-r-2026.10.07-02", "paleocar_v3-2026.10.07", "r-2026.10.07"])
def test_invalid_release_ids(release_id):
    with pytest.raises(ValueError):
        parse_release_id(release_id)


def check(release_id, ledger=Ledger(), created="2026-10-07T12:00:00Z", digest=DIGEST, dataset="synth"):
    report = Report()
    check_release_id(release_id, dataset_id=dataset, created=created, digest=digest, ledger=ledger, report=report)
    return report


def test_release_id_rules():
    assert check("synth-r-2026.10.07").ok
    assert not check("other-r-2026.10.07").ok  # names another dataset
    assert not check("synth-r-2026.10.08").ok  # date must equal release.created's UTC date
    assert not check("synth-r-2026.10.07-2").ok  # -2 only after the first ID of the day


def test_same_day_sequence_is_per_dataset():
    ledger = ids(("synth-r-2026.10.07", "b" * 64), ("other-r-2026.10.07", "c" * 64))
    assert not check("synth-r-2026.10.07-3", ledger).ok
    assert check("synth-r-2026.10.07-2", ledger).ok


def test_an_id_is_bound_to_one_digest():
    ledger = ids(("synth-r-2026.10.07", "b" * 64))
    assert not check("synth-r-2026.10.07", ledger).ok  # MAN-011
    assert check("synth-r-2026.10.07", ledger, digest="b" * 64).ok  # a retry keeps the ID


def test_two_ids_never_share_a_digest():
    ledger = ids(("synth-r-2026.10.06", DIGEST))
    assert not check("synth-r-2026.10.07", ledger).ok


def test_withdrawn_ids_are_not_reused():
    ledger = Ledger.model_validate(
        {"releases": [{"release_id": "synth-r-2026.10.07", "declaration_digest": DIGEST, "withdrawn": "bad nodata"}]}
    )
    assert not check("synth-r-2026.10.07", ledger).ok


def test_digest_is_deterministic_and_identity_profiled():
    cur, man = resolved()
    projection = declaration_projection(cur, man)
    assert projection["identity_profile"] == "openskope-release-declaration-v1"
    assert digest_of(cur, man) == digest_of(*resolved())
    assert len(digest_of(cur, man)) == 64


def test_digest_ignores_variable_order():
    cur, man = resolved()
    swapped = man.model_copy(update={"variables": list(reversed(man.variables))})
    assert digest_of(cur, man) == digest_of(cur, swapped)


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m["release"].update(chunk_size=3),
        lambda m: m["release"].update(created="2026-10-07T12:00:01Z"),
        lambda m: m["release"]["cog"].update(blocksize=512),
        lambda m: m["variables"][0]["encoding"].update(overview_resampling="NEAREST"),
        lambda m: m["variables"][0]["source"].update(checksum="1220" + "1" * 64),
    ],
)
def test_every_declared_input_changes_the_digest(change):
    cur, man = resolved()
    data = man.model_dump(mode="json")
    change(data)
    report = Report()
    changed = validate(SourceManifest, data, requirement="T", path="m", report=report)
    assert digest_of(cur, man) != digest_of(cur, changed)


def test_the_digest_of_a_fixed_declaration_is_stable():
    """REL-006: a model change that alters every declaration's digest would orphan the ledgers' release IDs."""
    assert digest_of(*resolved()) == "65c20065ab0f604844b5e3af328fadd930687ab1bae65beef66664937328c43c"


def test_a_prepared_source_projects_its_inputs_not_a_uri():
    """META-013, MAN-010: input checksums are part of the declaration; a source projects only its origin."""
    cur, man = resolved()
    data = man.model_dump(mode="json", exclude_none=True)
    data["variables"][0]["source"] = {
        "bands": "descriptions", "checksum": "1220" + "0" * 64,
        "prepare": {"script": "scripts/prepare_synth.py", "inputs": [{"uri": "https://example.org/a.nc", "checksum": "1220" + "2" * 64}]},
    }
    report = Report()
    prepared = validate(SourceManifest, data, requirement="T", path="m", report=report)
    source = declaration_projection(cur, prepared)["source_manifest"]["variables"]["alpha"]["source"]
    assert "uri" not in source and "prepare" in source
    assert "prepare" not in declaration_projection(cur, man)["source_manifest"]["variables"]["alpha"]["source"]
    data["variables"][0]["source"]["prepare"]["inputs"][0]["checksum"] = "1220" + "3" * 64
    changed = validate(SourceManifest, data, requirement="T", path="m", report=report)
    assert digest_of(cur, prepared) != digest_of(cur, changed)


def test_curated_text_changes_the_digest():
    cur, man = resolved()
    edited = cur.model_copy(update={"description": "A typo fixed."})
    assert digest_of(cur, man) != digest_of(edited, man)


def test_projection_excludes_release_ids_and_paths():
    cur, man = resolved()
    text = yaml.safe_dump(declaration_projection(cur, man))
    assert "-r-20" not in text
    assert "/vsis3/" not in text and "mirror" not in text


def test_projection_requires_resolved_checksums():
    cur, man = resolved()
    data = man.model_dump(mode="json")
    data["variables"][0]["source"]["checksum"] = None
    unresolved = SourceManifest.model_validate_json(__import__("json").dumps(data))
    with pytest.raises(ValueError):
        declaration_projection(cur, unresolved)
