"""STAC must agree with the COG bytes it describes (AT-021; VAL-007, AUTH-002)."""

import json

import pytest

from skope_release.crosscheck import check_stac_against_bytes
from skope_release.findings import Report
from skope_release.stac_serializer import write_stac

from stage_helpers import observed


@pytest.fixture
def staged(temporal_dataset, tmp_path):
    staging = tmp_path / "staging"
    _, obs = observed(temporal_dataset(), staging)
    write_stac(obs, staging)
    return obs, staging


def errors(obs, staging):
    report = Report()
    check_stac_against_bytes(obs, staging, report)
    return {f.requirement for f in report.errors}


def edit_asset(staging, change):
    item = staging / "items" / "synth--0100--0101.json"
    data = json.loads(item.read_text())
    change(data["assets"]["alpha"])
    item.write_text(json.dumps(data))


def test_agreement(staged):
    assert errors(*staged) == set()


@pytest.mark.parametrize(
    "change,requirement",
    [
        (lambda a: a.update({"file:checksum": "1220" + "0" * 64}), "AUTH-002"),
        (lambda a: a.update({"nodata": 0}), "AUTH-002"),
        (lambda a: a["bands"][0]["statistics"].update({"maximum": 1}), "STAC-008"),
        (lambda a: a.update({"href": "../cogs/alpha/elsewhere.tif"}), "VAL-007"),
    ],
)
def test_disagreement_is_detected(staged, change, requirement):
    obs, staging = staged
    edit_asset(staging, change)
    assert requirement in errors(obs, staging)
