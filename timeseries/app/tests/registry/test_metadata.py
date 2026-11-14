"""`/metadata` 1.0.0 (PROTO-001 to PROTO-004; APP-AT-005)."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from app.config import get_settings
from app.registry.compose import AppRegistry, ServedRelease, _layout, verify_releases
from app.registry.display import load_display
from app.registry.metadata import (
    SCHEMA_PATH,
    MetadataResponse,
    build_metadata,
    metadata_json_schema,
    resolution_label,
)
from app.registry.models import Overview, OverviewGrid, Pin
from app.tests.release_builder import build_release

REAL = Path(__file__).parent / "data" / "paleocar_v3-r-2026.10.07"


def _real_registry():
    """The paleocar_v3 prototype's overview, without its COGs."""
    overview = Overview.model_validate(
        yaml.safe_load((REAL / "overview.yml").read_text())
    )
    release = ServedRelease(
        Path("/releases") / overview.release_id, overview, _layout(overview)
    )
    return AppRegistry({"paleocar_v3": release})


def _committed_display():
    settings = get_settings()
    return load_display(
        Path(settings.display_preferences_path), Path(settings.display_palettes_path)
    )


# ---------------------------------------------------------------------------
# The contract: the committed JSON Schema (PROTO-002)


def test_committed_schema_matches_the_response_model():
    """Any change to the response's shape must update the committed schema.

    A renamed, removed or retyped field fails here; changed values don't.
    Regenerate after a deliberate change with `python -m app.registry.metadata`
    and review the diff: it is the contract the UI codes against.
    """
    assert SCHEMA_PATH.read_text() == metadata_json_schema()


def test_real_release_and_committed_display_files_build_a_response():
    """The real overview and the committed display files fit the contract."""
    response = build_metadata(_real_registry(), *_committed_display())

    round_trip = MetadataResponse.model_validate_json(response.model_dump_json())
    assert round_trip == response


# ---------------------------------------------------------------------------
# Behaviour


def test_metadata_has_no_legacy_or_renderer_fields():
    body = build_metadata(_real_registry(), *_committed_display()).model_dump(
        mode="json"
    )
    text = json.dumps(body)

    assert body["schema_version"] == "1.0.0"
    for legacy in ('"min"', '"max"', "wmsLayer", "timeseriesServiceUri", "colormap"):
        assert legacy not in text


def test_dataset_fields_come_from_the_overview_and_the_display_files():
    preferences, palettes = _committed_display()
    dataset = build_metadata(_real_registry(), preferences, palettes).datasets[0]
    shown = preferences.datasets["paleocar_v3"]

    # From the overview's grid and time axis.
    assert dataset.resolution_label == "30 arc-second (~800 m)"
    assert (dataset.time.origin, dataset.time.end, dataset.time.count) == (
        "0103",
        "2000",
        1898,
    )
    # From the display files, passed through unchanged (DISP-004, PROTO-003).
    assert dataset.default_variable == shown.default_variable
    for variable in dataset.variables:
        display = shown.variables[variable.id]
        assert variable.display.range == display.range
        assert variable.display.palette == display.palette
        assert variable.display.colors == palettes[display.palette].colors
        assert variable.display.ticks == (display.ticks or preferences.defaults.ticks)


def test_datasets_and_variables_sort_by_order_then_id(tmp_path):
    root = tmp_path / "releases"
    cube = np.zeros((2, 2, 2), dtype=np.float32)
    pins = [
        build_release(root, name, {"b": cube, "a": cube, "c": cube})
        for name in ("zeta", "alpha", "mid")
    ]
    registry = verify_releases(Pin(release_root=str(root), releases=pins), root)
    variable = {"palette": "ramp", "range": [0, 1]}
    preferences = {
        "datasets": {
            "zeta": {
                "order": 1,
                "variables": {
                    "a": {"order": 2, **variable},
                    "b": {"order": 1, **variable},
                    "c": {"order": 2, **variable},
                },
            },
            "alpha": {
                "order": 5,
                "variables": {v: {"order": 1, **variable} for v in "abc"},
            },
            "mid": {
                "order": 5,
                "variables": {v: {"order": 1, **variable} for v in "abc"},
            },
        }
    }
    (tmp_path / "p.yml").write_text(yaml.safe_dump(preferences))
    (tmp_path / "c.yml").write_text(
        yaml.safe_dump({"ramp": {"kind": "ramp", "colors": ["#000000", "#ffffff"]}})
    )

    response = build_metadata(
        registry, *load_display(tmp_path / "p.yml", tmp_path / "c.yml")
    )

    assert [d.id for d in response.datasets] == ["zeta", "alpha", "mid"]
    assert [v.id for v in response.datasets[0].variables] == ["b", "a", "c"]


@pytest.mark.parametrize(
    "code, transform, bbox, label",
    [
        (
            "EPSG:4269",
            (0.008333333333, 0, -115, 0, -0.008333333333, 43),
            (-115, 31, -102, 43),
            "30 arc-second (~800 m)",
        ),
        (
            "EPSG:4326",
            (0.25, 0, -180, 0, -0.25, 90),
            (-10, -10, 10, 10),
            "900 arc-second (~30 km)",
        ),
        ("EPSG:32612", (250.0, 0, 0, 0, -250.0, 0), (0, 0, 1, 1), "250 m"),
        ("EPSG:32612", (1000.0, 0, 0, 0, -1000.0, 0), (0, 0, 1, 1), "1 km"),
    ],
)
def test_resolution_label(code, transform, bbox, label):
    grid = OverviewGrid(code=code, shape=(10, 10), transform=transform)

    assert resolution_label(grid, bbox) == label
