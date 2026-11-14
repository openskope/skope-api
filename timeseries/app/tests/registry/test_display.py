"""Display preferences and palettes (DISP-001 to DISP-013; APP-AT-001, APP-AT-002)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from app.config import get_settings
from app.registry.compose import verify_releases
from app.registry.display import DisplayFileError, check_display, load_display
from app.registry.models import Pin
from app.tests.release_builder import build_release

PALETTES = {
    "ramp": {"kind": "ramp", "colors": ["#000000", "#ffffff"]},
    "pair": {"kind": "set", "colors": ["#e8e8e8", "#2b8c3e"]},
}


def _variable(**fields):
    return {"order": 10, "palette": "ramp", "range": [0, 10], **fields}


def _preferences(variables=None, **dataset):
    return {
        "datasets": {
            "ds": {"order": 10, "variables": variables or {"v": _variable()}, **dataset}
        }
    }


def _load(tmp_path, preferences, palettes=PALETTES):
    (tmp_path / "preferences.yml").write_text(yaml.safe_dump(preferences))
    (tmp_path / "palettes.yml").write_text(yaml.safe_dump(palettes))
    return load_display(tmp_path / "preferences.yml", tmp_path / "palettes.yml")


# ---------------------------------------------------------------------------
# Checks that need no release data (DISP-012)


def test_committed_display_files_are_valid():
    """CI's DISP-012 check on deploy/display/, which the test image mounts."""
    settings = get_settings()

    preferences, palettes = load_display(
        Path(settings.display_preferences_path), Path(settings.display_palettes_path)
    )

    assert "paleocar_v3" in preferences.datasets
    assert len(palettes["viridis"].colors) == 256


@pytest.mark.parametrize(
    "preferences, message",
    [
        (_preferences({"v": _variable(palette="missing")}), "is not defined"),
        (
            _preferences({"v": _variable(type="continuous", palette="pair")}),
            "needs a ramp",
        ),
        (
            _preferences({"v": _variable(type="continuous", range=None)}),
            "needs a range",
        ),
        (_preferences({"v": _variable(range=[10, 0])}), "lower first"),
        (_preferences({"v": _variable(range=[0, float("inf")])}), "finite"),
        (_preferences({"v": _variable(ticks=[0, 20])}), "within the range"),
        (_preferences({"v": _variable(order="10")}), "order"),
        (_preferences({"v": _variable(colormap="viridis")}), "colormap"),
        (_preferences(map_view={"center": [37, -108], "zoom": 4}), "center"),
        (
            _preferences(map_view={"center": {"lon": -108, "lat": 137}, "zoom": 4}),
            "lat",
        ),
    ],
)
def test_bad_display_files_are_rejected(tmp_path, preferences, message):
    with pytest.raises(DisplayFileError, match=message):
        _load(tmp_path, preferences)


def test_ramp_needs_two_colours(tmp_path):
    with pytest.raises(DisplayFileError, match="two colours"):
        _load(
            tmp_path, _preferences(), {"ramp": {"kind": "ramp", "colors": ["#000000"]}}
        )


# ---------------------------------------------------------------------------
# Checks against the pinned releases (DISP-013)


@pytest.fixture
def registry(tmp_path):
    root = tmp_path / "releases"
    cube = np.zeros((2, 2, 2), dtype=np.uint8)
    pinned = build_release(
        root,
        "ds",
        {"v": cube, "niche": cube},
        variable_fields={"niche": {"categories": {0: "Outside", 1: "Inside"}}},
    )
    return verify_releases(Pin(release_root=str(root), releases=[pinned]), root)


def _check(tmp_path, registry, **overrides):
    palettes = overrides.pop("palettes", PALETTES)
    variables = {
        "v": _variable(),
        "niche": {"order": 20, "palette": "pair"},
    }
    variables.update(overrides.pop("variables", {}))
    preferences, palettes = _load(
        tmp_path, _preferences(variables, **overrides), palettes
    )
    return [
        (r.requirement, r.check) for r in check_display(registry, preferences, palettes)
    ]


def test_good_entries_pass(tmp_path, registry):
    assert _check(tmp_path, registry, default_variable="niche") == []


def test_every_pinned_variable_needs_an_entry(tmp_path, registry):
    preferences, palettes = _load(tmp_path, _preferences({"v": _variable()}))

    refusals = check_display(registry, preferences, palettes)

    assert [(r.requirement, r.check) for r in refusals] == [
        ("DISP-001", "niche: variable display entry")
    ]


def test_entries_for_unpinned_datasets_are_allowed(tmp_path, registry):
    preferences = _preferences(
        {"v": _variable(), "niche": {"order": 20, "palette": "pair"}}
    )
    preferences["datasets"]["other"] = {"order": 30, "variables": {}}
    preferences, palettes = _load(tmp_path, preferences)

    assert check_display(registry, preferences, palettes) == []


def test_default_variable_must_exist(tmp_path, registry):
    assert _check(tmp_path, registry, default_variable="nope") == [
        ("DISP-007", "default_variable is a variable of the release")
    ]


def test_type_must_agree_with_the_curated_categories(tmp_path, registry):
    niche = _variable(type="continuous", range=[0, 1])

    refusals = _check(tmp_path, registry, variables={"niche": niche})

    assert ("DISP-002", "niche: display type") in refusals


def test_continuous_variable_needs_a_range(tmp_path, registry):
    assert _check(tmp_path, registry, variables={"v": _variable(range=None)}) == [
        ("DISP-002", "v: range of a continuous variable")
    ]


def test_categorical_variable_needs_a_set_with_enough_colours(tmp_path, registry):
    palettes = {**PALETTES, "one": {"kind": "set", "colors": ["#000000"]}}

    refusals = _check(
        tmp_path,
        registry,
        variables={"niche": {"order": 20, "palette": "one"}},
        palettes=palettes,
    )

    assert refusals == [("DISP-010", "niche: colours for every category")]


def test_labels_must_name_declared_categories(tmp_path, registry):
    refusals = _check(
        tmp_path,
        registry,
        variables={
            "niche": {"order": 20, "palette": "pair", "labels": {2: "Elsewhere"}}
        },
    )

    assert refusals == [("DISP-008", "niche: labels name declared categories")]
