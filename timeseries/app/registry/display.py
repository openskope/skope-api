"""Display preferences and palettes (release consumption spec Section 4).

`load_display` parses both files and runs the checks that need no release
data (DISP-012); CI runs them through the tests on the committed files.
`check_display` runs the checks that need the pinned overviews (DISP-013),
reported like the release checks so a failure stops the API.
"""

import math
from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StrictFloat,
    StrictInt,
    StringConstraints,
    ValidationError,
    model_validator,
)

from app.registry.compose import AppRegistry, Refusal
from app.registry.models import Identifier

HexColor = Annotated[str, StringConstraints(pattern=r"^#[0-9A-Fa-f]{6}$")]
PaletteName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
Number = Union[StrictInt, StrictFloat]
Ticks = Union[
    Annotated[StrictInt, Field(ge=2)], Annotated[list[Number], Field(min_length=2)]
]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MapCenter(Strict):
    lon: Number = Field(ge=-180, le=180)
    lat: Number = Field(ge=-90, le=90)


class MapView(Strict):
    """An override for fitting the dataset's bbox (DISP-006)."""

    center: MapCenter
    zoom: Number = Field(ge=0, le=24)


class VariableDisplay(Strict):
    order: Number
    palette: PaletteName
    type: Literal["continuous", "categorical"] | None = None
    range: tuple[Number, Number] | None = None
    ticks: Ticks | None = None
    note: str | None = None
    labels: dict[int, Annotated[str, StringConstraints(min_length=1)]] | None = None

    @model_validator(mode="after")
    def _range_and_ticks(self):
        if self.range is not None:
            lower, upper = self.range
            if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
                raise ValueError("range needs finite endpoints, lower first (DISP-003)")
            if isinstance(self.ticks, list) and not all(
                lower <= tick <= upper for tick in self.ticks
            ):
                raise ValueError(
                    "every explicit tick must lie within the range (DISP-008)"
                )
        return self


class DatasetDisplay(Strict):
    order: Number
    map_view: MapView | None = None
    default_variable: Identifier | None = None
    variables: dict[Identifier, VariableDisplay]


class Defaults(Strict):
    ticks: Ticks = 5
    nodata: Literal["transparent"] = "transparent"


class DisplayPreferences(Strict):
    defaults: Defaults = Defaults()
    datasets: dict[Identifier, DatasetDisplay]

    def variable(self, dataset_id: str, variable_id: str) -> VariableDisplay:
        return self.datasets[dataset_id].variables[variable_id]


class Palette(Strict):
    kind: Literal["ramp", "set"]
    colors: list[HexColor] = Field(min_length=1)

    @model_validator(mode="after")
    def _ramp_has_two_colours(self):
        if self.kind == "ramp" and len(self.colors) < 2:
            raise ValueError("a ramp needs at least two colours")
        return self


class Palettes(RootModel[dict[PaletteName, Palette]]):
    def __getitem__(self, name: str) -> Palette:
        return self.root[name]

    def __contains__(self, name: str) -> bool:
        return name in self.root


class DisplayFileError(ValueError):
    """The display files fail a check that needs no release data (DISP-012)."""


def load_display(
    preferences_path: Path, palettes_path: Path
) -> tuple[DisplayPreferences, Palettes]:
    try:
        preferences = DisplayPreferences.model_validate(
            yaml.safe_load(preferences_path.read_text("utf-8"))
        )
        palettes = Palettes.model_validate(
            yaml.safe_load(palettes_path.read_text("utf-8"))
        )
    except (ValidationError, yaml.YAMLError) as exc:
        raise DisplayFileError(str(exc)) from exc

    errors = []
    for dataset_id, dataset in preferences.datasets.items():
        for variable_id, variable in dataset.variables.items():
            where = f"{dataset_id}.{variable_id}"
            if variable.palette not in palettes:
                errors.append(
                    f"{where}: palette {variable.palette!r} is not defined (DISP-009)"
                )
                continue
            kind = palettes[variable.palette].kind
            if variable.type is not None and kind != _KIND[variable.type]:
                errors.append(
                    f"{where}: a {variable.type} variable needs a {_KIND[variable.type]} palette, "
                    f"{variable.palette!r} is a {kind} (DISP-010)"
                )
            if variable.type == "continuous" and variable.range is None:
                errors.append(
                    f"{where}: a continuous variable needs a range (DISP-002)"
                )
    if errors:
        raise DisplayFileError("; ".join(errors))
    return preferences, palettes


_KIND = {"continuous": "ramp", "categorical": "set"}


def check_display(
    registry: AppRegistry, preferences: DisplayPreferences, palettes: Palettes
) -> list[Refusal]:
    """Checks that need the pinned overviews (DISP-013)."""
    refusals = []
    for dataset_id, release in registry.releases.items():

        def refuse(requirement, check, expected, observed):
            refusals.append(
                Refusal(
                    requirement,
                    dataset_id,
                    release.overview.release_id,
                    check,
                    str(expected),
                    str(observed),
                )
            )

        dataset = preferences.datasets.get(dataset_id)
        if dataset is None:
            refuse("DISP-001", "dataset display entry", dataset_id, "missing")
            continue
        variables = release.overview.variables
        if (
            dataset.default_variable is not None
            and dataset.default_variable not in variables
        ):
            refuse(
                "DISP-007",
                "default_variable is a variable of the release",
                sorted(variables),
                dataset.default_variable,
            )
        for variable_id, variable in variables.items():
            display = dataset.variables.get(variable_id)
            if display is None:
                refuse(
                    "DISP-001",
                    f"{variable_id}: variable display entry",
                    variable_id,
                    "missing",
                )
                continue
            categorical = variable.categories is not None
            kind = "categorical" if categorical else "continuous"
            if display.type is not None and display.type != kind:
                refuse("DISP-002", f"{variable_id}: display type", kind, display.type)
            if not categorical and display.range is None:
                refuse(
                    "DISP-002",
                    f"{variable_id}: range of a continuous variable",
                    "[lower, upper]",
                    "missing",
                )
            palette = palettes[display.palette]
            if palette.kind != _KIND[kind]:
                refuse(
                    "DISP-010",
                    f"{variable_id}: palette kind",
                    _KIND[kind],
                    palette.kind,
                )
            if categorical:
                if len(palette.colors) < len(variable.categories):
                    refuse(
                        "DISP-010",
                        f"{variable_id}: colours for every category",
                        len(variable.categories),
                        len(palette.colors),
                    )
                undeclared = set(display.labels or {}) - set(variable.categories)
                if undeclared:
                    refuse(
                        "DISP-008",
                        f"{variable_id}: labels name declared categories",
                        sorted(variable.categories),
                        sorted(undeclared),
                    )
    return refusals
