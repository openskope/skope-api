"""The `/metadata` response, schema 1.0.0 (release consumption spec Section 5).

A deterministic projection of the app registry and the display files
(PROTO-001): datasets and variables sorted by their display `order`, then by
identifier (DISP-005). Rendering fields come only from the display entry
(PROTO-003); there are no `min`/`max` fields.

The response's JSON Schema (PROTO-002) is generated from `MetadataResponse`
and committed as `schemas/metadata-1.0.0.schema.json`. Regenerate it after a
deliberate change to the response with:

    python -m app.registry.metadata
"""

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from pyproj import CRS, Geod

from app.registry.compose import AppRegistry, ServedRelease
from app.registry.display import DisplayPreferences, Palettes
from app.registry.models import (
    OverviewGrid,
    OverviewLink,
    OverviewProvider,
    OverviewPublication,
    OverviewTimeRegular,
    OverviewUncertainty,
)

SCHEMA_VERSION = "1.0.0"
SCHEMA_PATH = (
    Path(__file__).parent / "schemas" / f"metadata-{SCHEMA_VERSION}.schema.json"
)


class Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Display(Out):
    type: Literal["continuous", "categorical"]
    palette: str
    colors: list[str]
    range: tuple[float, float] | None
    ticks: int | list[float]
    note: str | None


class Variable(Out):
    id: str
    title: str
    description: str
    unit: str
    category: str | None
    categories: dict[str, str] | None
    labels: dict[str, str] | None
    order: int | float
    display: Display


class Time(Out):
    kind: Literal["regular", "enumerated"]
    precision: str
    count: int
    origin: str
    end: str
    step: str | None
    values: list[str] | None


class MapCenter(Out):
    lon: float
    lat: float


class MapView(Out):
    center: MapCenter
    zoom: int | float


class Dataset(Out):
    id: str
    title: str
    description: str
    version: str
    license: str
    profile: str
    region_name: str
    bbox: tuple[float, float, float, float]
    resolution_label: str
    time: Time
    providers: list[OverviewProvider]
    citation: str | None
    doi: str | None
    publications: list[OverviewPublication]
    lineage: str | None
    uncertainty: OverviewUncertainty | None
    links: list[OverviewLink]
    order: int | float
    map_view: MapView | None
    default_variable: str | None
    release_id: str
    declaration_digest: str
    variables: list[Variable]


class MetadataResponse(Out):
    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    datasets: list[Dataset]


# ---------------------------------------------------------------------------
# Resolution label (PROTO-004)


def _distance(metres: float, digits: int = 2) -> str:
    """A length to `digits` significant figures, in m or km."""
    if metres >= 1000:
        return f"{float(f'{metres / 1000:.{digits}g}'):g} km"
    return f"{float(f'{metres:.{digits}g}'):g} m"


def resolution_label(
    grid: OverviewGrid, bbox: tuple[float, float, float, float]
) -> str:
    """A readable pixel size computed from the grid, e.g. "30 arc-second (~800 m)"."""
    width, height = abs(grid.transform[0]), abs(grid.transform[4])
    crs = CRS.from_user_input(grid.code)
    if crs.is_geographic:
        x, y = f"{width * 3600:.4g}", f"{height * 3600:.4g}"
        angle = x if x == y else f"{x} × {y}"
        # Ground size at the middle of the extent: the geometric mean of the
        # pixel's east-west and north-south lengths. It changes with latitude,
        # so one significant figure is as precise as it gets.
        latitude = (bbox[1] + bbox[3]) / 2
        geod = Geod(ellps="WGS84")
        east = geod.inv(0, latitude, width, latitude)[2]
        north = geod.inv(0, latitude, 0, latitude + height)[2]
        return f"{angle} arc-second (~{_distance(math.sqrt(east * north), 1)})"
    unit = crs.axis_info[0].unit_name if crs.axis_info else ""
    if unit in {"metre", "meter"}:
        if width == height:
            return _distance(width)
        return f"{_distance(width)} × {_distance(height)}"
    return f"{width:g} × {height:g} {unit}".strip()


# ---------------------------------------------------------------------------
# Projection


def _time(release: ServedRelease) -> Time:
    time = release.overview.dataset.time
    axis = release.axis
    regular = isinstance(time, OverviewTimeRegular)
    return Time(
        kind=time.kind,
        precision=time.precision,
        count=time.count,
        origin=axis.key(0),
        end=axis.key(axis.count - 1),
        step=time.step if regular else None,
        values=None if regular else list(time.values),
    )


def _sorted(entries: dict, order_of) -> list:
    return sorted(entries, key=lambda key: (order_of(key), key))


def build_metadata(
    registry: AppRegistry, preferences: DisplayPreferences, palettes: Palettes
) -> MetadataResponse:
    datasets = []
    for dataset_id in _sorted(
        registry.releases, lambda d: preferences.datasets[d].order
    ):
        release = registry.releases[dataset_id]
        overview = release.overview
        info = overview.dataset
        shown = preferences.datasets[dataset_id]
        variables = []
        for variable_id in _sorted(
            overview.variables, lambda v: shown.variables[v].order
        ):
            variable = overview.variables[variable_id]
            display = shown.variables[variable_id]
            categorical = variable.categories is not None
            categories = labels = None
            if categorical:
                categories = {str(k): v for k, v in sorted(variable.categories.items())}
                labels = {
                    str(k): (display.labels or {}).get(k, meaning)
                    for k, meaning in sorted(variable.categories.items())
                }
            variables.append(
                Variable(
                    id=variable_id,
                    title=variable.title,
                    description=variable.description,
                    unit=variable.unit,
                    category=variable.category,
                    categories=categories,
                    labels=labels,
                    order=display.order,
                    display=Display(
                        type="categorical" if categorical else "continuous",
                        palette=display.palette,
                        colors=list(palettes[display.palette].colors),
                        range=display.range,
                        ticks=(
                            display.ticks
                            if display.ticks is not None
                            else preferences.defaults.ticks
                        ),
                        note=display.note,
                    ),
                )
            )
        map_view = None
        if shown.map_view is not None:
            map_view = MapView(
                center=MapCenter(
                    lon=shown.map_view.center.lon, lat=shown.map_view.center.lat
                ),
                zoom=shown.map_view.zoom,
            )
        datasets.append(
            Dataset(
                id=info.id,
                title=info.title,
                description=info.description,
                version=info.version,
                license=info.license,
                profile=info.profile,
                region_name=info.region_name,
                bbox=info.extent.bbox,
                resolution_label=resolution_label(info.grid, info.extent.bbox),
                time=_time(release),
                providers=info.providers,
                citation=info.citation,
                doi=info.doi,
                publications=info.publications,
                lineage=info.lineage,
                uncertainty=info.uncertainty,
                links=info.links,
                order=shown.order,
                map_view=map_view,
                default_variable=shown.default_variable,
                release_id=overview.release_id,
                declaration_digest=overview.declaration_digest,
                variables=variables,
            )
        )
    return MetadataResponse(datasets=datasets)


# ---------------------------------------------------------------------------
# JSON Schema (PROTO-002)


def metadata_json_schema() -> str:
    """The response's JSON Schema, serialized the same way every time."""
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        **MetadataResponse.model_json_schema(),
        "title": f"SKOPE /metadata response {SCHEMA_VERSION}",
    }
    return json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    SCHEMA_PATH.parent.mkdir(exist_ok=True)
    SCHEMA_PATH.write_text(metadata_json_schema())
    print(SCHEMA_PATH)
