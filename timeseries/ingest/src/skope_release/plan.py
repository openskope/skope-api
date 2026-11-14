"""The two states of the typed observation model (Section 11).

`ValidatedBuildPlan` is frozen after complete source preflight and is the only
input the COG writer accepts. `FinalObservation` is frozen after every output
COG has been reopened and inspected, and is the only input STAC and manifest
serialization accept (OBS-012, OBS-013). Neither can be constructed outside the
module that is allowed to issue it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Curated, CuratedVariable, Encoding, SourceManifest
from .timeaxis import Axis, ChunkLayout

_PLAN_ISSUER = object()
_OBSERVATION_ISSUER = object()


class StateError(TypeError):
    """A stage received a state it is not allowed to consume (OBS-013)."""


@dataclass(frozen=True)
class Grid:
    crs_wkt: str
    crs_code: str  # e.g. "EPSG:4269"
    width: int
    height: int
    geotransform: tuple[float, float, float, float, float, float]  # GDAL order
    area_or_point: str

    @property
    def affine(self) -> tuple[float, float, float, float, float, float]:
        """`proj:transform` order: [a, b, c, d, e, f] (Projection 2.0.0)."""
        gt = self.geotransform
        return (gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])

    @property
    def shape(self) -> tuple[int, int]:
        """`proj:shape` order: [rows, columns]."""
        return (self.height, self.width)


@dataclass(frozen=True)
class Producer:
    name: str
    version: str
    revision: str


@dataclass(frozen=True)
class PlannedVariable:
    id: str
    curated: CuratedVariable
    encoding: Encoding
    source_uri: str | None  # None for a prepared source (META-013)
    source_read_path: str  # where GDAL reads it; never part of the declaration
    source_checksum: str
    source_band: int | None  # static datasets
    source_band_count: int
    source_nodata: int | float | None


@dataclass(frozen=True)
class ValidatedBuildPlan:
    release_id: str
    curated: Curated
    resolved_manifest: SourceManifest
    declaration_digest: str
    grid: Grid
    axis: Axis | None
    layout: ChunkLayout | None
    variables: tuple[PlannedVariable, ...]
    producer: Producer
    empty_timesteps: frozenset[str]  # declared, and verified against every source (OBS-007)
    prepare_scripts: dict[str, str]  # variable ID -> SHA-256 of its prepare script (MAN-002)
    _issuer: object

    def __post_init__(self):
        if self._issuer is not _PLAN_ISSUER:
            raise StateError("a ValidatedBuildPlan is issued only by source preflight (OBS-012)")

    @property
    def dataset_id(self) -> str:
        return self.curated.id

    @property
    def temporal(self) -> bool:
        return self.curated.profile == "TemporalCubeDataset"

    @property
    def created(self) -> str:
        return self.resolved_manifest.release.created


@dataclass(frozen=True)
class BandObservation:
    """A declared empty band (OBS-007) has valid_percent 0 and no other statistics."""

    name: str
    minimum: float | None
    maximum: float | None
    mean: float | None
    stddev: float | None
    valid_percent: float

    @property
    def statistics(self) -> dict[str, float]:
        """The STAC band `statistics` object (STAC-008)."""
        values = {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "stddev": self.stddev,
            "valid_percent": self.valid_percent,
        }
        return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True)
class AssetObservation:
    variable_id: str
    chunk: int | None
    path: str  # release-relative
    size: int
    checksum: str  # multihash
    data_type: str
    nodata: int | float | None
    scale: float
    offset: float
    unit: str
    crs_code: str
    shape: tuple[int, int]
    transform: tuple[float, float, float, float, float, float]  # affine order
    bands: tuple[BandObservation, ...]
    blocksize: int
    interleave: str
    compression: str
    overview_count: int
    overview_resampling: str


@dataclass(frozen=True)
class FinalObservation:
    release_id: str
    declaration_digest: str
    created: str
    curated: Curated
    resolved_manifest: SourceManifest
    producer: Producer
    gdal_version: str
    crs_code: str
    shape: tuple[int, int]
    transform: tuple[float, float, float, float, float, float]
    bbox_wgs84: tuple[float, float, float, float]
    axis: Axis | None
    layout: ChunkLayout | None
    assets: tuple[AssetObservation, ...]
    prepare_scripts: dict[str, str]  # variable ID -> SHA-256 of its prepare script (MAN-002)
    _issuer: object

    def __post_init__(self):
        if self._issuer is not _OBSERVATION_ISSUER:
            raise StateError("a FinalObservation is issued only by byte inspection (OBS-012)")

    @property
    def dataset_id(self) -> str:
        return self.curated.id

    @property
    def temporal(self) -> bool:
        return self.curated.profile == "TemporalCubeDataset"

    def asset(self, variable_id: str, chunk: int | None) -> AssetObservation:
        for asset in self.assets:
            if asset.variable_id == variable_id and asset.chunk == chunk:
                return asset
        raise KeyError((variable_id, chunk))


def require_plan(value: object) -> ValidatedBuildPlan:
    if not isinstance(value, ValidatedBuildPlan):
        raise StateError(f"the COG writer consumes only a ValidatedBuildPlan, not {type(value).__name__} (OBS-013)")
    return value


def require_observation(value: object) -> FinalObservation:
    if not isinstance(value, FinalObservation):
        raise StateError(f"serializers consume only a FinalObservation, not {type(value).__name__} (OBS-013)")
    return value
