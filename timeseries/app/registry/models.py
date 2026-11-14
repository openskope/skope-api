"""Typed models for the documents the API reads at startup.

- `Pin`: `deploy/releases/<environment>.yml`, the release set (PIN-001).
- `ReleaseManifest`: a release's `release-manifest.json` (release spec Section 13).
- `Overview`: a release's `overview.yml` (release spec Section 15.2).

The release documents are written by the release build in
`timeseries/ingest/src/skope_release/models.py`; these models mirror the
fields of that schema version. Unknown fields are errors, so a document from a
newer schema fails at startup instead of being half read.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

IDENTIFIER_PATTERN = r"^[a-z][a-z0-9_]*$"
TIMESTEP_PATTERN = r"^\d{4}(-\d{2}(-\d{2}(T\d{2}:\d{2}:\d{2}Z)?)?)?$"
RELEASE_ID_PATTERN = r"^[a-z][a-z0-9_]*-r-\d{4}\.\d{2}\.\d{2}(-([2-9]|[1-9]\d+))?$"

Identifier = Annotated[str, StringConstraints(pattern=IDENTIFIER_PATTERN)]
TimestepKey = Annotated[str, StringConstraints(pattern=TIMESTEP_PATTERN)]
ReleaseId = Annotated[str, StringConstraints(pattern=RELEASE_ID_PATTERN)]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Multihash = Annotated[str, StringConstraints(pattern=r"^1220[0-9a-f]{64}$")]
NonEmpty = Annotated[str, StringConstraints(min_length=1)]

Profile = Literal["TemporalCubeDataset", "StaticRasterDataset"]
Precision = Literal["year", "month", "day", "datetime"]
DataType = Literal[
    "uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"
]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Pin file (PIN-001)


class PinnedRelease(Strict):
    dataset: Identifier
    release_id: ReleaseId
    declaration_digest: Sha256Hex
    manifest_sha256: Sha256Hex


class Pin(Strict):
    release_root: NonEmpty
    releases: list[PinnedRelease] = Field(min_length=1)


# ---------------------------------------------------------------------------
# release-manifest.json


class ManifestDataset(Strict):
    id: Identifier
    version: NonEmpty
    stac_entrypoint: Literal["collection.json"]


class ManifestProducer(Strict):
    name: NonEmpty
    version: NonEmpty
    revision: NonEmpty


class ManifestDeclaration(Strict):
    identity_profile: Literal["openskope-release-declaration-v1"]
    digest_algorithm: Literal["sha256"]
    digest: Sha256Hex


class ManifestInput(Strict):
    href: NonEmpty
    checksum: Multihash


class ManifestPrepare(Strict):
    script: NonEmpty
    script_sha256: Sha256Hex
    inputs: list[ManifestInput] = Field(min_length=1)


class ManifestSource(Strict):
    id: Identifier
    href: NonEmpty | None = None
    prepare: ManifestPrepare | None = None
    checksum: Multihash

    @model_validator(mode="after")
    def _one_origin(self):
        if (self.href is None) == (self.prepare is None):
            raise ValueError(
                "a source records exactly one of `href` or `prepare` (MAN-002)"
            )
        return self


class ManifestFile(Strict):
    path: NonEmpty
    roles: list[NonEmpty] = Field(min_length=1)
    size: int = Field(ge=0)
    checksum: Multihash


class ReleaseManifest(Strict):
    schema_version: Literal["1.0.0"]
    release_id: ReleaseId
    created: NonEmpty
    status: Literal["complete"]
    dataset: ManifestDataset
    producer: ManifestProducer
    declaration: ManifestDeclaration
    sources: list[ManifestSource] = Field(min_length=1)
    files: list[ManifestFile] = Field(min_length=1)


# ---------------------------------------------------------------------------
# overview.yml


class OverviewGrid(Strict):
    code: NonEmpty
    shape: tuple[int, int]
    transform: tuple[float, float, float, float, float, float]


class OverviewExtent(Strict):
    bbox: tuple[float, float, float, float]


class _OverviewTimeBase(Strict):
    count: int = Field(ge=1)
    chunk_size: int = Field(ge=1)
    precision: Precision
    calendar: Literal["proleptic_gregorian"]
    timestep_meaning: Literal["aggregation_period", "instant"]
    endpoint_inclusion: Literal["inclusive"]


class OverviewTimeRegular(_OverviewTimeBase):
    kind: Literal["regular"]
    origin: TimestepKey
    step: NonEmpty


class OverviewTimeEnumerated(_OverviewTimeBase):
    kind: Literal["enumerated"]
    values: list[TimestepKey] = Field(min_length=1)


OverviewTime = Annotated[
    Union[OverviewTimeRegular, OverviewTimeEnumerated], Field(discriminator="kind")
]


class OverviewProvider(Strict):
    name: NonEmpty
    roles: list[Literal["producer", "licensor", "processor", "host"]]
    url: str | None = None


class OverviewPublication(Strict):
    citation: NonEmpty
    doi: str | None = None


class OverviewUncertainty(Strict):
    summary: NonEmpty
    methodology_href: str | None


class OverviewLink(Strict):
    rel: NonEmpty
    href: NonEmpty
    title: str | None = None


class OverviewDataset(Strict):
    id: Identifier
    version: NonEmpty
    title: NonEmpty
    description: NonEmpty
    profile: Profile
    license: NonEmpty
    region_name: NonEmpty
    extent: OverviewExtent
    grid: OverviewGrid
    time: OverviewTime | None = None
    providers: list[OverviewProvider]
    citation: str | None = None
    doi: str | None = None
    publications: list[OverviewPublication] = Field(default_factory=list)
    lineage: str | None = None
    uncertainty: OverviewUncertainty | None = None
    links: list[OverviewLink] = Field(default_factory=list)


class OverviewVariable(Strict):
    title: NonEmpty
    description: NonEmpty
    unit: NonEmpty
    category: str | None = None
    categories: dict[int, NonEmpty] | None = None
    data_type: DataType
    nodata: int | float | None
    scale: float
    offset: float
    asset_href: str | None = None
    band_name: str | None = None


class Overview(Strict):
    schema_version: Literal["1.0.0"]
    release_id: ReleaseId
    declaration_digest: Sha256Hex
    dataset: OverviewDataset
    variables: dict[Identifier, OverviewVariable] = Field(min_length=1)

    @model_validator(mode="after")
    def _time_matches_profile(self):
        temporal = self.dataset.profile == "TemporalCubeDataset"
        if temporal != (self.dataset.time is not None):
            raise ValueError(
                "`time` is required for a temporal dataset and absent for a static one"
            )
        return self
