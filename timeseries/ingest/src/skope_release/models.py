"""Typed models for every document the release workflow reads or writes.

Each model is strict and rejects unknown fields, so a schema violation fails
at parse time. The committed JSON Schemas in `schemas/` are generated from
these models (Section 20.3).

Documents:
- `Curated`: a dataset's `curated.yml` (Section 7).
- `SourceManifest`: a dataset's `source-manifest.yml` (META-004).
- `Ledger`: a dataset's `releases.yml` of assigned release IDs (Section 20.3).
- `ReleaseManifest`: `release-manifest.json` (Section 13).
- `Overview`: `overview.yml` (Section 15.2).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Annotated, Literal, Union

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

# ---------------------------------------------------------------------------
# Shared field types

IDENTIFIER_PATTERN = r"^[a-z][a-z0-9_]*$"  # ORG-007
TIMESTEP_PATTERN = r"^\d{4}(-\d{2}(-\d{2}(T\d{2}:\d{2}:\d{2}Z)?)?)?$"  # Section 5.3
RFC3339_UTC_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
MULTIHASH_SHA256_PATTERN = r"^1220[0-9a-f]{64}$"  # STAC-009, File Info 2.1.0
SHA256_HEX_PATTERN = r"^[0-9a-f]{64}$"
DOI_PATTERN = r"^10\.\d{4,9}/\S+$"
RELEASE_ID_PATTERN = r"^[a-z][a-z0-9_]*-r-\d{4}\.\d{2}\.\d{2}(-([2-9]|[1-9]\d+))?$"  # REL-006
PREPARE_SCRIPT_PATTERN = r"^scripts/[a-z][a-z0-9_]*\.py$"  # META-013: one file in scripts/

Identifier = Annotated[str, StringConstraints(pattern=IDENTIFIER_PATTERN)]
TimestepKey = Annotated[str, StringConstraints(pattern=TIMESTEP_PATTERN)]
Rfc3339Utc = Annotated[str, StringConstraints(pattern=RFC3339_UTC_PATTERN)]
Multihash = Annotated[str, StringConstraints(pattern=MULTIHASH_SHA256_PATTERN)]
Sha256Hex = Annotated[str, StringConstraints(pattern=SHA256_HEX_PATTERN)]
Doi = Annotated[str, StringConstraints(pattern=DOI_PATTERN)]
ReleaseId = Annotated[str, StringConstraints(pattern=RELEASE_ID_PATTERN)]
NonEmpty = Annotated[str, StringConstraints(min_length=1)]
PrepareScript = Annotated[str, StringConstraints(pattern=PREPARE_SCRIPT_PATTERN)]


def _https(value: str) -> str:
    if not re.match(r"^https://[^\s/]+(/\S*)?$", value):
        raise ValueError("must be an absolute HTTPS URL (REL-002)")
    return value


HttpsUrl = Annotated[str, AfterValidator(_https)]


def _orcid(value: str) -> str:
    """Canonical ORCID URI with a valid ISO 7064 11,2 check digit (META-010)."""
    match = re.fullmatch(r"https://orcid\.org/(\d{4})-(\d{4})-(\d{4})-(\d{3}[\dX])", value)
    if not match:
        raise ValueError("must be a canonical ORCID URI, https://orcid.org/0000-0000-0000-0000")
    digits = "".join(match.groups())
    total = 0
    for ch in digits[:-1]:
        total = (total + int(ch)) * 2
    check = (12 - total % 11) % 11
    if digits[-1] != ("X" if check == 10 else str(check)):
        raise ValueError("ORCID check digit is invalid")
    return value


def _ror(value: str) -> str:
    if not re.fullmatch(r"https://ror\.org/0[a-hj-km-np-tv-z0-9]{6}\d{2}", value):
        raise ValueError("must be a canonical ROR URI, https://ror.org/0xxxxxxNN")
    return value


Orcid = Annotated[str, AfterValidator(_orcid)]
Ror = Annotated[str, AfterValidator(_ror)]


def parse_rfc3339_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


# ---------------------------------------------------------------------------
# curated.yml (Section 7)

ProviderRole = Literal["producer", "licensor", "processor", "host"]
Profile = Literal["TemporalCubeDataset", "StaticRasterDataset"]
Precision = Literal["year", "month", "day", "datetime"]


class Provider(Strict):
    name: NonEmpty
    roles: list[ProviderRole] = Field(min_length=1)
    url: HttpsUrl | None = None
    ror: Ror | None = None
    description: str | None = None

    @model_validator(mode="after")
    def _unique_roles(self):
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("provider roles must be unique")
        return self


class Person(Strict):
    """A person explicitly selected for this dataset (META-010)."""

    name: NonEmpty
    orcid: Orcid | None = None


class Publication(Strict):
    citation: NonEmpty
    doi: Doi | None = None


class Uncertainty(Strict):
    summary: NonEmpty
    methodology_href: HttpsUrl | None = None


class Link(Strict):
    rel: NonEmpty
    href: HttpsUrl
    title: str | None = None
    type: str | None = None


class CuratedTemporal(Strict):
    """Reviewed temporal semantics (META-002, META-008)."""

    calendar: Literal["proleptic_gregorian"]
    precision: Precision
    origin: TimestepKey
    end: TimestepKey
    step: str | None = None
    values: list[TimestepKey] | None = None
    endpoint_inclusion: Literal["inclusive"]
    timestep_meaning: Literal["aggregation_period", "instant"]
    description: NonEmpty

    @model_validator(mode="after")
    def _one_axis_form(self):
        if (self.step is None) == (self.values is None):
            raise ValueError("declare exactly one of `step` (regular axis) or `values` (enumerated axis)")
        if self.values is not None:
            if not self.values or self.values[0] != self.origin or self.values[-1] != self.end:
                raise ValueError("`values` must start at `origin` and end at `end`")
        return self


class EmptyTimesteps(Strict):
    """An inclusive run of timesteps with no valid pixels in any variable (OBS-007)."""

    first: TimestepKey
    last: TimestepKey  # equal to `first` for a single timestep
    reason: NonEmpty


class CuratedVariable(Strict):
    id: Identifier
    title: NonEmpty
    description: NonEmpty
    unit: NonEmpty  # "unitless" when the quantity has no unit (META-002)
    category: NonEmpty | None = None
    categories: dict[int, NonEmpty] | None = None  # encoded value -> meaning
    valid_range: tuple[float, float] | None = None  # review evidence only; not published (META-007)

    @model_validator(mode="after")
    def _range_order(self):
        if self.valid_range is not None and not self.valid_range[0] < self.valid_range[1]:
            raise ValueError("valid_range must be [lower, upper] with lower < upper")
        return self


class Curated(Strict):
    schema_version: Literal["0.1.0"]
    id: Identifier
    version: NonEmpty
    profile: Profile
    title: NonEmpty
    description: NonEmpty
    license: NonEmpty  # SPDX identifier, or "other" with license_href
    license_href: HttpsUrl | None = None
    region_name: NonEmpty
    providers: list[Provider] = Field(min_length=1)
    people: list[Person] = Field(default_factory=list)
    citation: NonEmpty | None = None
    doi: Doi | None = None
    publications: list[Publication] = Field(default_factory=list)
    contact_href: HttpsUrl | None = None
    lineage: NonEmpty | None = None
    uncertainty: Uncertainty | None = None
    links: list[Link] = Field(default_factory=list)
    temporal: CuratedTemporal | None = None
    temporal_extent: tuple[Rfc3339Utc | None, Rfc3339Utc | None] | None = None
    # Ordered, non-overlapping; checked against the axis and the bytes in preflight.
    empty_timesteps: list[EmptyTimesteps] = Field(default_factory=list)
    variables: list[CuratedVariable] = Field(min_length=1)

    @model_validator(mode="after")
    def _profile_rules(self):
        if self.profile == "TemporalCubeDataset":
            if self.temporal is None:
                raise ValueError("a TemporalCubeDataset requires `temporal` (META-002)")
            if self.temporal_extent is not None:
                raise ValueError("`temporal_extent` is for a StaticRasterDataset; use `temporal`")
        else:
            if self.temporal is not None:
                raise ValueError("a StaticRasterDataset has no temporal axis (ORG-001, API-006)")
            if self.empty_timesteps:
                raise ValueError("`empty_timesteps` is for a TemporalCubeDataset; a static band is never empty (OBS-007)")
        if self.license == "other" and self.license_href is None:
            raise ValueError("license `other` requires `license_href`")
        ids = [v.id for v in self.variables]
        if len(ids) != len(set(ids)):
            raise ValueError("variable identifiers must be unique (VAL-003)")
        orcids = {}
        for person in self.people:
            if person.orcid and orcids.setdefault(person.orcid, person.name) != person.name:
                raise ValueError(f"ORCID {person.orcid} is associated with conflicting names (META-010)")
        return self


# ---------------------------------------------------------------------------
# source-manifest.yml (META-004)

DataType = Literal["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"]
Resampling = Literal["AVERAGE", "MODE", "NEAREST", "BILINEAR", "CUBIC", "LANCZOS"]


class CogOptions(Strict):
    """COG creation options that are part of the declaration (Section 12.1)."""

    blocksize: Literal[128, 256, 512]
    interleave: Literal["BAND", "TILE", "PIXEL"]
    compress: Literal["ZSTD", "DEFLATE", "LZW"]
    level: int | None = None
    bigtiff: Literal["IF_SAFER", "IF_NEEDED", "YES", "NO"]
    sparse_ok: Literal[False] = False  # COG-010


class ReleaseBlock(Strict):
    created: Rfc3339Utc  # REL-006; never read from the clock
    chunk_size: int | None = Field(default=None, ge=1)  # omitted for a static dataset (Section 20.3)
    cog: CogOptions


class Narrowing(Strict):
    """An approved per-variable datatype narrowing (COG-016, EXP-004)."""

    approved_by: NonEmpty
    evidence: NonEmpty


class Encoding(Strict):
    data_type: DataType
    nodata: int | float | None
    scale: float = 1.0
    offset: float = 0.0
    predictor: Literal[1, 2, 3]
    overview_resampling: Resampling
    narrowing: Narrowing | None = None


class PrepareInput(Strict):
    uri: NonEmpty
    checksum: Multihash | None = None


class Prepare(Strict):
    """A script that makes the source from upstream files the build cannot read (META-013)."""

    script: PrepareScript
    inputs: list[PrepareInput] = Field(min_length=1)


class Source(Strict):
    uri: NonEmpty | None = None
    prepare: Prepare | None = None
    checksum: Multihash | None = None  # of the prepared GeoTIFF for a prepared source
    # Temporal: band i carries the axis timestep equal to its description (OBS-006).
    # Static: the single band `band` carries the variable.
    bands: Literal["descriptions"] | None = None
    band: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _one_origin(self):
        if (self.uri is None) == (self.prepare is None):
            raise ValueError("declare exactly one of `uri` or `prepare` (META-013)")
        return self

    @model_validator(mode="after")
    def _band_mapping(self):
        if (self.bands is None) == (self.band is None):
            raise ValueError("declare exactly one of `bands: descriptions` (temporal) or `band: N` (static)")
        return self


class SourceVariable(Strict):
    id: Identifier
    source: Source
    encoding: Encoding


class SourceManifest(Strict):
    schema_version: Literal["0.1.0"]
    dataset: Identifier
    release: ReleaseBlock
    variables: list[SourceVariable] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self):
        ids = [v.id for v in self.variables]
        if len(ids) != len(set(ids)):
            raise ValueError("variable identifiers must be unique (OBS-001)")
        return self


# ---------------------------------------------------------------------------
# releases.yml ledger (Section 20.3)


class LedgerEntry(Strict):
    release_id: ReleaseId
    declaration_digest: Sha256Hex
    withdrawn: NonEmpty | None = None


class Ledger(Strict):
    releases: list[LedgerEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# release-manifest.json (Section 13)

FileRole = Literal["stac", "collection", "item", "data", "derived"]  # MAN-008


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
    script: PrepareScript
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
            raise ValueError("a source records exactly one of `href` or `prepare` (MAN-002)")
        return self


class ManifestFile(Strict):
    path: NonEmpty
    roles: list[FileRole] = Field(min_length=1)
    size: int = Field(ge=0)
    checksum: Multihash


class ReleaseManifest(Strict):
    schema_version: Literal["1.0.0"]
    release_id: ReleaseId
    created: Rfc3339Utc
    status: Literal["complete"]
    dataset: ManifestDataset
    producer: ManifestProducer
    declaration: ManifestDeclaration
    sources: list[ManifestSource] = Field(min_length=1)
    files: list[ManifestFile] = Field(min_length=1)


# ---------------------------------------------------------------------------
# overview.yml (Section 15.2)


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

    @model_validator(mode="after")
    def _count(self):
        if len(self.values) != self.count:
            raise ValueError("count must equal len(values) (API-002)")
        return self


OverviewTime = Annotated[Union[OverviewTimeRegular, OverviewTimeEnumerated], Field(discriminator="kind")]


class OverviewProvider(Strict):
    name: NonEmpty
    roles: list[ProviderRole]
    url: HttpsUrl | None = None


class OverviewUncertainty(Strict):
    summary: NonEmpty
    methodology_href: HttpsUrl | None


class OverviewLink(Strict):
    rel: NonEmpty
    href: HttpsUrl
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
    doi: Doi | None = None
    publications: list[Publication] = Field(default_factory=list)
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
    asset_href: str | None = None  # static datasets only (API-006)
    band_name: str | None = None


class Overview(Strict):
    schema_version: Literal["1.0.0"]
    release_id: ReleaseId
    declaration_digest: Sha256Hex
    dataset: OverviewDataset
    variables: dict[Identifier, OverviewVariable]

    @model_validator(mode="after")
    def _profile(self):
        temporal = self.dataset.profile == "TemporalCubeDataset"
        if temporal != (self.dataset.time is not None):
            raise ValueError("`time` is required for a temporal dataset and absent for a static one (API-006)")
        for vid, var in self.variables.items():
            if temporal and (var.asset_href or var.band_name):
                raise ValueError(f"{vid}: asset_href/band_name are for static datasets only")
            if not temporal and not (var.asset_href and var.band_name):
                raise ValueError(f"{vid}: a static dataset names its asset_href and band_name")
        return self


DOCUMENT_MODELS: dict[str, type[BaseModel]] = {
    "curated": Curated,
    "source-manifest": SourceManifest,
    "release-ledger": Ledger,
    "release-manifest": ReleaseManifest,
    "release-overview": Overview,
}
