"""Versioned internal extraction state; public status uses JobStatusResponse."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.registry.models import Identifier, Precision, ReleaseId, Sha256Hex
from app.schemas.timeseries import JobStatus, TimeseriesResponse
from app.vendor.timeaxis import parse_key

if TYPE_CHECKING:
    from app.registry.compose import ServedRelease


class InternalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class ExtractionProvenance(InternalModel):
    dataset_id: Identifier
    variable_id: Identifier
    release_id: ReleaseId
    declaration_digest: Sha256Hex
    precision: Precision

    @classmethod
    def from_release(cls, release: ServedRelease, variable_id: str):
        return cls(
            dataset_id=release.dataset_id,
            variable_id=variable_id,
            release_id=release.overview.release_id,
            declaration_digest=release.overview.declaration_digest,
            precision=release.axis.precision,
        )


class BaseSeries(InternalModel):
    timesteps: list[str] = Field(min_length=1)
    mean: list[float | None]
    median: list[float | None]

    @model_validator(mode="after")
    def matching_lengths(self):
        if len(self.mean) != len(self.timesteps) or len(self.median) != len(
            self.timesteps
        ):
            raise ValueError("Each base statistic needs one value per timestep.")
        return self


class ExtractionJobRecord(InternalModel):
    schema_version: Literal["1.0.0"]
    provenance: ExtractionProvenance
    status: JobStatus
    result: TimeseriesResponse | None = None
    base_series: BaseSeries | None = None
    error: str | None = None

    @model_validator(mode="after")
    def valid_success(self):
        if self.status == JobStatus.success:
            if self.result is None or self.base_series is None:
                raise ValueError("A successful job needs a result and base series.")
            if (self.result.dataset_id, self.result.variable_id) != (
                self.provenance.dataset_id,
                self.provenance.variable_id,
            ):
                raise ValueError(
                    "The result must identify the source dataset and variable."
                )
            instants = [
                parse_key(key, self.provenance.precision)
                for key in self.base_series.timesteps
            ]
            if any(a >= b for a, b in zip(instants, instants[1:])):
                raise ValueError("Stored timesteps must be strictly increasing.")
        return self

    def stored(self) -> dict:
        # Omit absent job fields only: null statistics inside a result are
        # required parts of the public response and must survive storage.
        return {
            key: value
            for key, value in self.model_dump(mode="json").items()
            if value is not None
        }
