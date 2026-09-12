from enum import Enum
from typing import List, Literal, Optional, Self, Union, Annotated
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
    ValidationInfo,
)

from app.config import get_settings
from .geometry import (
    SkopeFeatureCollectionModel,
    SkopeFeatureModel,
    SkopePointModel,
    SkopePolygonModel,
)

settings = get_settings()

# Strict ISO-8601 zero-padded pattern (YYYY-MM-DDTHH:MM:SSZ)

ISO_TIME_PATTERN = (
    r"^\d{4}(?:-\d{2}(?:-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:Z|[+-]\d{2}:\d{2})?)?)?)?$"
)


class ZonalStatistic(str, Enum):
    mean = "mean"
    median = "median"


class TimeRange(BaseModel):
    gte: str = Field(
        ..., pattern=ISO_TIME_PATTERN, description="Start time in ISO-8601 format"
    )
    lte: str = Field(
        ..., pattern=ISO_TIME_PATTERN, description="End time in ISO-8601 format"
    )

    @model_validator(mode="after")
    def check_time_range_valid(self) -> Self:
        if self.gte > self.lte:
            raise ValueError("Start date cannot be after end date.")
        return self

    model_config = ConfigDict(
        json_schema_extra={"example": {"gte": "0001-02-05", "lte": "0005-09-02"}}
    )


# ---------------------------
# Smoothers


class WindowType(str, Enum):
    centered = "centered"
    trailing = "trailing"


class NoSmoother(BaseModel):
    type: Literal["NoSmoother"] = "NoSmoother"
    model_config = ConfigDict(json_schema_extra={"example": {"type": "NoSmoother"}})


class MovingAverageSmoother(BaseModel):
    type: Literal["MovingAverageSmoother"] = "MovingAverageSmoother"
    method: WindowType
    width: int = Field(
        ...,
        description="Number of time steps from current time to use in the moving window",
        ge=1,
        le=200,
    )

    @field_validator("width")
    @classmethod
    def width_is_valid_for_window_type(cls, value: int, info: ValidationInfo):
        method = info.data.get("method")
        if method == WindowType.centered and value % 2 == 0:
            raise ValueError("Window width must be odd for centered windows")
        return value

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "type": "MovingAverageSmoother",
                "method": WindowType.centered.value,
                "width": 3,
            }
        }
    )


Smoother = Annotated[
    Union[NoSmoother, MovingAverageSmoother], Field(discriminator="type")
]

# ---------------------------
# Transforms


class ZScoreMovingInterval(BaseModel):
    """A moving Z-Score transform to the timeseries"""

    type: Literal["ZScoreMovingInterval"] = "ZScoreMovingInterval"
    width: int = Field(
        ...,
        description="Number of prior time steps to use in the moving window",
        ge=1,
        le=200,
    )
    model_config = ConfigDict(
        json_schema_extra={"example": {"type": "ZScoreMovingInterval", "width": 5}}
    )


class ZScoreFixedInterval(BaseModel):
    """A Z-Score transform to the timeseries using a fixed interval"""

    type: Literal["ZScoreFixedInterval"] = "ZScoreFixedInterval"
    time_range: Optional[TimeRange] = None
    model_config = ConfigDict(
        json_schema_extra={"example": {"type": "ZScoreFixedInterval"}}
    )


class NoTransform(BaseModel):
    """No transformation to the timeseries - return raw values"""

    type: Literal["NoTransform"] = "NoTransform"
    model_config = ConfigDict(json_schema_extra={"example": {"type": "NoTransform"}})


Transform = Annotated[
    Union[ZScoreMovingInterval, ZScoreFixedInterval, NoTransform],
    Field(discriminator="type"),
]

# ---------------------------
# Response Models


class SummaryStat(BaseModel):
    name: str
    mean: Optional[float]
    median: Optional[float]
    stdev: Optional[float]


class SeriesOptions(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[\w -]+$")
    smoother: Smoother
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "transformed",
                "smoother": {
                    "type": "MovingAverageSmoother",
                    "method": "centered",
                    "width": 3,
                },
            }
        }
    )


class Series(BaseModel):
    options: SeriesOptions
    time_range: TimeRange
    values: List[Optional[float]]


class TimeseriesResponse(BaseModel):
    dataset_id: str
    variable_id: str
    area: float = Field(
        ..., description="Area of cells in selected area in square meters"
    )
    n_cells: int = Field(..., description="Number of cells in selected area")
    summary_stats: List[SummaryStat]
    series: List[Series]
    transform: Transform
    zonal_statistic: ZonalStatistic


# ---------------------------
# Request Models


class SeriesOptionsRequest(BaseModel):
    requested_series_options: List[SeriesOptions] = Field(
        ..., min_length=1, max_length=settings.max_series_options
    )

    @field_validator("requested_series_options")
    @classmethod
    def series_names_must_be_unique(cls, value: List[SeriesOptions]):
        names = [option.name for option in value]
        if len(names) != len(set(names)):
            raise ValueError("Series option names must be unique.")
        return value


class TimeseriesRequest(SeriesOptionsRequest):
    dataset_id: str = Field(..., pattern=r"^[\w-]+$", description="Dataset ID")
    variable_id: str = Field(..., pattern=r"^[\w-]+$", description="Variable ID")
    selected_area: Union[
        SkopePointModel,
        SkopePolygonModel,
        SkopeFeatureModel,
        SkopeFeatureCollectionModel,
    ]
    zonal_statistic: ZonalStatistic
    transform: Transform
    time_range: Optional[TimeRange]
    max_processing_time: int = Field(
        settings.max_processing_time, ge=0, le=settings.max_processing_time
    )

    @field_validator("selected_area")
    @classmethod
    def selected_area_must_be_bounded(cls, value):
        value.validate_complexity(
            max_shapes=settings.max_geometry_shapes,
            max_coordinates=settings.max_geometry_coordinates,
        )
        return value


class TimeseriesAnalyzeRequest(SeriesOptionsRequest):
    extraction_id: str = Field(
        ..., description="Job ID from POST /v3/timeseries/extract"
    )
    zonal_statistic: ZonalStatistic = ZonalStatistic.mean
    transform: Transform
    time_range: Optional[TimeRange] = None
