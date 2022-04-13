from collections import namedtuple
from datetime import date
from enum import Enum
from pydantic import BaseModel, root_validator
from typing import Optional

import numpy as np

from app.exceptions import TimeRangeInvalid


class ZonalStatistic(str, Enum):
    mean = "nanmean"
    median = "nanmedian"

    def to_numpy_func(self):
        return getattr(np, self.value)


class Resolution(str, Enum):
    month = "month"
    year = "year"


class BandRange(namedtuple("BandRange", ["gte", "lte"])):
    """
    A range class describing what bands of a raster to read

    Raster bands are one-indexed so gte must be >= 1
    """

    __slots__ = ()

    def intersect(self, desired_br: "BandRange") -> "BandRange":
        return self.__class__(
            gte=max(self.gte, desired_br.gte), lte=min(self.lte, desired_br.lte)
        )

    def union(self, desired_br: "BandRange") -> "BandRange":
        return self.__class__(
            gte=min(self.gte, desired_br.gte), lte=max(self.lte, desired_br.lte)
        )

    def __add__(self, other) -> "BandRange":
        return self.__class__(gte=self.gte + other[0], lte=self.lte + other[1])

    def to_numpy_pair(self):
        return np.array([self.gte, self.lte])

    @classmethod
    def from_numpy_pair(cls, xs):
        return cls(gte=int(xs[0]), lte=int(xs[1]))

    def _as_range(self):
        return range(self.gte, self.lte + 1)

    def __iter__(self):
        return iter(self._as_range())

    def __len__(self):
        return len(self._as_range())


class TimeRange(BaseModel):
    gte: date
    lte: date

    @root_validator
    def check_time_range_valid(cls, values):
        gte, lte = values.get("gte"), values.get("lte")
        if gte > lte:
            raise TimeRangeInvalid()
        return values

    def intersect(self, tr: "TimeRange") -> "TimeRange":
        return TimeRange(gte=max(self.gte, tr.gte), lte=min(self.lte, tr.lte))

    class Config:
        schema_extra = {"example": {"gte": "0001-02-05", "lte": "0005-09-02"}}


class OptionalTimeRange(BaseModel):
    gte: Optional[date]
    lte: Optional[date]

    class Config:
        schema_extra = {"example": TimeRange.Config.schema_extra["example"]}
