from datetime import date, timedelta

import numpy as np
from collections import namedtuple
from functools import total_ordering
from pathlib import Path

import yaml

from app.exceptions import DatasetNotFoundError, VariableNotFoundError, TimeRangeContainmentError, TimeRangeInvalid
from app.settings import settings
from enum import Enum
from pydantic import BaseModel, Field, root_validator
from typing import Dict, Set, Union, Literal
from dateutil import relativedelta

from timeseries.app.exceptions import TimeRangeInvalid


class Resolution(str, Enum):
    month = 'month'
    year = 'year'


class BandRange(namedtuple('BandRange', ['gte', 'lte'])):
    """
    A range describing what bands of a raster to read

    Raster bands are one-indexed so gte should be at least one
    """
    __slots__ = ()

    def intersect(self, desired_br: 'BandRange') -> 'BandRange':
        return self.__class__(
            gte=max(self.gte, desired_br.gte),
            lte=min(self.lte, desired_br.lte))

    def to_numpy_pair(self):
        return np.array([self.gte, self.lte])

    @classmethod
    def from_numpy_pair(cls, xs, resolution: Resolution):
        return cls(gte=int(xs[0]), lte=int(xs[1]), resolution=resolution)

    def _get_range(self):
        return range(self.gte, self.lte + 1)

    def __iter__(self):
        return iter(self._get_range())

    def __len__(self):
        return len(self._get_range())


class YearRange(BaseModel):
    gte: int = Field(..., ge=0, le=2100)
    lte: int = Field(..., ge=0, le=2100)

    @root_validator
    def check_gte_less_than_lte(cls, values):
        gte, lte = values.get('gte'), values.get('lte')
        if gte > lte:
            raise TimeRangeInvalid()
        return values

    def contains(self, ymr: 'YearRange') -> bool:
        return self.gte <= ymr.gte and self.lte >= ymr.lte

    def find_band_range(self, yr: 'YearRange') -> BandRange:
        if not self.contains(yr):
            raise TimeRangeContainmentError(f'{self} does not contain {yr}')
        # raster bands are one indexed hence the plus 1
        return BandRange(yr.gte - self.gte + 1, yr.lte - self.gte + 1)

    def translate_band_range(self, br: BandRange) -> 'YearRange':
        return self.__class__(gte=self.gte + br.gte - 1, lte=self.gte + br.lte - 1)

    class Config:
        schema_extra = {
            "example": {
                "gte": 1985,
                "lte": 2010
            }
        }


@total_ordering
class YearMonth(BaseModel):
    year: int = Field(..., ge=0, le=2100)
    month: int = Field(..., ge=1, le=12)

    @classmethod
    def from_index(cls, index: int) -> 'YearMonth':
        year = index // 12
        month = index % 12 + 1
        return cls.construct(year=year, month=month)

    def to_months_since_0ce(self):
        # assumes there are not any BCE dates (0 is 0000-01 CE)
        return self.year*12 + self.month - 1

    def __eq__(self, other: 'YearMonth'):
        return self.year == other.year and self.month == other.month

    def __lt__(self, other: 'YearMonth'):
        return (self.year, self.month) < (other.year, other.month)

    class Config:
        schema_extra = {
            "example": {
                "year": 1985,
                "month": 5
            }
        }


class YearMonthRange(BaseModel):
    gte: YearMonth
    lte: YearMonth

    @root_validator
    def check_gte_less_than_lte(cls, values):
        gte, lte = values.get('gte'), values.get('lte')
        if gte >= lte:
            raise TimeRangeInvalid()
        return values

    def contains(self, ymr: 'YearMonthRange') -> bool:
        return self.gte <= ymr.gte and self.lte >= ymr.lte

    def find_band_range(self, ymr: 'YearMonthRange'):
        if not self.contains(ymr):
            raise TimeRangeContainmentError(f'{self} does not contain {ymr}')
        months_offset = self.gte.to_months_since_0ce()
        # raster bands are one indexed hence the plus one
        return BandRange(
            gte=ymr.gte.to_months_since_0ce() - months_offset + 1,
            lte=ymr.lte.to_months_since_0ce() - months_offset + 1
        )

    def translate_band_range(self, br: BandRange) -> 'YearMonthRange':
        months_offset = self.gte.to_months_since_0ce()
        # convert from a one indexed raster band index to
        # months since 0000-01 ce zero based index
        gte = br.gte - 1
        lte = br.lte - 1
        return self.__class__(
            gte=YearMonth.from_index(months_offset + gte),
            lte=YearMonth.from_index(months_offset + lte)
        )

    class Config:
        schema_extra = {
            "example": {
                "gte": YearMonth.Config.schema_extra["example"],
                "lte": {
                    "year": 2010,
                    "month": 8
                }
            }
        }


class TimeRange(BaseModel):
    gte: date
    lte: date

    @root_validator
    def check_time_range_valid(cls, values):
        gte, lte = values.get('gte'), values.get('lte')
        if gte > lte:
            raise TimeRangeInvalid()
        return values


class Repo(BaseModel):
    time_range: TimeRange
    variables: Set[str]
    resolution: Resolution


class DatasetMeta:
    def __init__(self, p: Path, time_range: TimeRange, resolution: Resolution):
        """
        :param p: path to the dataset
        :param time_range: span of time (and resolution) covered by the dataset
        """
        self.time_range = time_range
        self.p = p
        self.resolution = resolution

    def find_band_range(self, time_range: TimeRange) -> BandRange:
        """Translates time ranges from the metadata and request into a band range

        A band range is a linear index into the bands of a raster file
        """
        dataset_time_range = self.time_range
        if self.resolution == Resolution.month:
            min_offset = dataset_time_range.gte.year * 12 + (dataset_time_range.gte.month - 1) + 1
            max_offset = dataset_time_range.lte.year * 12 + (dataset_time_range.lte.month - 1) + 1
            min_index = time_range.gte.year * 12 + (time_range.gte.month - 1) + 1 - min_offset
            min_index = max(min_index, 1)
            max_index = time_range.lte.year * 12 + (time_range.lte.month - 1) + 1 - min_offset
            max_index = min(max_index, max_offset - min_offset)
        else:
            min_offset = dataset_time_range.gte.year + 1
            max_offset = dataset_time_range.lte.year + 1
            min_index = time_range.gte.year - min_offset
            min_index = max(min_index, 1)
            max_index = time_range.lte.year - min_offset
            max_index = min(max_index, max_offset - min_offset)
        return BandRange(gte=min_index, lte=max_index, resolution=self.resolution)

    def translate_band_range(self, br: BandRange) -> 'TimeRange':
        if self.resolution == Resolution.month:
            return TimeRange(
                gte=self.time_range.gte + relativedelta(months=br.gte - 1),
                lte=self.time_range.lte + relativedelta(months=br.lte - 1))
        elif self.resolution == Resolution.year:
            return TimeRange(
                gte=self.time_range.gte + relativedelta(years=br.gte - 1),
                lte=self.time_range.lte + relativedelta(years=br.lte - 1))
        else:
            raise ValueError(f'{self.resolution} is not valid. must be either year or month')


class DatasetRepo(BaseModel):
    repos: Dict[str, Repo]

    def _get_dataset(self, dataset_id: str):
        dataset = self.repos.get(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(f'Dataset {dataset_id} not found')
        return dataset

    def get_dataset_variables(self, dataset_id: str):
        dataset = self._get_dataset(dataset_id)
        return dataset.variables

    def get_dataset_meta(self, dataset_id: str, variable_id: str):
        dataset = self._get_dataset(dataset_id)

        if variable_id in dataset.variables:
            resolution = dataset.resolution
            time_range = dataset.time_range
        else:
            raise VariableNotFoundError(f'Variable {variable_id} not found in dataset {dataset_id}')

        p = settings.get_dataset_path(dataset_id=dataset_id, variable_id=variable_id)
        return DatasetMeta(p=p, time_range=time_range, resolution=resolution)


with settings.metadata_path.open() as f:
    dataset_repo = DatasetRepo(**yaml.safe_load(f))