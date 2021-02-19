import numpy as np
from collections import namedtuple
from functools import total_ordering
from pathlib import Path

import yaml

from app.exceptions import DatasetNotFoundError, VariableNotFoundError, TimeRangeContainmentError
from app.settings import settings
from pydantic import BaseModel, Field
from typing import Dict, Set, Union, Literal


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
    def from_numpy_pair(cls, xs):
        return cls(gte=int(xs[0]), lte=int(xs[1]))

    def __iter__(self):
        return iter(range(self.gte, self.lte + 1))


class YearRange(BaseModel):
    gte: int = Field(..., ge=0, le=2100)
    lte: int = Field(..., ge=0, le=2100)

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


class YearlyRepo(BaseModel):
    time_range: YearRange
    variables: Set[str]

    @property
    def resolution(self) -> Literal['year']:
        return 'year'


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


class MonthlyRepo(BaseModel):
    time_range: YearMonthRange
    variables: Set[str]

    @property
    def resolution(self) -> Literal['month']:
        return 'month'


TimeRange = Union[YearRange, YearMonthRange]


class DatasetMeta:
    def __init__(self, p: Path, time_range: TimeRange, resolution: Literal['year', 'month']):
        """
        :param p: path to the dataset
        :param time_range: span of time (and resolution) covered by the dataset
        """
        self.time_range = time_range
        self.p = p
        self.resolution = resolution

    def select_raster_band_indices(self, time_range: TimeRange):
        return self.time_range.select_raster_band_indices(time_range)


class DatasetRepo(BaseModel):
    yearlies: Dict[str, YearlyRepo]
    monthlies: Dict[str, MonthlyRepo]

    def _get_dataset(self, dataset_id: str):
        dataset_found = False
        repo = None
        if dataset_id in self.yearlies:
            repo = self.yearlies
            dataset_found = True
        elif dataset_id in self.monthlies:
            repo = self.monthlies
            dataset_found = True

        if not dataset_found:
            raise DatasetNotFoundError(f'Dataset {dataset_id} not found')
        return repo[dataset_id]

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