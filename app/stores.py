import numpy as np
from collections import namedtuple
from functools import total_ordering
from pathlib import Path

import yaml

from app.exceptions import DatasetNotFoundError, VariableNotFoundError, TimeRangeContainmentError, TimeRangeInvalid
from app.settings import settings
from pydantic import BaseModel, Field, root_validator
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

    def _get_range(self):
        return range(self.gte, self.lte + 1)

    def __iter__(self):
        return iter(self._get_range())

    def __len__(self):
        return len(self._get_range())


Resolution = Literal['month', 'year']


class YearRange(BaseModel):
    gte: int = Field(..., ge=0, le=2100)
    lte: int = Field(..., ge=0, le=2100)

    @root_validator
    def check_gte_less_than_lte(cls, values):
        gte, lte = values.get('gte'), values.get('lte')
        if gte > lte:
            raise TimeRangeInvalid('gte must be less than or equal to lte')
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

    @root_validator
    def check_gte_less_than_lte(cls, values):
        gte, lte = values.get('gte'), values.get('lte')
        if gte > lte:
            raise TimeRangeInvalid('gte must be less than or equal to lte')
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


class MonthlyRepo(BaseModel):
    time_range: YearMonthRange
    variables: Set[str]

    @property
    def resolution(self) -> Literal['month']:
        return 'month'


TimeRange = Union[YearRange, YearMonthRange]


class DatasetMeta:
    def __init__(self, p: Path, time_range: TimeRange, resolution: Resolution):
        """
        :param p: path to the dataset
        :param time_range: span of time (and resolution) covered by the dataset
        """
        self.time_range = time_range
        self.p = p
        self.resolution = resolution

    def find_band_range(self, time_range: TimeRange):
        return self.time_range.find_band_range(time_range)


class DatasetRepo(BaseModel):
    year: Dict[str, YearlyRepo]
    month: Dict[str, MonthlyRepo]

    def _get_dataset(self, dataset_id: str, resolution: Resolution):
        repo = getattr(self, resolution)
        if dataset_id in repo:
            return repo[dataset_id]
        else:
            raise DatasetNotFoundError(f'Dataset {dataset_id} not found')

    def get_dataset_variables(self, dataset_id: str, resolution: Resolution):
        dataset = self._get_dataset(dataset_id, resolution=resolution)
        return dataset.variables

    def get_dataset_meta(self, dataset_id: str, variable_id: str, resolution: Resolution):
        dataset = self._get_dataset(dataset_id, resolution=resolution)

        if variable_id in dataset.variables:
            resolution = dataset.resolution
            time_range = dataset.time_range
        else:
            raise VariableNotFoundError(f'Variable {variable_id} not found in dataset {dataset_id}')

        p = settings.get_dataset_path(dataset_id=dataset_id, variable_id=variable_id)
        return DatasetMeta(p=p, time_range=time_range, resolution=resolution)


with settings.metadata_path.open() as f:
    dataset_repo = DatasetRepo(**yaml.safe_load(f))