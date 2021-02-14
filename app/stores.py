from functools import total_ordering
from pathlib import Path

from app.settings import settings
from pydantic import BaseModel, Field
from typing import Dict, List, Set, Union, Sequence


class DatasetNotFound(KeyError):
    pass


class VariableNotFound(KeyError):
    pass


class TimeIndexOutOfBounds(IndexError):
    pass


class YearRange(BaseModel):
    gte: int = Field(..., ge=0, le=2100)
    lte: int = Field(..., ge=0, le=2100)

    def contains(self, other: 'YearRange'):
        return self.gte <= other.gte and self.lte >= other.lte

    def select_raster_band_indices(self, selected_time_range: 'YearRange') -> Sequence[int]:
        if not self.contains(selected_time_range):
            raise TimeIndexOutOfBounds(f'{self} does not contain {selected_time_range}')

        offset = self.gte
        lb = selected_time_range.gte - offset
        ub = selected_time_range.lte - offset + 1
        return range(lb, ub)


class YearlyRepo(BaseModel):
    time_range: YearRange
    variables: Set[str]


@total_ordering
class YearMonth(BaseModel):
    year: int = Field(..., ge=0, le=2100)
    month: int = Field(..., ge=1, le=12)

    def to_index(self):
        # assumes there are not any BCE dates (0 is 0000-01 CE)
        return self.year*12 + self.month

    def __eq__(self, other: 'YearMonth'):
        return self.year == other.year and self.month == other.month

    def __lt__(self, other: 'YearMonth'):
        return (self.year, self.month) < (other.year, other.month)


class YearMonthRange(BaseModel):
    gte: YearMonth
    lte: YearMonth

    def contains(self, other: 'YearMonthRange'):
        return self.gte <= other.gte and self.lte >= other.lte

    def select_raster_band_indices(self, selected_time_range: 'YearMonthRange') -> Sequence[int]:
        """
        :param selected_time_range: time range to extract from raster
        :return: range of integer indices to extract from raster
        """
        if not self.contains(selected_time_range):
            raise TimeIndexOutOfBounds(f'{self} does not contain {selected_time_range}')

        offset = self.gte.to_index()
        lb = selected_time_range.gte.to_index() - offset
        ub = selected_time_range.lte.to_index() - offset + 1
        return range(lb, ub)


class MonthlyRepo(BaseModel):
    time_range: YearMonthRange
    variables: Set[str]


class DatasetMeta:
    def __init__(self, p: Path, time_range: Union[YearRange, YearMonthRange]):
        """
        :param p: path to the dataset
        :param time_range: span of time (and resolution) covered by the dataset
        """
        self.time_range = time_range
        self.p = p

    def select_raster_band_indices(self, time_range: Union[YearRange, YearMonthRange]):
        return self.time_range.select_raster_band_indices(time_range)


class DatasetRepo(BaseModel):
    yearlies: Dict[str, YearlyRepo]
    monthlies: Dict[str, MonthlyRepo]

    def get_dataset_meta(self, dataset_id: str, variable_id: str):
        time_range = None
        dataset_found = False
        variable_found = False
        if dataset_id in self.yearlies:
            dataset_found = True
            if variable_id in self.yearlies[dataset_id].variables:
                variable_found = True
                time_range = self.yearlies[dataset_id].time_range
        elif dataset_id in self.monthlies:
            dataset_found = True
            if variable_id in self.monthlies[dataset_id].variables:
                variable_found = True
                time_range = self.monthlies[dataset_id].time_range

        if not dataset_found:
            raise DatasetNotFound(f'Dataset {dataset_id} not found')

        if not variable_found:
            raise VariableNotFound(f'Variable {variable_id} not found in dataset {dataset_id}')

        p = settings.get_dataset_path(dataset_id=dataset_id, variable_id=variable_id)
        return DatasetMeta(p=p, time_range=time_range)


dataset_repo = DatasetRepo(
    yearlies={
        'annual_5x5x5_dataset': YearlyRepo(
            time_range=YearRange(gte=500, lte=504),
            variables=['float32_variable']
        )
    },
    monthlies={
        'monthly_5x5x60_dataset': MonthlyRepo(
            time_range=YearMonthRange(
                gte=YearMonth(year=2002, month=2),
                lte=YearMonth(year=2007, month=1)
            ),
            variables=['float32_variable']
        )
    }
)