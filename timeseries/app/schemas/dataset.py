from collections import namedtuple
from datetime import date
from dateutil.relativedelta import relativedelta
from enum import Enum
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, Set

from app.exceptions import DatasetNotFoundError, VariableNotFoundError, TimeRangeContainmentError, TimeRangeInvalid
from app.schemas.common import BandRange, OptionalTimeRange, Resolution, TimeRange
from app.settings import settings

import logging
import yaml

logger = logging.getLogger(__name__)


"""
Pydantic schemas used to describe datasets in incoming timeseries queries
"""


class Dataset(BaseModel):
    time_range: TimeRange
    variables: Set[str]
    resolution: Resolution


class VariableMetadata:
    def __init__(self, path: Path, time_range: TimeRange, resolution: Resolution):
        """
        :param path: path to the dataset
        :param time_range: span of time (and resolution) covered by the dataset
        """
        self.time_range = time_range
        self.path = path
        self.resolution = resolution

    def normalize_time_range(self, otr: OptionalTimeRange):
        return TimeRange(
            gte=otr.gte if otr.gte is not None else self.time_range.gte,
            lte=otr.lte if otr.lte is not None else self.time_range.lte
        )

    def find_band_range(self, time_range: OptionalTimeRange) -> BandRange:
        """Translates time ranges from the metadata and request into a band range

        A band range is a linear 1-based index into the bands of a raster file
        """
        time_range = self.normalize_time_range(time_range)
        dataset_time_range = self.time_range
        if not (dataset_time_range.gte <= time_range.gte <= dataset_time_range.lte):
            raise TimeRangeContainmentError(f'{time_range.gte} is not within [{dataset_time_range.gte}, {dataset_time_range.lte}].')
        if not (dataset_time_range.gte <= time_range.lte <= dataset_time_range.lte):
            raise TimeRangeContainmentError(f'{time_range.lte} is not within [{dataset_time_range.gte}, {dataset_time_range.lte}].')
        gte_relative_delta = relativedelta(time_range.gte, dataset_time_range.gte)
        lte_relative_delta = relativedelta(time_range.lte, dataset_time_range.gte)
        if self.resolution == Resolution.month:
            min_index = gte_relative_delta.months + (gte_relative_delta.years * 12) + 1
            max_index = lte_relative_delta.months + (lte_relative_delta.years * 12) + 1
        else:
            min_index = gte_relative_delta.years + 1
            max_index = lte_relative_delta.years + 1
        return BandRange(gte=min_index, lte=max_index)

    def translate_band_range(self, br: BandRange) -> 'TimeRange':
        if self.resolution == Resolution.month:
            return TimeRange(
                gte=self.time_range.gte + relativedelta(months=br.gte - 1),
                lte=self.time_range.gte + relativedelta(months=br.lte - 1))
        elif self.resolution == Resolution.year:
            return TimeRange(
                gte=self.time_range.gte + relativedelta(years=br.gte - 1),
                lte=self.time_range.gte + relativedelta(years=br.lte - 1))
        else:
            raise ValueError(f'{self.resolution} is not valid. must be either year or month')


class DatasetManager(BaseModel):
    datasets: Dict[str, Dataset]

    def _get_dataset(self, dataset_id: str):
        dataset = self.datasets.get(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(f'Dataset {dataset_id} not found')
        return dataset

    def get_variables(self, dataset_id: str):
        dataset = self._get_dataset(dataset_id)
        return dataset.variables

    def get_variable_metadata(self, dataset_id: str, variable_id: str):
        dataset = self._get_dataset(dataset_id)
        logger.debug("retrieving dataset metadata %s", dataset)

        if variable_id in dataset.variables:
            resolution = dataset.resolution
            time_range = dataset.time_range
        else:
            logger.error("Unable to find variable %s in dataset variables %s", variable_id, dataset.variables)
            raise VariableNotFoundError(f'Variable {variable_id} not found in dataset {dataset_id}')

        path = settings.get_dataset_path(dataset_id=dataset_id, variable_id=variable_id)
        return VariableMetadata(path=path, time_range=time_range, resolution=resolution)



def load_metadata():
    """ FIXME: refactor
    Returns a dict of dataset ids mapped to simplified dataset metadata with 
    only 
    - time_range (gte/lte), 
    - resolution (always 'year' for now)
    - list of variable names
    """
    with open(settings.metadata_path) as f:
        datasets = yaml.safe_load(f)
    metadata_dict = {}
    for dataset in datasets:
        dataset_id = dataset['id']
        metadata_dict[dataset_id] = Dataset(**dataset)
    return metadata_dict


def load_api_metadata():
    """
    Returns a dict of all dataset metadata exposed by the metadata API
    endpoint
    """
    with open('metadata.yml') as f:
        datasets = yaml.safe_load(f)
    metadata_dict = {}
    for dataset in datasets:
        dataset_id = dataset['id']
        metadata_dict[dataset_id] = dataset
    return metadata_dict

def get_dataset_manager():
    return DatasetManager(datasets=load_metadata())