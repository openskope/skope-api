from pathlib import Path
from .base import Settings


def get_dataset_path(dataset_id: str, variable_id: str) -> Path:
    return Path(f'data/{dataset_id}_{variable_id}.tif')


def get_uncertainty_dataset_path(dataset_id: str, variable_id: str) -> Path:
    return Path(f'data/{dataset_id}_{variable_id}_uncertainty.tif')


settings = Settings(
    envir='dev',
    name='SKOPE Timeseries Service',
    base_url='timeseries-service/api/v1',
    max_processing_time=5000,
    get_dataset_path=get_dataset_path,
    get_uncertainty_dataset_path=get_uncertainty_dataset_path
)