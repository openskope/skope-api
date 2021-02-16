from .base import Settings
from .dev import get_dataset_path, get_uncertainty_dataset_path

settings = Settings(
    envir='prod',
    name='SKOPE Timeseries Service',
    base_url='timeseries-service/api/v1',
    max_processing_time=5000,
    get_dataset_path=get_dataset_path,
    get_uncertainty_dataset_path=get_uncertainty_dataset_path
)