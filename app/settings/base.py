import logging.config
import yaml

from pathlib import Path
from pydantic import BaseModel
from typing import Callable


class Settings(BaseModel):
    envir: str
    name: str
    base_url: str
    max_processing_time: int
    get_dataset_path: Callable[[str, str], Path]
    get_uncertainty_dataset_path: Callable[[str, str], Path]

    def init(self):
        with open(f'deploy/logging/{self.envir}.yml') as f:
            logging.config.dictConfig(yaml.safe_load(f))

    @property
    def metadata_path(self):
        return Path(f'deploy/metadata/{self.envir}.yml')