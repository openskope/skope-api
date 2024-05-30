from functools import lru_cache
from logging.config import dictConfig
from pathlib import Path
from pydantic import BaseSettings, BaseModel
from typing import Dict, Any

import yaml
import logging


logger = logging.getLogger(__name__)

YAML_CONFIG_PATH = "deploy/settings/config.yml"


class Store(BaseModel):
    base_path: str
    template: str
    uncertainty_template: str


def yaml_config_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    with Path(YAML_CONFIG_PATH).open() as f:
        return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    allowed_origins = ["*"]
    environment = "dev"
    name = "SKOPE API Services (development)"
    base_uri = "timeseries"
    max_processing_time = 15000  # in milliseconds
    default_max_cells = 500000  # max number of cells to extract from data cubes
    store: Store
    sentry_dsn = "https://9b9dc2f60562380edeb675c39fe1c896@sentry.comses.net/4"

    @classmethod
    def create(cls):
        instance = Settings()
        with open(instance.logging_config_file) as f:
            dictConfig(yaml.safe_load(f))
        return instance

    @property
    def is_production(self):
        return self.environment == "prod"

    @property
    def logging_config_file(self):
        return f"deploy/logging/{self.environment}.yml"

    @property
    def metadata_path(self):
        """
        FIXME: dataset metadata is currently duplicated across
        deploy/metadata/prod.yml and metadata.yml and should
        be de-duplicated but this brings some pain into how
        the pydantic base classes for Dataset
        were constructed
        """
        return Path(f"deploy/metadata/{self.environment}.yml")

    def _get_path(self, template, dataset_id, variable_id):
        base = Path(self.store.base_path).resolve()
        path = Path(
            template.format(dataset_id=dataset_id, variable_id=variable_id)
        ).resolve()
        try:
            path.relative_to(base)
        except ValueError as e:
            logger.warning(
                "path traversal detected: base path %s, data path %s", base, path
            )
            raise e
        return path

    def get_dataset_path(self, dataset_id: str, variable_id: str) -> Path:
        return self._get_path(
            template=self.store.template, dataset_id=dataset_id, variable_id=variable_id
        )

    def get_uncertainty_dataset_path(self, dataset_id: str, variable_id: str) -> Path:
        return self._get_path(
            template=self.store.uncertainty_template,
            dataset_id=dataset_id,
            variable_id=variable_id,
        )

    class Config:
        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            return (
                init_settings,
                yaml_config_settings_source,
                env_settings,
                file_secret_settings,
            )


@lru_cache()
def get_settings():
    return Settings.create()
