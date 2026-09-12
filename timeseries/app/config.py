from functools import lru_cache
from logging.config import dictConfig
from pathlib import Path
from typing import List, Optional, Tuple, Type
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)

import yaml
import logging

logger = logging.getLogger(__name__)


class Store(BaseModel):
    base_path: str
    template: str
    uncertainty_template: str


class Settings(BaseSettings):
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])
    environment: str = "dev"
    name: str = "SKOPE API Services (development)"
    base_uri: str = "timeseries"
    max_processing_time: int = 15000  # in milliseconds
    max_concurrent_jobs: int = Field(default=1, ge=1, le=32)
    default_max_cells: int = 1_000_000
    max_series_options: int = Field(default=10, ge=1, le=100)
    max_geometry_shapes: int = Field(default=100, ge=1, le=1_000)
    max_geometry_coordinates: int = Field(default=10_000, ge=4, le=1_000_000)
    store: Store
    redis_url: Optional[str] = None
    sentry_dsn: str = "https://9b9dc2f60562380edeb675c39fe1c896@sentry.comses.net/4"
    tile_server_url: str
    storage_base_url: str

    model_config = SettingsConfigDict(yaml_file="config/app_settings.yml")

    @classmethod
    def create(cls):
        instance = cls()
        with open(instance.logging_config_file) as f:
            dictConfig(yaml.safe_load(f))
        return instance

    @property
    def is_production(self):
        return self.environment == "prod"

    @property
    def logging_config_file(self):
        return "config/logging.yml"

    @property
    def registry_path(self):
        return Path("metadata.yml")

    @property
    def colormaps_path(self):
        return Path("config/colormaps.json")

    def _get_path(self, template, dataset_id, variable_id):
        base = Path(self.store.base_path).resolve()
        path = Path(
            template.format(dataset_id=dataset_id, variable_id=variable_id)
        ).resolve()
        try:
            path.relative_to(base)
        except ValueError:
            logger.warning(
                "path traversal detected: base path %s, data path %s", base, path
            )
            raise
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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache()
def get_settings():
    return Settings.create()
