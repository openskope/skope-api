from functools import lru_cache
from logging.config import dictConfig
from typing import List, Optional, Tuple, Type
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)

import yaml
import logging

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])
    environment: str = "dev"
    name: str = "SKOPE API Services (development)"
    base_uri: str = "timeseries"
    max_processing_time: int = 20000  # in milliseconds
    max_concurrent_jobs: int = Field(default=1, ge=1, le=32)
    default_max_cells: int = 1_000_000
    max_series_options: int = Field(default=10, ge=1, le=100)
    max_geometry_shapes: int = Field(default=100, ge=1, le=1_000)
    max_geometry_coordinates: int = Field(default=10_000, ge=4, le=1_000_000)
    redis_url: Optional[str] = None
    sentry_dsn: str = "https://9b9dc2f60562380edeb675c39fe1c896@sentry.comses.net/4"
    tile_server_url: str
    # The pin file and where the pin's release_root is mounted (PIN-001).
    release_pin_path: str = "config/releases.yml"
    release_root: str = "/releases"
    # How datasets are shown (release consumption spec Section 4).
    display_preferences_path: str = "config/display/preferences.yml"
    display_palettes_path: str = "config/display/palettes.yml"

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
