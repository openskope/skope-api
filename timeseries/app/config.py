from pathlib import Path
from pydantic import BaseSettings, BaseModel, validator, ValidationError

import yaml
import logging


logger = logging.getLogger(__name__)

BASE_CONFIG_PATH = "deploy/settings/base.yml"
_VALID_ENVIRONMENTS = ["dev", "staging", "prod"]


class Store(BaseModel):
    base_path: str
    template: str
    uncertainty_template: str


def yaml_config_settings_source(settings: BaseSettings) -> Dict[str, Any]:
    with Path(BASE_CONFIG_PATH).open() as f:
        return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    environment: str = "dev"
    name: str
    base_uri: str = "timeseries"
    max_processing_time: int = 50000
    store: Store

    @validator("environment")
    def validate_environment(cls, v):
        if v not in _VALID_ENVIRONMENTS:
            raise ValueError(f"ENVIRONMENT {v} should be one of {_VALID_ENVIRONMENTS}")
        return v

    @classmethod
    def from_envir(cls, environment):
        with open(f"deploy/settings/{environment}.yml") as f:
            overrides = yaml.safe_load(f) or {}

        settings_dict = settings_override(base, overrides)
        settings_dict["environment"] = environment

        instance = cls(**settings_dict)

        with open(instance.logging_config_file) as f:
            logging.config.dictConfig(yaml.safe_load(f))

        return instance

    @property
    def logging_config_file(self):
        return f"deploy/logging/{self.environment}.yml"

    @property
    def environment_settings_file(self):
        return f"deploy/settings/{self.environment}.yml"

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


settings = Settings()
