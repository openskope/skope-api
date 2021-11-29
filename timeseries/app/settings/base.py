import logging.config
import yaml

from pathlib import Path
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def settings_override(base, overrides):
    if isinstance(base, dict) and isinstance(overrides, dict):
        for k in overrides.keys():
            if k in base and isinstance(base[k], dict):
                settings_override(base[k], overrides[k])
            else:
                base[k] = overrides[k]
    return base


class Store(BaseModel):
    base_path: str
    template: str
    uncertainty_template: str


class Settings(BaseModel):
    envir: str
    name: str
    base_url: str
    max_processing_time: int
    store: Store

    def init(self):
        with open(f'deploy/logging/{self.envir}.yml') as f:
            logging.config.dictConfig(yaml.safe_load(f))

    @classmethod
    def from_envir(cls, envir):
        with open('deploy/settings/base.yml') as f:
            base = yaml.safe_load(f) or {}

        with open(f'deploy/settings/{envir}.yml') as f:
            overrides = yaml.safe_load(f) or {}

        settings = settings_override(base, overrides)
        settings['envir'] = envir
        return cls(**settings)

    @property
    def metadata_path(self):
        if self.envir == 'dev':
            return Path('deploy/metadata/dev.yml')
        return Path('metadata.yml')

    def _get_path(self, template, dataset_id, variable_id):
        base = Path(self.store.base_path).resolve()
        path = Path(template.format(dataset_id=dataset_id, variable_id=variable_id)).resolve()
        try:
            path.relative_to(base)
        except ValueError as e:
            logger.warning('path traversal detected: base path %s, data path %s', base, path)
            raise e
        return path

    def get_dataset_path(self, dataset_id: str, variable_id: str) -> Path:
        return self._get_path(
            template=self.store.template,
            dataset_id=dataset_id,
            variable_id=variable_id)

    def get_uncertainty_dataset_path(self, dataset_id: str, variable_id: str) -> Path:
        return self._get_path(
            template=self.store.uncertainty_template,
            dataset_id=dataset_id,
            variable_id=variable_id)
