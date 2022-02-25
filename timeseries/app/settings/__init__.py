import os

from .base import Settings


class MissedDeploymentSetting(KeyError):
    pass


_VALID_ENVIRONMENTS = ['dev', 'staging', 'prod']


def _get_environ():
    try:
        environment = os.environ['ENVIRONMENT']
        if environment not in _VALID_ENVIRONMENTS:
            raise MissedDeploymentSetting(
                f'Invalid ENVIRONMENT "{environment}", should be one of {_VALID_ENVIRONMENTS}')
        return environment
    except KeyError:
        raise MissedDeploymentSetting(
            f'Set ENVIRONMENT to one of {_VALID_ENVIRONMENTS}')


settings = Settings.from_envir(_get_environ())