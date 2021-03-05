import importlib
import os

from .base import Settings


class MissedDeploymentSetting(KeyError):
    pass


_VALID_ENVIRONS = ['dev', 'staging', 'prod']


def _get_environ():
    try:
        environ = os.environ['ENVIR']
        if environ not in _VALID_ENVIRONS:
            raise MissedDeploymentSetting(
                f'ENVIR variable "{environ}" not valid. Must be {", ".join(_VALID_ENVIRONS)}')
        return environ
    except KeyError:
        raise MissedDeploymentSetting(
            f'Unset ENVIR environment variable to one of {", ".join(_VALID_ENVIRONS)}')


settings = Settings.from_envir(_get_environ())