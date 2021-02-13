import importlib
import os


class MissedDeploymentSetting(KeyError):
    pass


_VALID_SETTINGS_MODULES = ['app.settings.dev', 'app.settings.staging', 'app.settings.prod']


def _get_settings_module():
    try:
        settings_module = os.environ['SETTINGS_MODULE']
        if settings_module not in _VALID_SETTINGS_MODULES:
            raise MissedDeploymentSetting(
                f'SETTINGS_MODULE variable "{settings_module}" not valid. Must be {", ".join(_VALID_SETTINGS_MODULES)}')
        return settings_module
    except KeyError:
        raise MissedDeploymentSetting(
            f'Unset SETTINGS_MODULE environment variable to one of {", ".join(_VALID_SETTINGS_MODULES)}')


def _get_settings():
    module_dynamic = _get_settings_module()
    module = importlib.import_module(module_dynamic)
    return getattr(module, 'settings')


settings = _get_settings()