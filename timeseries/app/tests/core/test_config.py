from app.config import Settings


def test_environment_overrides_yaml_settings(monkeypatch):
    monkeypatch.setenv("NAME", "Environment override")

    assert Settings().name == "Environment override"
