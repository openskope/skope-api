"""`python -m app.registry.verify`: promotion's full byte check (TXN-011)."""

import json
import logging

import numpy as np

from app.registry import verify
from app.tests.release_builder import build_release, write_pin


def _setup(tmp_path, monkeypatch):
    root = tmp_path / "releases"
    pinned = build_release(root, "annual", {"ppt": np.zeros((3, 2, 2), np.float32)})
    pin = write_pin(tmp_path / "releases.yml", root, [pinned])
    settings = verify.get_settings()
    monkeypatch.setattr(settings, "release_pin_path", str(pin))
    monkeypatch.setattr(settings, "release_root", str(root))
    return root / pinned.release_id


def test_prints_a_record_per_verified_release(tmp_path, monkeypatch, capsys):
    path = _setup(tmp_path, monkeypatch)

    assert verify.main() == 0

    record = json.loads(capsys.readouterr().out)
    assert record["release_id"] == "annual-r-2026.01.01"
    assert record["path"] == str(path)
    assert record["result"] == "passed"


def test_logs_progress_per_file(tmp_path, monkeypatch, caplog):
    _setup(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO, logger="app.registry.compose")

    assert verify.main() == 0

    messages = [r.getMessage() for r in caplog.records]
    progress = [m for m in messages if "files checked" in m]
    assert len(progress) > 1
    assert progress[0].startswith("annual-r-2026.01.01: 1 of ")
    n = len(progress)
    assert progress[-1].startswith(f"annual-r-2026.01.01: {n} of {n} files checked")


def test_fails_on_changed_bytes(tmp_path, monkeypatch, capsys):
    path = _setup(tmp_path, monkeypatch)
    cog = next((path / "cogs" / "ppt").iterdir())
    data = bytearray(cog.read_bytes())
    data[-1] ^= 0xFF
    cog.write_bytes(bytes(data))

    assert verify.main() == 1
    assert "verification failed: 1 check(s)" in capsys.readouterr().out
