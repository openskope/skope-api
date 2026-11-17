"""`python -m app.registry.verify`: promotion's full byte check (TXN-011)."""

import json
import logging

import numpy as np
import pytest

from app.registry import compose, verify
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

    assert verify.main([]) == 0

    record = json.loads(capsys.readouterr().out)
    assert record["release_id"] == "annual-r-2026.01.01"
    assert record["path"] == str(path)
    assert record["result"] == "passed"


def test_logs_progress_per_file(tmp_path, monkeypatch, caplog):
    _setup(tmp_path, monkeypatch)
    caplog.set_level(logging.INFO, logger="app.registry.compose")

    assert verify.main([]) == 0

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

    assert verify.main(["--record", str(tmp_path / "pending.json")]) == 1
    assert "verification failed: 1 check(s)" in capsys.readouterr().out
    assert not (tmp_path / "pending.json").exists()


def test_records_the_verified_releases(tmp_path, monkeypatch):
    path = _setup(tmp_path, monkeypatch)
    pending = tmp_path / "pending.json"

    assert verify.main(["--record", str(pending)]) == 0

    (release,) = json.loads(pending.read_text())["releases"]
    assert release["release_id"] == "annual-r-2026.01.01"
    assert release["path"] == str(path)
    assert (release["verification"], release["result"]) == ("full", "passed")


def test_metadata_only_skips_asset_hashes_and_records_its_mode(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    pending = tmp_path / "pending.json"

    def unexpected_hash(path):
        pytest.fail(f"metadata verification hashed an asset: {path}")

    monkeypatch.setattr(compose, "multihash", unexpected_hash)
    assert verify.main(["--metadata-only", "--record", str(pending)]) == 0

    (release,) = json.loads(pending.read_text())["releases"]
    assert (release["verification"], release["result"]) == ("metadata", "passed")


@pytest.mark.parametrize("damage", ["manifest", "overview", "missing", "size"])
def test_metadata_only_still_rejects_invalid_releases(tmp_path, monkeypatch, damage):
    path = _setup(tmp_path, monkeypatch)
    cog = next((path / "cogs" / "ppt").iterdir())
    if damage == "manifest":
        manifest = path / "release-manifest.json"
        manifest.write_text(manifest.read_text() + "\n")
    elif damage == "overview":
        overview = path / "overview.yml"
        data = overview.read_bytes()
        overview.write_bytes(data.replace(b"annual", b"forged", 1))
    elif damage == "missing":
        cog.unlink()
    else:
        cog.write_bytes(cog.read_bytes()[:-1])
    pending = tmp_path / "pending.json"

    assert verify.main(["--metadata-only", "--record", str(pending)]) == 1
    assert not pending.exists()


def test_metadata_only_cannot_detect_same_size_asset_corruption(tmp_path, monkeypatch):
    path = _setup(tmp_path, monkeypatch)
    cog = next((path / "cogs" / "ppt").iterdir())
    data = bytearray(cog.read_bytes())
    data[-1] ^= 0xFF
    cog.write_bytes(data)

    assert verify.main(["--metadata-only"]) == 0
    assert verify.main([]) == 1
