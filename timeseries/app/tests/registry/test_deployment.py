"""`python -m app.registry.deployment`: the deployment log (TXN-011, DISP-011)."""

import json
from datetime import datetime, timezone

import pytest

from app.registry import deployment

RELEASE = {
    "dataset": "annual",
    "release_id": "annual-r-2026.01.01",
    "declaration_digest": "a" * 64,
    "manifest_sha256": "b" * 64,
    "path": "/releases/annual-r-2026.01.01",
    "verification": "full",
    "result": "passed",
}


def _pending(tmp_path):
    pending = tmp_path / ".pending-staging.json"
    pending.write_text(json.dumps({"releases": [RELEASE]}) + "\n")
    return pending


def test_appends_one_line_per_deploy(tmp_path):
    log = tmp_path / "staging.jsonl"
    for day in (1, 2):
        pending = _pending(tmp_path)
        deployment.record(
            pending,
            log,
            environment="staging",
            commit="8fbfa9a",
            deployed_by="someone",
            images={"titiler": "sha256:2", "server": "sha256:1"},
            now=datetime(2026, 11, day, 14, 2, 11, tzinfo=timezone.utc),
        )
        assert not pending.exists()

    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [entry["deployed_at"] for entry in lines] == [
        "2026-11-01T14:02:11Z",
        "2026-11-02T14:02:11Z",
    ]
    assert lines[0] == {
        "deployed_at": "2026-11-01T14:02:11Z",
        "environment": "staging",
        "deployed_by": "someone",
        "commit": "8fbfa9a",
        "images": {"server": "sha256:1", "titiler": "sha256:2"},
        "releases": [RELEASE],
    }


def test_command_line(tmp_path, capsys):
    pending, log = _pending(tmp_path), tmp_path / "staging.jsonl"
    args = [str(pending), str(log), "--environment", "staging", "--commit", "c-dirty"]
    args += ["--deployed-by", "someone", "--image", "server=sha256:1"]

    assert deployment.main(args) == 0

    entry = json.loads(log.read_text())
    assert (entry["commit"], entry["images"]) == ("c-dirty", {"server": "sha256:1"})
    assert "recorded the staging deploy" in capsys.readouterr().out


def test_rejects_a_malformed_image(tmp_path):
    with pytest.raises(SystemExit):
        deployment.main(
            [
                str(_pending(tmp_path)),
                str(tmp_path / "log.jsonl"),
                "--environment",
                "staging",
                "--commit",
                "c",
                "--deployed-by",
                "someone",
                "--image",
                "server",
            ]
        )
