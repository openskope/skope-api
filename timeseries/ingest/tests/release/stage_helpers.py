"""Run the build stages up to a point, for tests of the later stages."""

from __future__ import annotations

from pathlib import Path

from skope_release.findings import Report
from skope_release.plan import Producer
from skope_release.preflight import DatasetFiles, preflight_dataset

PRODUCER = Producer("skope-api dataset pipeline", "0.1.0", "abc123")


def plan_for(directory: Path, release_id: str | None = "synth-r-2026.10.07", producer: Producer = PRODUCER):
    report = Report()
    plan = preflight_dataset(DatasetFiles(directory), release_id=release_id, mirror=None, producer=producer, report=report)
    assert plan is not None, report.to_text()
    return plan


def observed(directory: Path, staging: Path, release_id: str = "synth-r-2026.10.07"):
    """Plan, write COGs into `staging`, and return (plan, observation)."""
    from skope_release.cog_writer import write_cogs
    from skope_release.inspect_bytes import observe

    plan = plan_for(directory, release_id)
    write_cogs(plan, staging, staging.parent / f"{staging.name}.scratch")
    report = Report()
    observation = observe(plan, staging, report)
    assert observation is not None, report.to_text()
    return plan, observation
