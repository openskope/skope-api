"""Build one release in staging and publish it locally (TXN-001 to TXN-003, TXN-008, TXN-018).

Steps, in the order TXN-001 requires:
  1. write COGs                      (cog_writer)
  2. inspect the final COG bytes     (inspect_bytes)
  3. freeze the FinalObservation     (inspect_bytes)
  4. generate STAC and the overview  (stac_serializer, overview)
  5. every pre-manifest validation   (stac_validate, crosscheck, overview rule check)
  6. write the manifest last         (manifest)
  7. atomically rename staging to the release path, making it read-only for
     every user around the rename (TXN-018)

A failure leaves the staging directory in place and names it, and never
touches a published release (TXN-008).
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import shutil
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .cog_writer import write_cogs
from .gdalio import sha256_multihash
from .crosscheck import check_stac_against_bytes
from .findings import ReleaseError, Report
from .inspect_bytes import observe
from .manifest import build_manifest, release_files, write_manifest
from .overview import write_overview
from .plan import ValidatedBuildPlan, require_plan
from .stac_serializer import write_stac
from .stac_validate import validate_stac

log = logging.getLogger(__name__)

STAGING_DIR = ".staging"  # readers ignore dot-directories (TXN-003)
RELEASE_FILE_MODE = 0o444  # TXN-018
RELEASE_DIR_MODE = 0o555


@dataclass(frozen=True)
class Published:
    release_id: str
    path: Path
    declaration_digest: str
    manifest_sha256: str


class Lease:
    """Marks a staging directory as owned by a running build (TXN-013)."""

    def __init__(self, path: Path, release_id: str):
        self.path = path
        self.release_id = release_id

    def touch(self, stage: str) -> None:
        payload = {
            "release_id": self.release_id,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "stage": stage,
            "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        self.path.write_text(json.dumps(payload, sort_keys=True) + "\n")

    def release(self) -> None:
        self.path.unlink(missing_ok=True)


def _check_revision(plan: ValidatedBuildPlan, report: Report) -> None:
    revision = plan.producer.revision
    if not revision or revision == "unknown" or revision.endswith("-dirty"):
        report.add(
            "REL-009",
            f"producer revision {revision!r} is not a clean commit; release builds need a clean checkout",
            dataset=plan.dataset_id,
        )


def _make_contents_read_only(staging: Path) -> None:
    """Readable and not writable by every user (TXN-018), whatever the umask was.

    Leaves `staging` itself writable: moving a directory to a new parent needs
    write permission on it, so it gets its mode after the rename.
    """
    for directory, subdirectories, files in os.walk(staging):
        for name in files:
            os.chmod(os.path.join(directory, name), RELEASE_FILE_MODE)
        for name in subdirectories:
            os.chmod(os.path.join(directory, name), RELEASE_DIR_MODE)


def stage_release(plan: object, staging: Path) -> tuple[str, Report]:
    """Run TXN-001 steps 1–6 into `staging`. Returns the manifest SHA-256."""
    plan = require_plan(plan)
    report = Report()
    scratch = staging.parent / f"{staging.name}.scratch"
    log.info("%s: writing COGs", plan.release_id)
    write_cogs(plan, staging, scratch)

    log.info("%s: inspecting COG bytes", plan.release_id)
    observation = observe(plan, staging, report)
    report.raise_if_errors("COG byte validation")

    log.info("%s: writing STAC", plan.release_id)
    write_stac(observation, staging)
    validate_stac(staging, report)
    report.raise_if_errors("STAC validation")
    check_stac_against_bytes(observation, staging, report)
    report.raise_if_errors("cross-artifact validation")

    log.info("%s: generating overview.yml", plan.release_id)
    write_overview(staging, release_id=plan.release_id, declaration_digest=plan.declaration_digest, report=report)
    report.raise_if_errors("overview generation")

    manifest = build_manifest(observation, staging, report)
    report.raise_if_errors("release manifest")
    return write_manifest(manifest, staging), report


def build_and_publish(plan: object, release_root: Path) -> tuple[Published, Report]:
    plan = require_plan(plan)
    report = Report()
    _check_revision(plan, report)
    destination = release_root / plan.release_id
    if destination.exists():
        report.add("TXN-002", f"{destination} already exists; a published release is never rebuilt in place")
    report.raise_if_errors("publication preconditions")

    staging_root = release_root / STAGING_DIR
    staging_root.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(4)
    staging = staging_root / f"{plan.release_id}.{token}"
    staging.mkdir()
    if staging.stat().st_dev != release_root.stat().st_dev:
        report.add("TXN-002", "staging and the release root are on different filesystems")
        report.raise_if_errors("publication preconditions")

    lease = Lease(staging_root / f"{staging.name}.lease", plan.release_id)
    lease.touch("started")
    try:
        manifest_sha, stage_report = stage_release(plan, staging)
        report.extend(stage_report.findings)
        lease.touch("publishing")
        # TXN-001 step 7: atomic rename to a previously absent path (TXN-002).
        if destination.exists():
            report.add("TXN-002", f"{destination} appeared during the build")
            report.raise_if_errors("publication")
        _make_contents_read_only(staging)
        os.rename(staging, destination)
        os.chmod(destination, RELEASE_DIR_MODE)
    except ReleaseError as exc:
        exc.report.add("TXN-008", f"build failed; staging left at {staging} for inspection or cleanup", severity="warning")
        raise
    finally:
        lease.release()
    log.info("published %s", destination)
    return Published(plan.release_id, destination, plan.declaration_digest, manifest_sha), report


def verify_reproducible(plan: object, published: Path) -> Report:
    """Rebuild a published release into scratch and byte-compare every file (API-007, Section 20.3)."""
    plan = require_plan(plan)
    report = Report()
    staging_root = published.parent / STAGING_DIR
    staging_root.mkdir(parents=True, exist_ok=True)
    rebuild = staging_root / f"reproduce-{plan.release_id}.{secrets.token_hex(4)}"
    rebuild.mkdir()
    try:
        _, stage_report = stage_release(plan, rebuild)
        report.extend(stage_report.findings)
        ours, theirs = release_files(rebuild) + ["release-manifest.json"], release_files(published) + ["release-manifest.json"]
        if sorted(ours) != sorted(theirs):
            report.add("REL-009", f"file lists differ: {sorted(set(ours) ^ set(theirs))}")
        for rel in sorted(set(ours) & set(theirs)):
            if sha256_multihash(rebuild / rel) != sha256_multihash(published / rel):
                requirement = "API-007" if rel == "overview.yml" else "REL-009"
                report.add(requirement, "rebuilt bytes differ from the published bytes", path=rel)
    finally:
        shutil.rmtree(rebuild, ignore_errors=True)
    return report
