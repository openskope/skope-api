"""The release manifest and the closed release layout (Section 6, Section 13)."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from . import gdalio
from .documents import dumps_json, load_document
from .findings import Report
from .identity import IDENTITY_PROFILE
from .models import ReleaseManifest
from .plan import FinalObservation, require_observation

log = logging.getLogger(__name__)

MANIFEST ="release-manifest.json"
SCHEMA_VERSION = "1.0.0"

# REL-001: the only files a release may contain, with their roles (MAN-008).
_LAYOUT = [
    (re.compile(r"^collection\.json$"), ["stac", "collection"]),
    (re.compile(r"^overview\.yml$"), ["derived"]),
    (re.compile(r"^items/[a-z][a-z0-9_]*--[0-9TZ-]+--[0-9TZ-]+\.json$"), ["stac", "item"]),
    (re.compile(r"^cogs/[a-z][a-z0-9_]*/[a-z][a-z0-9_]*--[0-9TZ-]+--[0-9TZ-]+\.tif$"), ["data"]),
    (re.compile(r"^cogs/[a-z][a-z0-9_]*\.tif$"), ["data"]),
]


def roles_for(path: str) -> list[str] | None:
    for pattern, roles in _LAYOUT:
        if pattern.match(path):
            return roles
    return None


def release_files(root: Path) -> list[str]:
    """Every file in a release directory except the manifest, as sorted relative paths."""
    files = []
    for path in root.rglob("*"):
        if path.is_file() or path.is_symlink():
            rel = path.relative_to(root).as_posix()
            if rel != MANIFEST:
                files.append(rel)
    return sorted(files)


def build_manifest(observation: object, root: Path, report: Report) -> dict | None:
    """Inventory and checksum every file, then assemble the manifest (MAN-002 to MAN-006)."""
    obs = require_observation(observation)
    ctx = {"dataset": obs.dataset_id}
    observed = {a.path: a for a in obs.assets}
    files = []
    rels = release_files(root)
    total_gb = sum((root / rel).lstat().st_size for rel in rels) / 1e9
    checked_gb = 0.0
    for done, rel in enumerate(rels, 1):
        roles = roles_for(rel)
        path = root / rel
        if roles is None or path.is_symlink():
            report.add("REL-001", "file is not part of the closed release layout", path=rel, **ctx)
            continue
        checksum = gdalio.sha256_multihash(path)
        size = path.stat().st_size
        checked_gb += size / 1e9
        log.info("checksummed %d of %d files, %.1f of %.1f GB: %s", done, len(rels), checked_gb, total_gb, rel)
        fact = observed.get(rel)
        if fact is not None and (fact.size, fact.checksum) != (size, checksum):
            report.add("COG-009", "COG bytes changed after validation", path=rel, **ctx)
        files.append({"path": rel, "roles": roles, "size": size, "checksum": checksum})
    missing = set(observed) - {f["path"] for f in files}
    for rel in sorted(missing):
        report.add("MAN-006", "an observed COG is missing from the release", path=rel, **ctx)
    collections = [f for f in files if f["roles"] == ["stac", "collection"]]
    if [f["path"] for f in collections] != ["collection.json"]:
        report.add("MAN-008", "stac_entrypoint must resolve to exactly one Collection record", **ctx)
    if not report.ok:
        return None
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "release_id": obs.release_id,
        "created": obs.created,
        "status": "complete",
        "dataset": {"id": obs.dataset_id, "version": obs.curated.version, "stac_entrypoint": "collection.json"},
        "producer": {"name": obs.producer.name, "version": obs.producer.version, "revision": obs.producer.revision},
        "declaration": {"identity_profile": IDENTITY_PROFILE, "digest_algorithm": "sha256", "digest": obs.declaration_digest},
        "sources": [
            {"id": v.id, "href": v.source.uri, "checksum": v.source.checksum}
            for v in sorted(obs.resolved_manifest.variables, key=lambda v: v.id)
        ],
        "files": files,
    }
    from .documents import validate

    if validate(ReleaseManifest, manifest, requirement="MAN-001", path=MANIFEST, report=report, **ctx) is None:
        return None
    return manifest


def write_manifest(manifest: dict, root: Path) -> str:
    """Write the manifest last (MAN-007); returns its SHA-256 hex for the pin (PIN-001)."""
    content = dumps_json(manifest)
    (root / MANIFEST).write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def verify_release(root: Path, report: Report, *, full: bool = True) -> ReleaseManifest | None:
    """Verify a published release against its manifest.

    `full=True` checks every file's size and checksum (promotion, TXN-011);
    `full=False` checks presence and size only (startup, PIN-004).
    """
    manifest = load_document(ReleaseManifest, root / MANIFEST, requirement="MAN-001", report=report)
    if manifest is None:
        return None
    ctx = {"dataset": manifest.dataset.id}
    if root.name != manifest.release_id:
        report.add("TXN-011", f"directory {root.name} does not match release {manifest.release_id}", **ctx)
    listed = {f.path: f for f in manifest.files}
    for rel in release_files(root):
        if rel not in listed:
            report.add("REL-001", "file is not in the manifest inventory", path=rel, **ctx)
    total_gb = sum(f.size for f in manifest.files) / 1e9
    checked_gb = 0.0
    for done, (rel, entry) in enumerate(sorted(listed.items()), 1):
        if roles_for(rel) != list(entry.roles):
            report.add("MAN-008", f"roles {list(entry.roles)} do not match the layout", path=rel, **ctx)
        if ".." in rel.split("/") or rel.startswith("/"):
            report.add("ORG-010", "path must be release-relative and normalized", path=rel, **ctx)
            continue
        path = root / rel
        if not path.is_file():
            report.add("MAN-005", "inventoried file is missing", path=rel, **ctx)
            continue
        if path.stat().st_size != entry.size:
            report.add("MAN-005", f"size {path.stat().st_size} differs from the manifest's {entry.size}", path=rel, **ctx)
        elif full:
            if gdalio.sha256_multihash(path) != entry.checksum:
                report.add("MAN-005", "checksum differs from the manifest", path=rel, **ctx)
            checked_gb += entry.size / 1e9
            log.info("verified %d of %d files, %.1f of %.1f GB: %s", done, len(listed), checked_gb, total_gb, rel)
    return manifest if report.ok else None


def manifest_sha256(root: Path) -> str:
    return hashlib.sha256((root / MANIFEST).read_bytes()).hexdigest()
