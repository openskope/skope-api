"""Release identity: release IDs, the ledger, and the declaration digest.

- REL-006: `<dataset-id>-r-YYYY.MM.DD[-N]`, dated by `release.created`.
- MAN-010: SHA-256 of the RFC 8785 projection of the resolved declaration.
- MAN-011 and Section 20.3: one release ID is bound to one digest forever,
  recorded in the dataset's `releases.yml` ledger.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import rfc8785

from .findings import Report
from .models import Curated, Ledger, SourceManifest, parse_rfc3339_utc

IDENTITY_PROFILE = "openskope-release-declaration-v1"
_RELEASE_ID = re.compile(r"^(?P<dataset>[a-z][a-z0-9_]*)-r-(?P<date>\d{4}\.\d{2}\.\d{2})(?:-(?P<n>\d+))?$")


def parse_release_id(release_id: str) -> tuple[str, str, int]:
    """Return (dataset_id, YYYY.MM.DD, sequence) where the first ID of a day is 1."""
    match = _RELEASE_ID.match(release_id)
    if not match:
        raise ValueError(f"{release_id!r} does not match <dataset-id>-r-YYYY.MM.DD[-N] (REL-006)")
    n = match.group("n")
    if n is not None and (int(n) < 2 or n.startswith("0")):
        raise ValueError(f"{release_id!r}: a same-day suffix starts at -2 (REL-006)")
    return match.group("dataset"), match.group("date"), int(n) if n else 1


def check_release_id(
    release_id: str,
    *,
    dataset_id: str,
    created: str,
    digest: str,
    ledger: Ledger,
    report: Report,
) -> None:
    """REL-006, MAN-011, and the ledger rules of Section 20.3."""
    try:
        id_dataset, id_date, sequence = parse_release_id(release_id)
    except ValueError as exc:
        report.add("REL-006", str(exc), dataset=dataset_id)
        return
    if id_dataset != dataset_id:
        report.add("REL-006", f"release ID names dataset {id_dataset!r}, not {dataset_id!r}", dataset=dataset_id)
    created_date = parse_rfc3339_utc(created).strftime("%Y.%m.%d")
    if id_date != created_date:
        report.add(
            "REL-006",
            f"release ID date {id_date} differs from the UTC date of release.created ({created_date})",
            dataset=dataset_id,
        )

    entries = {entry.release_id: entry for entry in ledger.releases}
    existing = entries.get(release_id)
    if existing is not None:
        if existing.declaration_digest != digest:
            report.add(
                "MAN-011",
                f"{release_id} is already bound to digest {existing.declaration_digest}; "
                f"this declaration's digest is {digest}. A changed declaration needs a new release ID.",
                dataset=dataset_id,
            )
        if existing.withdrawn:
            report.add("REL-008", f"{release_id} was withdrawn ({existing.withdrawn}); its ID is not reused", dataset=dataset_id)
        return

    for entry in ledger.releases:
        if entry.declaration_digest == digest:
            report.add(
                "MAN-011",
                f"this declaration is already released as {entry.release_id}; two release IDs never share a digest",
                dataset=dataset_id,
            )
    same_day = [
        parse_release_id(entry.release_id)[2]
        for entry in ledger.releases
        if parse_release_id(entry.release_id)[:2] == (dataset_id, id_date)
    ]
    expected = max(same_day, default=0) + 1
    if sequence != expected:
        suffix = "" if expected == 1 else f"-{expected}"
        report.add(
            "REL-006",
            f"the next release ID for {dataset_id} on {id_date} is {dataset_id}-r-{id_date}{suffix}",
            dataset=dataset_id,
        )


def load_ledger(path: Path, report: Report) -> Ledger:
    from .documents import load_document

    if not path.exists():
        return Ledger()
    ledger = load_document(Ledger, path, requirement="REL-006", report=report)
    return ledger if ledger is not None else Ledger()


def declaration_projection(curated: Curated, resolved: SourceManifest) -> dict[str, Any]:
    """The canonical declaration graph (MAN-010, Section 20.3).

    `resolved` must carry every source checksum. Unordered collections
    (variables) are keyed by identifier; ordered arrays keep their order.
    Release IDs, host paths, generated outputs, and request metadata are not
    part of the declaration and never appear here.
    """
    if any(v.source.checksum is None for v in resolved.variables):
        raise ValueError("the declaration projection needs every source checksum resolved")
    curated_doc = curated.model_dump(mode="json")
    curated_doc["variables"] = {v.pop("id"): v for v in curated_doc["variables"]}
    manifest_doc = resolved.model_dump(mode="json")
    manifest_doc["variables"] = {v.pop("id"): v for v in manifest_doc["variables"]}
    # A source has a `uri` or a `prepare` block (META-013); it projects only the one it has.
    for v in manifest_doc["variables"].values():
        for origin in ("uri", "prepare"):
            if v["source"][origin] is None:
                del v["source"][origin]
    return {"identity_profile": IDENTITY_PROFILE, "curated": curated_doc, "source_manifest": manifest_doc}


def declaration_digest(projection: dict[str, Any]) -> str:
    return hashlib.sha256(rfc8785.dumps(projection)).hexdigest()
