"""Pin a published release from its manifest (spec §20.3, release consumption PIN-001).

Pinning a release means three edits, all with values its release manifest holds:
  1. the ledger entry in `datasets/<dataset>/releases.yml` (REL-006);
  2. the dataset's entry in `deploy/releases/<environment>.yml` (PIN-001);
  3. a `checksum` for every source the source manifest declares without one
     (META-004), as the build measured it.

`plan_pin` works out each file's new text and changes nothing, so the caller can
show the edits before `write_pin` applies them. Edits are made line by line, so
comments and layout survive, and each one is checked by parsing the result
against the document it is meant to produce.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from .documents import load_document
from .findings import Report
from .identity import load_ledger
from .manifest import MANIFEST
from .models import ReleaseManifest

LEDGER_HEADER = """\
# Release IDs assigned to {dataset} and the declaration each one was built
# from (REL-006, spec §20.3). An ID is never reused; a bad release gets
# `withdrawn: <reason>` instead of being removed.
releases:
"""


@dataclass(frozen=True)
class FileEdit:
    path: Path
    label: str  # the path as the reader knows it, e.g. deploy/releases/staging.yml
    before: str  # "" for a new file
    after: str

    def diff(self) -> str:
        return "".join(difflib.unified_diff(
            self.before.splitlines(keepends=True), self.after.splitlines(keepends=True),
            f"a/{self.label}", f"b/{self.label}",
        ))


def manifest_path(path: Path) -> Path:
    """A release directory or its manifest file."""
    return path / MANIFEST if path.is_dir() else path


def plan_pin(manifest_file: Path, *, datasets: Path, pins: Path, environment: str, report: Report) -> list[FileEdit] | None:
    """The edits that pin this release in `environment`, or None if any check fails."""
    manifest = load_document(ReleaseManifest, manifest_file, requirement="MAN-001", report=report)
    if manifest is None:
        return None
    dataset = manifest.dataset.id
    authoring = datasets / dataset
    if not authoring.is_dir():
        report.add("META-001", f"no authoring directory for {dataset}", path=str(authoring))
        return None
    pin = {
        "dataset": dataset,
        "release_id": manifest.release_id,
        "declaration_digest": manifest.declaration.digest,
        "manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
    }
    edits = [
        _ledger_edit(authoring / "releases.yml", f"datasets/{dataset}/releases.yml", pin, report),
        _pin_edit(pins / f"{environment}.yml", f"deploy/releases/{environment}.yml", pin, report),
        _checksum_edit(authoring / "source-manifest.yml", f"datasets/{dataset}/source-manifest.yml", manifest, report),
    ]
    if not report.ok:
        return None
    return [edit for edit in edits if edit is not None]


def write_pin(edits: list[FileEdit]) -> None:
    for edit in edits:
        edit.path.write_text(edit.after, encoding="utf-8")


def _ledger_edit(path: Path, label: str, pin: dict, report: Report) -> FileEdit | None:
    ledger = load_ledger(path, report)
    entry = {"release_id": pin["release_id"], "declaration_digest": pin["declaration_digest"]}
    for existing in ledger.releases:
        if existing.release_id != pin["release_id"]:
            continue
        if existing.declaration_digest != pin["declaration_digest"]:
            report.add("REL-006", f"the ledger records {pin['release_id']} with digest {existing.declaration_digest}", path=label)
        elif existing.withdrawn:
            report.add("REL-006", f"{pin['release_id']} is withdrawn ({existing.withdrawn}) and is never pinned", path=label)
        return None
    before = path.read_text("utf-8") if path.exists() else ""
    after = _with_newline(before) if before else LEDGER_HEADER.format(dataset=pin["dataset"])
    after += f'  - release_id: "{entry["release_id"]}"\n    declaration_digest: "{entry["declaration_digest"]}"\n'
    expected = (_load(before) or {}).get("releases") or []
    _check(after, lambda doc: doc.get("releases") == [*expected, entry], label, report)
    return FileEdit(path, label, before, after)


def _pin_edit(path: Path, label: str, pin: dict, report: Report) -> FileEdit | None:
    if not path.is_file():
        report.add("PIN-001", "pin file not found", path=label)
        return None
    before = path.read_text("utf-8")
    doc = _load(before) or {}
    entries = doc.get("releases") or []
    if pin in entries:
        return None
    lines = _with_newline(before).splitlines(keepends=True)
    start = _find(lines, rf"(\s*)- dataset:\s*[\"']?{re.escape(pin['dataset'])}[\"']?\s*(#.*)?")
    if start is None:
        # A dataset not pinned yet goes at the end of the list.
        at = end = len(lines)
        indent = "  "
        expected = [*entries, pin]
    else:
        at, indent = start
        end = _block_end(lines, at, len(indent))
        expected = [pin if e.get("dataset") == pin["dataset"] else e for e in entries]
    block = [
        f"{indent}- dataset: {pin['dataset']}\n",
        f"{indent}  release_id: {pin['release_id']}\n",
        f"{indent}  declaration_digest: {pin['declaration_digest']}\n",
        f"{indent}  manifest_sha256: {pin['manifest_sha256']}\n",
    ]
    after = "".join(lines[:at] + block + lines[end:])
    rest = {k: v for k, v in doc.items() if k != "releases"}
    _check(after, lambda new: new.get("releases") == expected and {k: v for k, v in new.items() if k != "releases"} == rest,
           label, report)
    return FileEdit(path, label, before, after)


def _checksum_edit(path: Path, label: str, manifest: ReleaseManifest, report: Report) -> FileEdit | None:
    if not path.is_file():
        report.add("META-004", "source manifest not found", path=label)
        return None
    before = path.read_text("utf-8")
    doc = _load(before) or {}
    measured = {source.id: source.checksum for source in manifest.sources}
    variables = doc.get("variables") or []
    if sorted(v.get("id") for v in variables) != sorted(measured):
        report.add("META-004", "the source manifest's variables differ from the release's sources; "
                   "it has changed since this release was built", path=label)
        return None
    lines = before.splitlines(keepends=True)
    expected = yaml.safe_load(before)
    for variable, wanted in zip(variables, expected["variables"]):
        declared = variable["source"].get("checksum")
        checksum = measured[variable["id"]]
        if declared is not None:
            if declared != checksum:
                report.add("META-004", f"declared checksum {declared} differs from the measured {checksum}",
                           path=label, variable=variable["id"])
            continue
        start = _find(lines, rf"(\s*)- id:\s*[\"']?{re.escape(variable['id'])}[\"']?\s*(#.*)?")
        uri = None
        if start is not None:
            at, indent = start
            found = _find(lines[at:_block_end(lines, at, len(indent))], r"(\s*)uri:.*")
            if found is not None:
                uri = (at + found[0], found[1])
        if uri is None:
            report.add("META-004", "cannot find the source's uri line; add the checksum by hand",
                       path=label, variable=variable["id"])
            continue
        lines.insert(uri[0] + 1, f'{uri[1]}checksum: "{checksum}"\n')
        wanted["source"]["checksum"] = checksum
    after = "".join(lines)
    if after == before:
        return None
    _check(after, lambda new: new == expected, label, report)
    return FileEdit(path, label, before, after)


def _find(lines: list[str], pattern: str) -> tuple[int, str] | None:
    """The first line matching `pattern` in full, with its first group (the indent)."""
    for i, line in enumerate(lines):
        match = re.fullmatch(pattern, line.rstrip("\n"))
        if match:
            return i, match.group(1)
    return None


def _block_end(lines: list[str], start: int, indent: int) -> int:
    """The line after the list item that starts at `start`, whose dash is at `indent`.

    Blank and comment lines after the item's last key stay outside it, so a
    comment above the next item is kept.
    """
    end = start + 1
    for i in range(start + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            continue
        if len(lines[i]) - len(lines[i].lstrip()) <= indent:
            break
        end = i + 1
    return end


def _with_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def _load(text: str):
    return yaml.safe_load(text) if text else None


def _check(after: str, holds, label: str, report: Report) -> None:
    """Refuse an edit whose result does not parse to the intended document."""
    try:
        ok = holds(yaml.safe_load(after) or {})
    except yaml.YAMLError:
        ok = False
    if not ok:
        report.add("PIN-001", "could not edit this file safely; make the change by hand", path=label)
