"""Copy a dataset's declared sources into the source mirror (OBS-014).

Preflight and build read a source from the mirror when a copy is there, so a
fetched source is downloaded once rather than streamed by every run. A copy is
written under a temporary name and takes its final name only when it is
complete and matches the source manifest's checksum, when one is declared.

A prepared source (META-013) is made here and nowhere else: fetch copies its
inputs, runs its script, checks the output, and records how it was made
(OBS-015).
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from . import gdalio, prepare
from .documents import load_document
from .findings import Report
from .models import Prepare, SourceManifest, SourceVariable
from .preflight import DatasetFiles

# The last lines of a failing script's output that go into its finding.
SCRIPT_OUTPUT_LINES = 20


def fetch_dataset(
    files: DatasetFiles,
    *,
    mirror: Path,
    report: Report,
    remote: Callable[[str], str] = lambda uri: gdalio.read_path(uri, None),
    scripts_root: Path = prepare.INGEST_ROOT,
    toolchain: dict | None = None,
) -> list[tuple[str, str]]:
    """Fetch every mirrorable source; return (variable, outcome) pairs.

    `remote` maps a source URI to the path GDAL streams it from.
    """
    ctx = {"dataset": files.dataset_id}
    manifest = load_document(SourceManifest, files.source_manifest, requirement="META-004", report=report, **ctx)
    if manifest is None:
        return []
    tools = toolchain if toolchain is not None else prepare.toolchain()
    outcomes = []
    for var in manifest.variables:
        vctx = {**ctx, "variable": var.id}
        if var.source.prepare is not None:
            outcome = _prepare(files.dataset_id, var, mirror=mirror, report=report, remote=remote,
                               scripts_root=scripts_root, tools=tools, ctx=vctx)
        else:
            outcome, _ = _copy(var.source.uri, var.source.checksum, mirror=mirror, report=report, remote=remote, ctx=vctx)
        outcomes.append((var.id, outcome))
    return outcomes


def _copy(uri: str, declared: str | None, *, mirror: Path, report: Report, remote, ctx) -> tuple[str, str | None]:
    """Mirror one URI; return its outcome and, when the copy is good, its checksum."""
    target = gdalio.mirror_path(uri, mirror)
    if target is None:
        report.warn("OBS-014", f"{uri} has no mirror location; preflight reads it directly", **ctx)
        return "skipped", None

    if target.is_file():
        checksum = gdalio.sha256_multihash(target)
        if declared is None:
            report.warn("OBS-011", f"mirror copy has no declared checksum to match; its checksum is {checksum}", path=str(target), **ctx)
        elif checksum != declared:
            report.add("OBS-011", f"mirror copy has checksum {checksum}, not the declared {declared}", path=str(target), **ctx)
            return "mismatch", None
        return "present", checksum

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    try:
        checksum = gdalio.copy_with_multihash(remote(uri), partial)
    except Exception as exc:  # noqa: BLE001 - every source failure becomes a finding
        partial.unlink(missing_ok=True)
        report.add("OBS-014", f"cannot fetch {uri}: {exc}", **ctx)
        return "failed", None
    if declared is not None and checksum != declared:
        partial.unlink()
        report.add("OBS-011", f"fetched {uri} has checksum {checksum}, not the declared {declared}", **ctx)
        return "mismatch", None
    partial.replace(target)
    if declared is None:
        report.warn("OBS-011", f"fetched without a declared checksum; its checksum is {checksum}", path=str(target), **ctx)
    return "fetched", checksum


def _prepare(dataset_id: str, var: SourceVariable, *, mirror: Path, report: Report, remote,
             scripts_root: Path, tools: dict, ctx) -> str:
    """Make a prepared source in the mirror, or confirm the one there (OBS-015)."""
    spec: Prepare = var.source.prepare
    script = prepare.script_path(spec, scripts_root)
    if not script.is_file():
        report.add("META-013", f"prepare script {spec.script} does not exist", **ctx)
        return "failed"

    inputs, paths = [], []
    for item in spec.inputs:
        outcome, checksum = _copy(item.uri, item.checksum, mirror=mirror, report=report, remote=remote, ctx=ctx)
        if checksum is None:
            if outcome == "skipped":
                report.add("OBS-014", f"prepare input {item.uri} has no mirror location", **ctx)
            return outcome if outcome == "mismatch" else "failed"
        inputs.append({"uri": item.uri, "checksum": checksum})
        paths.append(gdalio.mirror_path(item.uri, mirror))

    declared = var.source.checksum
    output = prepare.output_path(mirror, dataset_id, var.id)
    script_sha256 = prepare.sha256_hex(script)
    record = prepare.read_record(output) if output.is_file() else None
    if prepare.stale_reason(record, spec, script_sha256, tools) is None and record["inputs"] == inputs:
        checksum = gdalio.sha256_multihash(output)
        if checksum == record["output_checksum"] and declared in (None, checksum):
            if declared is None:
                report.warn("OBS-011", f"prepared without a declared checksum; its checksum is {checksum}", path=str(output), **ctx)
            return "present"

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + ".part.tif")
    partial.unlink(missing_ok=True)
    env = {**os.environ, "GDAL_PAM_ENABLED": "NO"}  # no .aux.xml beside the output
    run = subprocess.run(
        [sys.executable, str(script), var.id, str(partial), *map(str, paths)],
        capture_output=True, text=True, env=env, check=False,
    )
    if run.returncode != 0 or not partial.is_file():
        partial.unlink(missing_ok=True)
        tail = "\n".join((run.stdout + run.stderr).strip().splitlines()[-SCRIPT_OUTPUT_LINES:])
        report.add("OBS-015", f"prepare script {spec.script} failed (exit {run.returncode}): {tail}", **ctx)
        return "failed"

    checksum = gdalio.sha256_multihash(partial)
    if declared is not None and checksum != declared:
        partial.unlink()
        output.unlink(missing_ok=True)
        prepare.record_path(output).unlink(missing_ok=True)
        report.add("OBS-015", f"prepared output has checksum {checksum}, not the declared {declared}", **ctx)
        return "mismatch"
    partial.replace(output)
    prepare.write_record(output, prepare.make_record(spec, script_sha256, tools, inputs, checksum))
    if declared is None:
        report.warn("OBS-011", f"prepared without a declared checksum; its checksum is {checksum}", path=str(output), **ctx)
    return "prepared"
