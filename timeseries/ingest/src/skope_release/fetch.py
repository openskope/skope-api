"""Copy a dataset's declared sources into the source mirror (OBS-014).

Preflight and build read a source from the mirror when a copy is there, so a
fetched source is downloaded once rather than streamed by every run. A copy is
written under a temporary name and takes its final name only when it is
complete and matches the source manifest's checksum, when one is declared.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from . import gdalio
from .documents import load_document
from .findings import Report
from .models import SourceManifest
from .preflight import DatasetFiles


def fetch_dataset(
    files: DatasetFiles,
    *,
    mirror: Path,
    report: Report,
    remote: Callable[[str], str] = lambda uri: gdalio.read_path(uri, None),
) -> list[tuple[str, str]]:
    """Fetch every mirrorable source; return (variable, outcome) pairs.

    `remote` maps a source URI to the path GDAL streams it from.
    """
    ctx = {"dataset": files.dataset_id}
    manifest = load_document(SourceManifest, files.source_manifest, requirement="META-004", report=report, **ctx)
    if manifest is None:
        return []
    outcomes = []
    for var in manifest.variables:
        src = var.source
        vctx = {**ctx, "variable": var.id}
        target = gdalio.mirror_path(src.uri, mirror)
        if target is None:
            report.warn("OBS-014", f"{src.uri} has no mirror location; preflight reads it directly", **vctx)
            outcomes.append((var.id, "skipped"))
            continue

        if target.is_file():
            checksum = gdalio.sha256_multihash(target)
            if src.checksum is None:
                report.warn("OBS-011", f"mirror copy has no declared checksum to match; its checksum is {checksum}", path=str(target), **vctx)
            elif checksum != src.checksum:
                report.add("OBS-011", f"mirror copy has checksum {checksum}, not the declared {src.checksum}", path=str(target), **vctx)
                outcomes.append((var.id, "mismatch"))
                continue
            outcomes.append((var.id, "present"))
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        try:
            checksum = gdalio.copy_with_multihash(remote(src.uri), partial)
        except Exception as exc:  # noqa: BLE001 - every source failure becomes a finding
            partial.unlink(missing_ok=True)
            report.add("OBS-014", f"cannot fetch {src.uri}: {exc}", **vctx)
            outcomes.append((var.id, "failed"))
            continue
        if src.checksum is not None and checksum != src.checksum:
            partial.unlink()
            report.add("OBS-011", f"fetched {src.uri} has checksum {checksum}, not the declared {src.checksum}", **vctx)
            outcomes.append((var.id, "mismatch"))
            continue
        partial.replace(target)
        if src.checksum is None:
            report.warn("OBS-011", f"fetched without a declared checksum; its checksum is {checksum}", path=str(target), **vctx)
        outcomes.append((var.id, "fetched"))
    return outcomes
