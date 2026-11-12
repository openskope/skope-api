"""Prepared sources: where fetch puts each output and its record (META-013, OBS-015).

A prepared source's GeoTIFF exists only in the mirror. Beside it, a record
says how it was made: the script's SHA-256, the toolchain, the inputs with
their checksums, and the output's checksum. Fetch reruns the script when the
record no longer matches; preflight and build refuse a stale one (OBS-016).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .models import Prepare

# The ingest image copies the prepare scripts to <root>/scripts/.
INGEST_ROOT = Path(os.environ.get("SKOPE_RELEASE_INGEST_ROOT", "/ingest"))
PYTHON_LOCK = INGEST_ROOT / "requirements.lock"


def output_path(mirror: Path, dataset_id: str, variable_id: str) -> Path:
    return mirror / "prepared" / dataset_id / f"{variable_id}.tif"


def record_path(output: Path) -> Path:
    return output.with_name(output.name + ".prepared.json")


def script_path(prepare: Prepare, root: Path = INGEST_ROOT) -> Path:
    return root / prepare.script


def sha256_hex(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def toolchain(lock: Path = PYTHON_LOCK) -> dict:
    """What besides the script can change its output: the GDAL base image and the locked Python packages.

    Rebuilding the image for a code change leaves both unchanged.
    """
    return {
        "gdal_image": os.environ.get("SKOPE_RELEASE_GDAL_IMAGE", "unknown"),
        "python_lock_sha256": sha256_hex(lock) if lock.is_file() else "unknown",
    }


def make_record(prepare: Prepare, script_sha256: str, tools: dict, inputs: list[dict], output_checksum: str) -> dict:
    return {
        "script": prepare.script,
        "script_sha256": script_sha256,
        "toolchain": tools,
        "inputs": inputs,
        "output_checksum": output_checksum,
    }


def read_record(output: Path) -> dict | None:
    try:
        return json.loads(record_path(output).read_text())
    except (OSError, ValueError):
        return None


def write_record(output: Path, record: dict) -> None:
    target = record_path(output)
    partial = target.with_name(target.name + ".part")
    partial.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    partial.replace(target)


def stale_reason(record: dict | None, prepare: Prepare, script_sha256: str, tools: dict) -> str | None:
    """Why a record no longer describes the declared preparation, or None when it does."""
    if record is None:
        return "there is no record"
    if record.get("script") != prepare.script or record.get("script_sha256") != script_sha256:
        return "the script changed"
    if record.get("toolchain") != tools:
        return "the toolchain changed"
    recorded = record.get("inputs") or []
    if [i.get("uri") for i in recorded] != [i.uri for i in prepare.inputs]:
        return "the declared inputs changed"
    for declared, made in zip(prepare.inputs, recorded):
        if declared.checksum is not None and declared.checksum != made.get("checksum"):
            return "an input's declared checksum changed"
    return None
