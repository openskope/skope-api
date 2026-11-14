"""Cross-artifact agreement (VAL-007, AUTH-001, AUTH-002).

Compares the serialized STAC with the byte observation it was generated from,
so a serializer bug cannot publish a fact that disagrees with the COG bytes.
"""

from __future__ import annotations

import json
import posixpath
from pathlib import Path

from .findings import Report
from .plan import FinalObservation, require_observation


def check_stac_against_bytes(observation: object, root: Path, report: Report) -> None:
    obs = require_observation(observation)
    collection = json.loads((root / "collection.json").read_text("utf-8"))
    ctx = {"dataset": obs.dataset_id}
    if obs.temporal:
        documents = {
            f"items/{p.name}": json.loads(p.read_text("utf-8")) for p in sorted((root / "items").glob("*.json"))
        }
        assets = [
            (rel, vid, asset) for rel, item in documents.items() for vid, asset in item.get("assets", {}).items()
        ]
    else:
        assets = [("collection.json", vid, asset) for vid, asset in collection.get("assets", {}).items()]

    observed = {a.path: a for a in obs.assets}
    seen = set()
    for rel, vid, asset in assets:
        path = posixpath.normpath(posixpath.join(posixpath.dirname(rel), asset["href"]))
        actx = {**ctx, "variable": vid, "path": path}
        fact = observed.get(path)
        if fact is None:
            report.add("VAL-007", "STAC references a COG that was not observed", **actx)
            continue
        seen.add(path)
        expected = {
            "file:size": fact.size,
            "file:checksum": fact.checksum,
            "proj:code": fact.crs_code,
            "proj:shape": list(fact.shape),
            "proj:transform": list(fact.transform),
            "data_type": fact.data_type,
            "raster:scale": fact.scale,
            "raster:offset": fact.offset,
        }
        if fact.nodata is not None:
            expected["nodata"] = fact.nodata
        for key, value in expected.items():
            if asset.get(key) != value:
                report.add("AUTH-002", f"{key} is {asset.get(key)!r} in STAC but {value!r} in the bytes", **actx)
        stac_bands = [(b["name"], b["statistics"]) for b in asset.get("bands", [])]
        byte_bands = [(b.name, b.statistics) for b in fact.bands]
        if stac_bands != byte_bands:
            report.add("STAC-008", "Band names or statistics in STAC differ from the bytes", **actx)
    for missing in sorted(set(observed) - seen):
        report.add("VAL-007", "an observed COG is not referenced by STAC", path=missing, **ctx)

    if obs.temporal:
        values = collection.get("cube:dimensions", {}).get("time", {}).get("values")
        if values != list(obs.axis.keys()):
            report.add("VAL-007", "cube:dimensions.time.values differs from the observed axis", path="collection.json", **ctx)
