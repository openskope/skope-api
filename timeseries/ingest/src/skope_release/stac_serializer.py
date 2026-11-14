"""Serialize STAC from a `FinalObservation` (Section 9, Section 10).

The serializer reads only the final observation: byte facts come from the
inspected COGs and descriptive facts from the curated declaration it carries.
Links inside the release are relative and no object has a `self` link (REL-002).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__
from .documents import dumps_json
from .plan import AssetObservation, FinalObservation, require_observation
from .timeaxis import end_of_period, parse_key

STAC_VERSION = "1.1.0"
COG_MEDIA_TYPE = "image/tiff; application=geotiff; profile=cloud-optimized"  # STAC-006
EXTENSIONS = {  # Section 9.1: pinned versions only (STAC-002)
    "proj": "https://stac-extensions.github.io/projection/v2.0.0/schema.json",
    "file": "https://stac-extensions.github.io/file/v2.1.0/schema.json",
    "sci": "https://stac-extensions.github.io/scientific/v1.0.0/schema.json",
    "raster": "https://stac-extensions.github.io/raster/v2.0.0/schema.json",
    "cube": "https://stac-extensions.github.io/datacube/v2.3.0/schema.json",
    "version": "https://stac-extensions.github.io/version/v1.2.0/schema.json",
    "processing": "https://stac-extensions.github.io/processing/v1.2.0/schema.json",
}
# The SKOPE extension is validated locally and not referenced until its URI
# serves the approved schema (SKOPE-005).
SKOPE_SCHEMA_ID = "https://api.openskope.org/schemas/stac/v0.1.0/schema.json"


def rfc3339(instant: datetime) -> str:
    """RFC 3339 UTC with a four-digit year, including years before 1000."""
    return (
        f"{instant.year:04d}-{instant.month:02d}-{instant.day:02d}"
        f"T{instant.hour:02d}:{instant.minute:02d}:{instant.second:02d}Z"
    )


def _used_extensions(obj: dict[str, Any]) -> list[str]:
    prefixes: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if ":" in key:
                    prefixes.add(key.split(":", 1)[0])
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk({k: v for k, v in obj.items() if k != "links"})
    if "version" in obj:
        prefixes.add("version")
    return sorted(EXTENSIONS[p] for p in prefixes if p in EXTENSIONS)


def _period(obs: FinalObservation, first_key: str, last_key: str) -> tuple[datetime, datetime]:
    """Inclusive start and end instants of a run of timesteps (ORG-004, Section 20.3)."""
    temporal = obs.curated.temporal
    start = parse_key(first_key, obs.axis.precision)
    last = parse_key(last_key, obs.axis.precision)
    if temporal.timestep_meaning == "aggregation_period" and obs.axis.step:
        return start, end_of_period(last, obs.axis.step)
    return start, last


def _asset_common(obs: FinalObservation, variable_id: str) -> dict[str, Any]:
    var = next(v for v in obs.curated.variables if v.id == variable_id)
    return {"type": COG_MEDIA_TYPE, "roles": ["data"], "title": var.title, "description": var.description}


def _asset_encoding(asset: AssetObservation) -> dict[str, Any]:
    """Band properties that do not vary are serialized once on the asset (STAC-008)."""
    fields: dict[str, Any] = {
        "data_type": asset.data_type,
        "raster:scale": asset.scale,
        "raster:offset": asset.offset,
    }
    if asset.nodata is not None:
        fields["nodata"] = asset.nodata
    if asset.unit:
        fields["unit"] = asset.unit
    return fields


def _asset(obs: FinalObservation, asset: AssetObservation, href: str) -> dict[str, Any]:
    return {
        "href": href,
        **_asset_common(obs, asset.variable_id),
        **_asset_encoding(asset),
        "proj:code": asset.crs_code,  # STAC-007: from the inspected bytes
        "proj:shape": list(asset.shape),
        "proj:transform": list(asset.transform),
        "file:size": asset.size,
        "file:checksum": asset.checksum,
        "bands": [
            {"name": band.name, "statistics": band.statistics}  # STAC-011, STAC-008
            for band in asset.bands
        ],
    }


def _footprint(bbox: tuple[float, float, float, float]) -> dict[str, Any]:
    west, south, east, north = bbox
    return {
        "type": "Polygon",
        "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
    }


def build_items(obs: FinalObservation) -> list[dict[str, Any]]:
    items = []
    layout = obs.layout
    for chunk in range(layout.chunk_count):
        keys = layout.chunk_keys(chunk)
        start, end = _period(obs, keys[0], keys[-1])
        properties: dict[str, Any] = {"datetime": None, "start_datetime": rfc3339(start), "end_datetime": rfc3339(end)}
        if start == end:
            properties = {"datetime": rfc3339(start)}
        item = {
            "type": "Feature",
            "stac_version": STAC_VERSION,
            "id": layout.item_id(chunk),
            "collection": obs.dataset_id,
            "geometry": _footprint(obs.bbox_wgs84),
            "bbox": list(obs.bbox_wgs84),
            "properties": properties,
            "links": [
                {"rel": "collection", "href": "../collection.json", "type": "application/json"},
                {"rel": "parent", "href": "../collection.json", "type": "application/json"},
                {"rel": "root", "href": "../collection.json", "type": "application/json"},
            ],
            "assets": {
                variable.id: _asset(obs, obs.asset(variable.id, chunk), f"../{layout.cog_path(variable.id, chunk)}")
                for variable in sorted(obs.curated.variables, key=lambda v: v.id)
            },
        }
        item["stac_extensions"] = _used_extensions(item)
        items.append(item)
    return items


def build_collection(obs: FinalObservation) -> dict[str, Any]:
    cur = obs.curated
    variables = sorted(cur.variables, key=lambda v: v.id)
    providers = []
    for p in cur.providers:
        provider: dict[str, Any] = {"name": p.name, "roles": list(p.roles)}
        url = p.url or p.ror  # META-010: a ROR URI may be the Provider url
        if url:
            provider["url"] = url
        if p.description:
            provider["description"] = p.description
        providers.append(provider)

    links: list[dict[str, Any]] = [{"rel": "root", "href": "./collection.json", "type": "application/json"}]
    if cur.license_href:
        links.append({"rel": "license", "href": cur.license_href})
    if cur.doi:
        links.append({"rel": "cite-as", "href": f"https://doi.org/{cur.doi}"})
    for person in cur.people:
        if person.orcid:
            links.append({"rel": "author", "href": person.orcid, "title": person.name})
    if cur.contact_href:
        links.append({"rel": "about", "href": cur.contact_href, "title": "Contact"})
    for link in cur.links:
        links.append({k: v for k, v in link.model_dump().items() if v is not None})

    collection: dict[str, Any] = {
        "type": "Collection",
        "stac_version": STAC_VERSION,
        "id": cur.id,
        "title": cur.title,
        "description": cur.description,
        "license": cur.license,
        "version": cur.version,
        "providers": providers,
        "skope:region_name": cur.region_name,
    }
    # Processing 1.2.0 places its fields on a processor (or producer) provider.
    processing = {"processing:software": {"skope-release": __version__, "GDAL": obs.gdal_version}}
    if cur.lineage:
        processing["processing:lineage"] = cur.lineage
    target = next((p for p in providers if "processor" in p["roles"]), None)
    target = target or next((p for p in providers if "producer" in p["roles"]), providers[0])
    target.update(processing)
    if cur.citation:
        collection["sci:citation"] = cur.citation
    if cur.doi:
        collection["sci:doi"] = cur.doi
    if cur.publications:
        collection["sci:publications"] = [p.model_dump(exclude_none=True) for p in cur.publications]
    if cur.uncertainty:
        collection["skope:uncertainty"] = {
            "summary": cur.uncertainty.summary,
            "methodology_href": cur.uncertainty.methodology_href,
        }
    skope_vars = {}
    for v in variables:
        entry: dict[str, Any] = {}
        if v.category:
            entry["category"] = v.category
        if v.categories:
            entry["categories"] = {str(k): label for k, label in sorted(v.categories.items())}
        if entry:
            skope_vars[v.id] = entry
    if skope_vars:
        collection["skope:variables"] = skope_vars

    bbox = list(obs.bbox_wgs84)
    if obs.temporal:
        keys = obs.axis.keys()
        start, end = _period(obs, keys[0], keys[-1])
        t = cur.temporal
        collection["skope:temporal"] = {
            "calendar": t.calendar,
            "precision": t.precision,
            "timestep_meaning": t.timestep_meaning,
            "endpoint_inclusion": t.endpoint_inclusion,
            "description": t.description,
        }
        collection["extent"] = {"spatial": {"bbox": [bbox]}, "temporal": {"interval": [[rfc3339(start), rfc3339(end)]]}}
        a, _, c, _, e, f = obs.transform
        height, width = obs.shape
        epsg = int(obs.crs_code.split(":")[1]) if obs.crs_code.startswith("EPSG:") else obs.crs_code
        time_dimension: dict[str, Any] = {"type": "temporal", "extent": [rfc3339(start), rfc3339(end)], "values": list(keys)}
        if obs.axis.step:
            time_dimension["step"] = obs.axis.step
        collection["cube:dimensions"] = {
            "x": {"type": "spatial", "axis": "x", "extent": [c, c + a * width], "step": a, "reference_system": epsg},
            "y": {"type": "spatial", "axis": "y", "extent": [f + e * height, f], "step": e, "reference_system": epsg},
            "time": time_dimension,
        }
        collection["cube:variables"] = {
            v.id: {"dimensions": ["time", "y", "x"], "type": "data", "description": v.description, "unit": v.unit}
            for v in variables
        }
        first_chunk = {v.id: obs.asset(v.id, 0) for v in variables}
        collection["item_assets"] = {
            v.id: {**_asset_common(obs, v.id), **_asset_encoding(first_chunk[v.id])} for v in variables
        }
        for chunk in range(obs.layout.chunk_count):
            links.append({"rel": "item", "href": f"./items/{obs.layout.item_id(chunk)}.json", "type": "application/geo+json"})
    else:
        interval = list(cur.temporal_extent) if cur.temporal_extent else [None, None]  # STAC-004
        collection["extent"] = {"spatial": {"bbox": [bbox]}, "temporal": {"interval": [interval]}}
        collection["assets"] = {
            v.id: _asset(obs, obs.asset(v.id, None), f"./cogs/{v.id}.tif") for v in variables  # STAC-012
        }
    collection["links"] = links
    collection["stac_extensions"] = _used_extensions(collection)
    return collection


def write_stac(observation: object, staging: Path) -> list[Path]:
    """Write `collection.json` and Items into staging (TXN-001 step 4)."""
    obs = require_observation(observation)
    written = []
    collection = build_collection(obs)
    target = staging / "collection.json"
    target.write_bytes(dumps_json(collection))
    written.append(target)
    if obs.temporal:
        (staging / "items").mkdir(exist_ok=True)
        for item in build_items(obs):
            target = staging / "items" / f"{item['id']}.json"
            target.write_bytes(dumps_json(item))
            written.append(target)
    return written
