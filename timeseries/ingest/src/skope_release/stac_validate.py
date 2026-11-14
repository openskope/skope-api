"""Offline STAC validation of a staged or published release (VAL-005).

Schema validation uses the vendored pinned schemas only, then checks what a
schema cannot: links and relative paths, extension declarations, extents,
Item and asset membership, Band names, and SKOPE referential integrity.
"""

from __future__ import annotations

import json
import posixpath
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT7

from .documents import SCHEMA_DIR
from .findings import Report
from .stac_serializer import EXTENSIONS, SKOPE_SCHEMA_ID, STAC_VERSION

VENDOR = SCHEMA_DIR / "vendor"
CORE = {
    "Collection": "https://schemas.stacspec.org/v1.1.0/collection-spec/json-schema/collection.json",
    "Feature": "https://schemas.stacspec.org/v1.1.0/item-spec/json-schema/item.json",
}
SKOPE_SCHEMA = SCHEMA_DIR / "skope" / "v0.1.0" / "schema.json"


@lru_cache(maxsize=1)
def _registry() -> Registry:
    resources = []
    for path in VENDOR.rglob("*.json"):
        url = "https://" + path.relative_to(VENDOR).as_posix()
        contents = json.loads(path.read_text("utf-8"))
        resource = Resource.from_contents(contents, default_specification=DRAFT7)
        resources.append((url, resource))
        schema_id = contents.get("$id", "").split("#", 1)[0]
        if schema_id and schema_id != url:
            resources.append((schema_id, resource))
        if url.startswith("https://geojson.org"):  # referenced over plain HTTP by STAC
            resources.append(("http" + url[5:], resource))
    skope = json.loads(SKOPE_SCHEMA.read_text("utf-8"))
    resources.append((SKOPE_SCHEMA_ID, Resource.from_contents(skope, default_specification=DRAFT7)))
    return Registry().with_resources(resources)


def _schema(url: str) -> dict:
    return _registry().contents(url)


def _schema_errors(instance: dict, url: str) -> list[str]:
    validator = Draft7Validator(_schema(url), registry=_registry())
    return [
        f"{'/'.join(str(p) for p in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(instance), key=lambda e: list(e.absolute_path))
    ]


def _check_schemas(obj: dict, path: str, report: Report, ctx: dict) -> None:
    kind = obj.get("type")
    if kind not in CORE:
        report.add("STAC-001", f"unexpected STAC object type {kind!r}", path=path, **ctx)
        return
    if obj.get("stac_version") != STAC_VERSION:
        report.add("STAC-001", f"stac_version must be {STAC_VERSION}", path=path, **ctx)
    for message in _schema_errors(obj, CORE[kind]):
        report.add("VAL-005", f"STAC {STAC_VERSION}: {message}", path=path, **ctx)
    allowed = set(EXTENSIONS.values())
    for url in obj.get("stac_extensions", []):
        if url not in allowed:
            report.add("STAC-002", f"extension {url} is not a pinned version", path=path, **ctx)
            continue
        for message in _schema_errors(obj, url):
            report.add("VAL-005", f"{url}: {message}", path=path, **ctx)
    if SKOPE_SCHEMA_ID in obj.get("stac_extensions", []):
        report.add("SKOPE-005", "published STAC must not reference the SKOPE extension URI yet", path=path, **ctx)


def _walk_keys(node: Any):
    if isinstance(node, dict):
        for key, value in node.items():
            yield key
            yield from _walk_keys(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_keys(item)


def _check_links(obj: dict, rel_path: str, root: Path, report: Report, ctx: dict) -> None:
    base = posixpath.dirname(rel_path)
    targets = [(link.get("rel"), link.get("href", "")) for link in obj.get("links", [])]
    targets += [("asset", asset.get("href", "")) for asset in obj.get("assets", {}).values()]
    for rel, href in targets:
        if rel == "self":
            report.add("REL-002", "STAC objects must not carry a self link", path=rel_path, **ctx)
            continue
        if "://" in href:
            if not href.startswith("https://"):
                report.add("REL-002", f"external link {href} must be absolute HTTPS", path=rel_path, **ctx)
            continue
        if href.startswith("/"):
            report.add("ORG-010", f"{href} is an absolute path", path=rel_path, **ctx)
            continue
        target = posixpath.normpath(posixpath.join(base, href))
        if target.startswith(".."):
            report.add("OBS-010", f"{href} leaves the release", path=rel_path, **ctx)
        elif not (root / target).is_file():
            report.add("OBS-010", f"{href} does not resolve within the release", path=rel_path, **ctx)


def validate_stac(root: Path, report: Report) -> dict | None:
    """Validate the STAC tree of a release directory; returns the Collection."""
    ctx: dict = {}
    try:
        collection = json.loads((root / "collection.json").read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        report.add("STAC-001", f"collection.json is missing or unreadable: {exc}", path="collection.json")
        return None
    ctx = {"dataset": collection.get("id")}
    _check_schemas(collection, "collection.json", report, ctx)
    for message in _schema_errors(collection, SKOPE_SCHEMA_ID):
        report.add("SKOPE-001", message, path="collection.json", **ctx)
    _check_links(collection, "collection.json", root, report, ctx)
    for obj_keys in [list(_walk_keys(collection))]:
        if "proj:epsg" in obj_keys:
            report.add("STAC-007", "proj:epsg must not be emitted", path="collection.json", **ctx)
        if any(k.startswith("titiler:") for k in obj_keys):
            report.add("STAC-010", "titiler:* fields are not metadata authorities", path="collection.json", **ctx)

    temporal = "item_assets" in collection
    data_keys = set(collection.get("item_assets" if temporal else "assets", {}))
    skope_vars = set(collection.get("skope:variables", {}))
    if not skope_vars <= data_keys:
        report.add("SKOPE-002", f"skope:variables keys {sorted(skope_vars - data_keys)} are not data assets", path="collection.json", **ctx)
    cube_vars = set(collection.get("cube:variables", {}))
    if temporal and cube_vars != data_keys:
        report.add("STAC-003", "cube:variables must match item_assets", path="collection.json", **ctx)
    if not temporal and "cube:dimensions" in collection:
        report.add("ORG-001", "a static Collection carries no temporal datacube", path="collection.json", **ctx)

    item_links = [l["href"] for l in collection.get("links", []) if l.get("rel") == "item"]
    item_files = sorted(p.relative_to(root).as_posix() for p in (root / "items").glob("*.json")) if (root / "items").is_dir() else []
    if not temporal and item_files:
        report.add("STAC-012", "a static release has no Items", **ctx)
    linked = sorted(posixpath.normpath(h) for h in item_links)
    if temporal and linked != item_files:
        report.add("STAC-005", "Collection item links and items/ disagree", path="collection.json", **ctx)

    items = []
    for rel in item_files:
        item = json.loads((root / rel).read_text("utf-8"))
        ictx = {**ctx, "chunk": item.get("id")}
        _check_schemas(item, rel, report, ictx)
        _check_links(item, rel, root, report, ictx)
        if posixpath.basename(rel) != f"{item.get('id')}.json":
            report.add("ORG-008", "Item filename must equal its ID", path=rel, **ictx)
        if item.get("collection") != collection.get("id"):
            report.add("STAC-005", "Item names another Collection", path=rel, **ictx)
        if set(item.get("assets", {})) != data_keys:
            report.add("ORG-005", f"Item assets {sorted(item.get('assets', {}))} differ from {sorted(data_keys)}", path=rel, **ictx)
        if "proj:epsg" in set(_walk_keys(item)):
            report.add("STAC-007", "proj:epsg must not be emitted", path=rel, **ictx)
        items.append(item)

    if temporal:
        _check_axis(collection, items, report, ctx)
    _check_extents(collection, items, report, ctx)
    return collection


def _item_sort_key(item: dict) -> tuple:
    p = item["properties"]
    return (p.get("start_datetime") or p.get("datetime"), p.get("end_datetime") or p.get("datetime"), item["id"])


def _check_axis(collection: dict, items: list[dict], report: Report, ctx: dict) -> None:
    """STAC-011: Band names concatenate to the Collection's temporal axis."""
    values = collection.get("cube:dimensions", {}).get("time", {}).get("values")
    if not values:
        report.add("STAC-003", "cube:dimensions.time.values must enumerate the axis", path="collection.json", **ctx)
        return
    if len(values) != len(set(values)):
        report.add("STAC-003", "cube:dimensions.time.values must be unique", path="collection.json", **ctx)
    ordered = sorted(items, key=_item_sort_key)
    for variable in collection.get("item_assets", {}):
        names: list[str] = []
        for item in ordered:
            sequences = {k: [b.get("name") for b in a.get("bands", [])] for k, a in item["assets"].items()}
            if len({tuple(s) for s in sequences.values()}) > 1:
                report.add("STAC-011", "variable assets of one Item have different Band names", path=f"items/{item['id']}.json", **ctx)
            names.extend(sequences.get(variable, []))
        if names != list(values):
            report.add("STAC-011", f"Band names of {variable} do not concatenate to cube:dimensions.time.values", variable=variable, **ctx)


def _check_extents(collection: dict, items: list[dict], report: Report, ctx: dict) -> None:
    """STAC-004: the Collection extent is derived from its members."""
    extent = collection.get("extent", {})
    bbox = extent.get("spatial", {}).get("bbox", [[None]])[0]
    if items:
        boxes = [item["bbox"] for item in items]
        union = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
        if union != bbox:
            report.add("STAC-004", f"spatial extent {bbox} differs from the Items' union {union}", path="collection.json", **ctx)
        starts = [i["properties"].get("start_datetime") or i["properties"].get("datetime") for i in items]
        ends = [i["properties"].get("end_datetime") or i["properties"].get("datetime") for i in items]
        interval = extent.get("temporal", {}).get("interval", [[None, None]])[0]
        if interval != [min(starts), max(ends)]:
            report.add("STAC-004", f"temporal extent {interval} differs from the Items' {[min(starts), max(ends)]}", path="collection.json", **ctx)
