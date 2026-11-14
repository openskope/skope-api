"""Generate `overview.yml` from a release's validated STAC (Section 15).

The generator reads only `collection.json` and `items/` (API-001); the release
ID and declaration digest are passed in because they identify the release
rather than describe the data (API-008). The time-to-band rule is verified
against the STAC Band names and the COG band counts before the overview is
accepted (API-002, API-004).
"""

from __future__ import annotations

import json
import posixpath
from pathlib import Path
from typing import Any

from osgeo import gdal

from .documents import validate
from .findings import Report
from .models import Overview
from .timeaxis import Axis, AxisError, ChunkLayout

SCHEMA_VERSION = "1.0.0"
HEADER = (
    "# Generated from validated STAC. Do not edit.\n"
    "# Authority: collection.json{items}. Regenerated and byte-compared on build.\n"
)
STRUCTURAL_RELS = {"self", "root", "parent", "child", "collection", "item", "license"}
FLOW_KEYS = {"bbox", "shape", "transform", "values", "roles"}
WIDTH = 100


# ---------------------------------------------------------------------------
# Deterministic YAML emitter (REL-005, Section 20.3)


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite number in overview (REL-005)")
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)  # a valid YAML double-quoted scalar
    raise TypeError(f"cannot emit {type(value).__name__}")


def _flow(items: list[Any], indent: int, prefix_len: int) -> str:
    parts = [_scalar(v) for v in items]
    single = "[" + ", ".join(parts) + "]"
    if prefix_len + len(single) <= WIDTH:
        return single
    lines, line = [], "["
    pad = " " * (indent + 2)
    for i, part in enumerate(parts):
        piece = part + ("," if i < len(parts) - 1 else "]")
        candidate = line + ("" if line in ("[", pad) else " ") + piece
        limit = WIDTH - (prefix_len if not lines else 0)  # the first line follows its key
        if len(candidate) > limit and line not in ("[", pad):
            lines.append(line)
            line = pad + piece
        else:
            line = candidate
    lines.append(line)
    return "\n".join(lines)


def emit(value: Any, indent: int = 0) -> str:
    pad = " " * indent
    out: list[str] = []
    for key, item in value.items():
        label = f"{pad}{key}:"
        if isinstance(item, dict):
            if item:
                out.append(label)
                out.append(emit(item, indent + 2))
            else:
                out.append(f"{label} {{}}")
        elif isinstance(item, (list, tuple)):
            if key in FLOW_KEYS or not item:
                out.append(f"{label} {_flow(list(item), indent, len(label) + 1)}")
            else:
                out.append(label)
                for element in item:
                    if isinstance(element, dict):
                        block = emit(element, indent + 4).splitlines()
                        out.append(f"{pad}  - {block[0].lstrip()}")
                        out.extend(block[1:])
                    else:
                        out.append(f"{pad}  - {_scalar(element)}")
        else:
            out.append(f"{label} {_scalar(item)}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Generation


def _external_links(links: list[dict]) -> list[dict]:
    out = []
    for link in links:
        if link.get("rel") in STRUCTURAL_RELS or not link.get("href", "").startswith("https://"):
            continue
        entry = {"rel": link["rel"], "href": link["href"]}
        if link.get("title"):
            entry["title"] = link["title"]
        out.append(entry)
    return out


def _variables(collection: dict, temporal: bool) -> dict[str, dict]:
    source = collection["item_assets"] if temporal else collection["assets"]
    skope_vars = collection.get("skope:variables", {})
    out = {}
    for vid in sorted(source):
        asset = source[vid]
        entry: dict[str, Any] = {"title": asset["title"], "description": asset["description"], "unit": asset.get("unit", "unitless")}
        meta = skope_vars.get(vid, {})
        if "category" in meta:
            entry["category"] = meta["category"]
        if "categories" in meta:
            entry["categories"] = {int(k): v for k, v in sorted(meta["categories"].items(), key=lambda kv: int(kv[0]))}
        entry["data_type"] = asset["data_type"]
        entry["nodata"] = asset.get("nodata")
        entry["scale"] = asset.get("raster:scale", 1.0)
        entry["offset"] = asset.get("raster:offset", 0.0)
        if not temporal:
            entry["asset_href"] = posixpath.normpath(asset["href"])
            entry["band_name"] = asset["bands"][0]["name"]
        out[vid] = entry
    return out


def _load_items(root: Path) -> list[dict]:
    items_dir = root / "items"
    if not items_dir.is_dir():
        return []
    items = [json.loads(p.read_text("utf-8")) for p in sorted(items_dir.glob("*.json"))]
    return sorted(items, key=lambda i: (i["properties"].get("start_datetime") or i["properties"]["datetime"], i["id"]))


def build_overview(root: Path, *, release_id: str, declaration_digest: str, report: Report) -> dict | None:
    collection = json.loads((root / "collection.json").read_text("utf-8"))
    items = _load_items(root)
    temporal = "item_assets" in collection
    ctx = {"dataset": collection["id"]}

    if temporal:
        first_asset = next(iter(items[0]["assets"].values()))
    else:
        first_asset = next(iter(collection["assets"].values()))
    dataset: dict[str, Any] = {
        "id": collection["id"],
        "version": collection["version"],
        "title": collection["title"],
        "description": collection["description"],
        "profile": "TemporalCubeDataset" if temporal else "StaticRasterDataset",
        "license": collection["license"],
        "region_name": collection["skope:region_name"],
        "extent": {"bbox": collection["extent"]["spatial"]["bbox"][0]},
        "grid": {
            "code": first_asset["proj:code"],
            "shape": first_asset["proj:shape"],
            "transform": first_asset["proj:transform"][:6],
        },
    }

    if temporal:
        time = collection["cube:dimensions"]["time"]
        values = time["values"]
        semantics = collection["skope:temporal"]
        precision = semantics["precision"]
        chunk_size = len(next(iter(items[0]["assets"].values()))["bands"])
        axis: Axis | None = None
        if "step" in time:
            try:
                candidate = Axis.regular(values[0], time["step"], values[-1], precision)
                if candidate.keys() == tuple(values):
                    axis = candidate
            except AxisError:
                axis = None
        if axis is None:
            axis = Axis.enumerated(values, precision)
        dataset["time"] = {
            "kind": axis.kind,
            **({"origin": axis.origin, "step": axis.step} if axis.kind == "regular" else {"values": list(values)}),
            "count": axis.count,
            "chunk_size": chunk_size,
            "precision": precision,
            "calendar": semantics["calendar"],
            "timestep_meaning": semantics["timestep_meaning"],
            "endpoint_inclusion": semantics["endpoint_inclusion"],
        }
        verify_rule(root, collection["id"], axis, chunk_size, items, list(collection["item_assets"]), report)

    dataset["providers"] = [
        {k: p[k] for k in ("name", "roles", "url") if k in p} for p in collection.get("providers", [])
    ]
    for key, field in (("citation", "sci:citation"), ("doi", "sci:doi")):
        if field in collection:
            dataset[key] = collection[field]
    lineage = [p["processing:lineage"] for p in collection.get("providers", []) if "processing:lineage" in p]
    if lineage:
        dataset["lineage"] = lineage[0]
    if "sci:publications" in collection:
        dataset["publications"] = collection["sci:publications"]
    if "skope:uncertainty" in collection:
        uncertainty = collection["skope:uncertainty"]
        dataset["uncertainty"] = {"summary": uncertainty["summary"], "methodology_href": uncertainty["methodology_href"]}
    dataset["links"] = _external_links(collection.get("links", []))

    overview = {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "declaration_digest": declaration_digest,
        "dataset": dataset,
        "variables": _variables(collection, temporal),
    }
    if validate(Overview, overview, requirement="API-010", path="overview.yml", report=report, **ctx) is None:
        return None
    return overview


def verify_rule(root: Path, dataset_id: str, axis: Axis, chunk_size: int, items: list[dict], variables: list[str], report: Report) -> None:
    """The rule must reproduce STAC Band names, asset paths, and COG band counts (API-002, API-004)."""
    layout = ChunkLayout(dataset_id, axis, chunk_size)
    if len(items) != layout.chunk_count:
        report.add("API-004", f"{len(items)} Items; the rule derives {layout.chunk_count} chunks", dataset=dataset_id)
        return
    for chunk, item in enumerate(items):
        if item["id"] != layout.item_id(chunk):
            report.add("API-004", f"Item {item['id']} should be {layout.item_id(chunk)}", dataset=dataset_id)
        for variable in variables:
            asset = item["assets"].get(variable)
            ctx = {"dataset": dataset_id, "variable": variable, "chunk": item["id"]}
            if asset is None:
                report.add("API-004", "asset missing", **ctx)
                continue
            path = posixpath.normpath(posixpath.join("items", asset["href"]))
            if path != layout.cog_path(variable, chunk):
                report.add("API-004", f"asset path {path} differs from the rule's {layout.cog_path(variable, chunk)}", **ctx)
            names = [b["name"] for b in asset.get("bands", [])]
            if tuple(names) != layout.chunk_keys(chunk):
                report.add("API-004", "Band names differ from the rule's timesteps", **ctx)
            try:
                with gdal.Open(str(root / path)) as ds:
                    if ds.RasterCount != len(names):
                        report.add("API-004", f"COG has {ds.RasterCount} bands; STAC lists {len(names)}", **ctx)
            except RuntimeError as exc:
                report.add("API-004", f"cannot open {path}: {exc}", **ctx)


def render(overview: dict, temporal: bool) -> bytes:
    header = HEADER.format(items=" and items/" if temporal else "")
    return (header + emit(overview) + "\n").encode("utf-8")


def write_overview(root: Path, *, release_id: str, declaration_digest: str, report: Report) -> Path | None:
    """Generate the overview as the build's final derived step (API-005)."""
    overview = build_overview(root, release_id=release_id, declaration_digest=declaration_digest, report=report)
    if overview is None or not report.ok:
        return None
    target = root / "overview.yml"
    target.write_bytes(render(overview, overview["dataset"]["profile"] == "TemporalCubeDataset"))
    return target
