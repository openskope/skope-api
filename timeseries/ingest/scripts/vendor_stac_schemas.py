"""Vendor the pinned STAC and extension JSON Schemas for offline validation (VAL-005).

Run once per pinned-version change; the result is committed. Every `$ref` is
followed recursively and stored at `vendor/<host>/<path>`.

    python3 scripts/vendor_stac_schemas.py
"""

import json
import pathlib
import urllib.parse
import urllib.request

ROOTS = [
    "https://schemas.stacspec.org/v1.1.0/collection-spec/json-schema/collection.json",
    "https://schemas.stacspec.org/v1.1.0/item-spec/json-schema/item.json",
    "https://stac-extensions.github.io/projection/v2.0.0/schema.json",
    "https://stac-extensions.github.io/file/v2.1.0/schema.json",
    "https://stac-extensions.github.io/scientific/v1.0.0/schema.json",
    "https://stac-extensions.github.io/raster/v2.0.0/schema.json",
    "https://stac-extensions.github.io/datacube/v2.3.0/schema.json",
    "https://stac-extensions.github.io/version/v1.2.0/schema.json",
    "https://stac-extensions.github.io/processing/v1.2.0/schema.json",
]
OUT = pathlib.Path(__file__).resolve().parents[1] / "src/skope_release/schemas/vendor"


def refs(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "$ref" and isinstance(value, str):
                yield value
            else:
                yield from refs(value)
    elif isinstance(node, list):
        for item in node:
            yield from refs(item)


def local_path(url: str) -> pathlib.Path:
    parsed = urllib.parse.urlparse(url)
    return OUT / parsed.netloc / parsed.path.lstrip("/")


def main() -> None:
    seen: set[str] = set()
    queue = list(ROOTS)
    while queue:
        url = queue.pop()
        if url in seen:
            continue
        seen.add(url)
        request = urllib.request.Request(url, headers={"User-Agent": "skope-release-schema-vendor"})
        try:
            with urllib.request.urlopen(request) as response:
                doc = json.load(response)
        except Exception as exc:
            raise SystemExit(f"cannot fetch {url}: {exc}") from exc
        path = local_path(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        for ref in refs(doc):
            target = urllib.parse.urljoin(url, ref).split("#", 1)[0]
            if target and target not in seen:
                queue.append(target)
    print(f"vendored {len(seen)} schemas into {OUT}")


if __name__ == "__main__":
    main()
