"""Reading and writing release documents deterministically.

- YAML authoring files are parsed without implicit typing of timestamps and
  with duplicate keys rejected, then validated through their strict models in
  JSON mode, so an unquoted `0103` (an integer in YAML) fails rather than being
  coerced (META-005).
- JSON outputs are UTF-8, key-sorted, two-space indented, LF-terminated, with no
  non-finite numbers (REL-005).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from .findings import Report
from .models import DOCUMENT_MODELS

SCHEMA_DIR = Path(__file__).parent / "schemas"
RELEASE_SCHEMA_DIR = SCHEMA_DIR / "release"

M = TypeVar("M", bound=BaseModel)


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader without timestamp coercion and with duplicate-key detection."""


_StrictLoader.yaml_implicit_resolvers = {
    key: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:timestamp"]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def _construct_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key {key!r}", key_node.start_mark
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return yaml.load(handle, Loader=_StrictLoader)  # noqa: S506 - SafeLoader subclass


def validate(model: type[M], data: Any, *, requirement: str, path: Path | str, report: Report, **context) -> M | None:
    """Validate `data` against `model` in strict JSON mode, recording failures."""
    try:
        return model.model_validate_json(json.dumps(data, allow_nan=False))
    except ValidationError as exc:
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"]) or "<document>"
            report.add(requirement, f"{location}: {error['msg']}", path=str(path), **context)
    except (TypeError, ValueError) as exc:
        report.add(requirement, f"not representable as JSON: {exc}", path=str(path), **context)
    return None


def load_document(model: type[M], path: Path, *, requirement: str, report: Report, **context) -> M | None:
    if not path.is_file():
        report.add(requirement, "file not found", path=str(path), **context)
        return None
    try:
        data = load_yaml(path) if path.suffix in {".yml", ".yaml"} else json.loads(path.read_text("utf-8"))
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        report.add(requirement, f"cannot parse: {exc}", path=str(path), **context)
        return None
    return validate(model, data, requirement=requirement, path=path, report=report, **context)


def _check_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite number in JSON output (REL-005)")
    if isinstance(value, dict):
        for item in value.values():
            _check_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _check_finite(item)


def dumps_json(value: Any) -> bytes:
    """Deterministic JSON bytes (REL-005)."""
    _check_finite(value)
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return (text + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# Committed JSON Schemas (Section 20.3: generated from the models)

SCHEMA_VERSIONS = {
    "curated": "0.1.0",
    "source-manifest": "0.1.0",
    "release-ledger": "0.1.0",
    "release-manifest": "1.0.0",
    "release-overview": "1.0.0",
}


def generated_schemas() -> dict[str, bytes]:
    out = {}
    for name, model in DOCUMENT_MODELS.items():
        version = SCHEMA_VERSIONS[name]
        schema = model.model_json_schema(mode="validation")
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": f"https://api.openskope.org/schemas/release/{name}/v{version}/schema.json",
            **schema,
        }
        out[f"{name}.schema.json"] = dumps_json(schema)
    return out


def export_schemas(directory: Path = RELEASE_SCHEMA_DIR) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for filename, content in generated_schemas().items():
        target = directory / filename
        target.write_bytes(content)
        written.append(target)
    return written


if __name__ == "__main__":
    for written in export_schemas():
        print(written)
