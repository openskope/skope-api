from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ManifestVariable:
    id: str
    uri: str


def load_input_manifest(
    manifest_path: str, expected_dataset_id: str
) -> list[ManifestVariable]:
    path = Path(manifest_path)
    if not path.is_file():
        raise ValueError(f"Input manifest not found at: {manifest_path}")

    with path.open(encoding="utf-8") as manifest_file:
        content = yaml.safe_load(manifest_file)

    if not isinstance(content, dict):
        raise ValueError("Input manifest must be a YAML mapping.")

    dataset_id = content.get("dataset_id")
    if dataset_id != expected_dataset_id:
        raise ValueError(
            f"Input manifest dataset_id '{dataset_id}' does not match "
            f"DATASET_NAME '{expected_dataset_id}'."
        )

    variables = content.get("variables")
    if not isinstance(variables, list) or not variables:
        raise ValueError("Input manifest must contain a non-empty variables list.")

    parsed: list[ManifestVariable] = []
    seen_ids: set[str] = set()
    for index, variable in enumerate(variables):
        if not isinstance(variable, dict):
            raise ValueError(f"Manifest variable at index {index} must be a mapping.")

        variable_id = variable.get("id")
        uri = variable.get("uri")
        if not isinstance(variable_id, str) or not variable_id.strip():
            raise ValueError(f"Manifest variable at index {index} has no valid id.")
        if variable_id in seen_ids:
            raise ValueError(f"Duplicate variable id in input manifest: {variable_id}")
        if not isinstance(uri, str) or not uri.strip():
            raise ValueError(f"Manifest variable '{variable_id}' has no valid uri.")
        if not uri.lower().endswith((".tif", ".tiff")):
            raise ValueError(
                f"Manifest variable '{variable_id}' must reference a TIFF: {uri}"
            )

        seen_ids.add(variable_id)
        parsed.append(ManifestVariable(id=variable_id, uri=uri))

    return parsed


def validate_manifest_metadata(
    variables: list[ManifestVariable], dataset_metadata: dict
) -> None:
    manifest_ids = {variable.id for variable in variables}
    metadata_variables = dataset_metadata.get("variables")
    if not isinstance(metadata_variables, list):
        raise ValueError("Dataset metadata must contain a variables list.")

    metadata_ids = {
        variable.get("id")
        for variable in metadata_variables
        if isinstance(variable, dict) and variable.get("id")
    }
    missing_from_metadata = sorted(manifest_ids - metadata_ids)
    missing_from_manifest = sorted(metadata_ids - manifest_ids)
    if missing_from_metadata or missing_from_manifest:
        details = []
        if missing_from_metadata:
            details.append(
                "missing from metadata.yml: " + ", ".join(missing_from_metadata)
            )
        if missing_from_manifest:
            details.append(
                "missing from input manifest: " + ", ".join(missing_from_manifest)
            )
        raise ValueError("Manifest/metadata variable mismatch: " + "; ".join(details))
