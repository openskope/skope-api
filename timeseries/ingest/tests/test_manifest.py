import pytest
import yaml

from cog_stac_pipeline.manifest import (
    ManifestVariable,
    load_input_manifest,
    validate_manifest_metadata,
)


def write_manifest(tmp_path, content):
    path = tmp_path / "manifest.yml"
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_load_input_manifest_maps_ids_to_arbitrary_tiff_uris(tmp_path):
    manifest_path = write_manifest(
        tmp_path,
        """
dataset_id: paleocar_v3
variables:
  - id: ppt_annual
    uri: s3://skope/paleocar_v3/ppt_annual/prediction_scaled.tif
  - id: local_variable
    uri: /data/source/cube.tiff
""",
    )

    variables = load_input_manifest(manifest_path, "paleocar_v3")

    assert [(variable.id, variable.uri) for variable in variables] == [
        (
            "ppt_annual",
            "s3://skope/paleocar_v3/ppt_annual/prediction_scaled.tif",
        ),
        ("local_variable", "/data/source/cube.tiff"),
    ]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("dataset_id: wrong\nvariables: []\n", "does not match"),
        ("dataset_id: paleocar_v3\nvariables: []\n", "non-empty"),
        (
            """
dataset_id: paleocar_v3
variables:
  - id: ppt
    uri: /data/ppt.tif
  - id: ppt
    uri: /data/other.tif
""",
            "Duplicate variable id",
        ),
        (
            """
dataset_id: paleocar_v3
variables:
  - id: ppt
    uri: /data/ppt.nc
""",
            "must reference a TIFF",
        ),
    ],
)
def test_load_input_manifest_rejects_invalid_contract(tmp_path, content, message):
    manifest_path = write_manifest(tmp_path, content)

    with pytest.raises(ValueError, match=message):
        load_input_manifest(manifest_path, "paleocar_v3")


def test_validate_manifest_metadata_accepts_exact_variable_set():
    variables = [
        ManifestVariable(id="ppt", uri="/data/ppt.tif"),
        ManifestVariable(id="gdd", uri="/data/gdd.tif"),
    ]

    validate_manifest_metadata(
        variables,
        {"variables": [{"id": "gdd"}, {"id": "ppt"}]},
    )


def test_validate_manifest_metadata_reports_both_sides_of_mismatch():
    variables = [
        ManifestVariable(id="ppt", uri="/data/ppt.tif"),
        ManifestVariable(id="manifest_only", uri="/data/manifest.tif"),
    ]

    with pytest.raises(ValueError, match="missing from metadata.yml: manifest_only"):
        validate_manifest_metadata(
            variables,
            {"variables": [{"id": "ppt"}, {"id": "metadata_only"}]},
        )


@pytest.mark.parametrize(
    "dataset_id",
    ["lbda_v2", "paleocar_v2", "paleocar_v3", "prism", "srtm"],
)
def test_checked_in_manifests_match_migration_metadata(dataset_id):
    manifest_subdir = "" if dataset_id == "paleocar_v3" else "legacy/"
    manifest_path = f"manifests/{manifest_subdir}{dataset_id}.yml"
    metadata = yaml.safe_load(open("legacy-metadata.yml", encoding="utf-8"))
    dataset_metadata = next(item for item in metadata if item["id"] == dataset_id)

    variables = load_input_manifest(manifest_path, dataset_id)

    validate_manifest_metadata(variables, dataset_metadata)
