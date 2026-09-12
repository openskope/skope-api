#!/bin/sh
set -eu

if [ "$#" -ne 3 ]; then
    echo "usage: $0 LEGACY_DATA_ROOT MIGRATED_DATA_ROOT MIGRATION_SCRATCH_ROOT" >&2
    exit 2
fi

legacy_data_root=$1
migrated_data_root=$2
migration_scratch_root=$3
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(dirname -- "$script_dir")

if [ ! -d "$legacy_data_root/datasets" ]; then
    echo "legacy dataset directory not found: $legacy_data_root/datasets" >&2
    exit 2
fi

if [ "${MIGRATION_PREFLIGHT_ONLY:-false}" != true ] && \
    [ -e "$migrated_data_root" ] && \
    [ -n "$(find "$migrated_data_root" -mindepth 1 -print -quit)" ]; then
    echo "migration output must be absent or empty: $migrated_data_root" >&2
    exit 2
fi

required_inputs='
datasets/lbda_v2/pmdi/cube.tif
datasets/paleocar_v2/gdd_may_sept/cube.tif
datasets/paleocar_v2/maize_farming_niche/cube.tif
datasets/paleocar_v2/ppt_water_year/cube.tif
datasets/prism/ppt/cube.tif
datasets/prism/tmax/cube.tif
datasets/prism/tmin/cube.tif
datasets/srtm/srtm_elevation/cube.tif
'

echo "$required_inputs" | while IFS= read -r relative_path; do
    [ -z "$relative_path" ] && continue
    if [ ! -r "$legacy_data_root/$relative_path" ]; then
        echo "required legacy input is missing or unreadable: $legacy_data_root/$relative_path" >&2
        exit 2
    fi
done

legacy_data_root=$(CDPATH= cd -- "$legacy_data_root" && pwd)

compose() {
    docker compose \
        --project-name skope-api-migration \
        --project-directory "$repository_root" \
        -f "$repository_root/deploy/compose/base.yml" \
        -f "$repository_root/deploy/compose/dev.yml" \
        "$@"
}

run_dataset() {
    dataset_id=$1
    manifest_path=$2
    start_datetime=$3
    time_delta=$4
    truncate_to_uint16=$5
    preflight_only=$6

    if [ "$preflight_only" = true ]; then
        echo "Preflighting $dataset_id"
    else
        echo "Migrating $dataset_id"
    fi

    if [ "$preflight_only" = true ]; then
        compose --profile ingest run --rm --no-deps \
            -e "DATASET_NAME=$dataset_id" \
            -e "DATASET_START_DATETIME=$start_datetime" \
            -e "DATASET_TIME_DELTA=$time_delta" \
            -e "INPUT_MANIFEST_PATH=$manifest_path" \
            -e METADATA_FILE_PATH=/ingest/legacy-metadata.yml \
            -e PREFLIGHT_ONLY=true \
            -v "$legacy_data_root:/legacy:ro" \
            ingest
        return
    fi

    compose --profile ingest run --rm --no-deps \
        -e "DATASET_NAME=$dataset_id" \
        -e "DATASET_START_DATETIME=$start_datetime" \
        -e "DATASET_TIME_DELTA=$time_delta" \
        -e "INPUT_MANIFEST_PATH=$manifest_path" \
        -e METADATA_FILE_PATH=/migration/metadata.yml \
        -e "OUTPUT_DIR=/output/$dataset_id" \
        -e "TRUNC_TO_UINT16=$truncate_to_uint16" \
        -e "PREFLIGHT_ONLY=$preflight_only" \
        -e TMPDIR=/scratch \
        -v "$legacy_data_root:/legacy:ro" \
        -v "$migrated_data_root:/output" \
        -v "$migrated_data_root/_migration:/migration" \
        -v "$migration_scratch_root:/scratch" \
        ingest
}

run_all_datasets() {
    preflight_only=$1

    run_dataset lbda_v2 /ingest/manifests/legacy/lbda_v2.yml \
        0001-01-01T00:00:00Z '{"years": 1}' false "$preflight_only"
    run_dataset paleocar_v2 /ingest/manifests/legacy/paleocar_v2.yml \
        0001-01-01T00:00:00Z '{"years": 1}' false "$preflight_only"
    run_dataset paleocar_v3 /ingest/manifests/paleocar_v3.yml \
        0103-01-01T00:00:00Z '{"years": 1}' true "$preflight_only"
    run_dataset prism /ingest/manifests/legacy/prism.yml \
        1895-01-01T00:00:00Z '{"months": 1}' false "$preflight_only"
    run_dataset srtm /ingest/manifests/legacy/srtm.yml \
        2009-01-01T00:00:00Z '{"years": 1}' false "$preflight_only"
}

compose --profile ingest build ingest

echo "Validating all migration inputs"
run_all_datasets true

echo "All migration inputs passed preflight"
if [ "${MIGRATION_PREFLIGHT_ONLY:-false}" = true ]; then
    exit 0
fi

mkdir -p "$migrated_data_root/_migration"
mkdir -p "$migration_scratch_root"
migrated_data_root=$(CDPATH= cd -- "$migrated_data_root" && pwd)
migration_scratch_root=$(CDPATH= cd -- "$migration_scratch_root" && pwd)
cp "$repository_root/timeseries/ingest/legacy-metadata.yml" \
    "$migrated_data_root/_migration/metadata.yml"

run_all_datasets false

echo "Migration complete: $migrated_data_root"
echo "Generated metadata: $migrated_data_root/_migration/metadata.yml"
