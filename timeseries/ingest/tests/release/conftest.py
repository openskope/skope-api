"""Fixtures for release tests; helpers live in release_fixtures.py."""

import pytest

from release_fixtures import curated_doc, manifest_doc, write_dataset, write_source, years


@pytest.fixture
def temporal_dataset(tmp_path):
    """A 5-year, 2-variable temporal dataset with chunk_size 2 (chunks of 2, 2, 1)."""

    def make(dataset_id="synth", variables=("alpha", "beta"), *, keys=None, curated=None, manifest=None, source_kwargs=None):
        keys = keys or years(100, 5)
        sources = {}
        for i, vid in enumerate(variables):
            path = tmp_path / "sources" / f"{vid}.tif"
            path.parent.mkdir(exist_ok=True)
            write_source(path, keys, seed=i, **(source_kwargs or {}).get(vid, {}))
            sources[vid] = str(path)
        cur = curated_doc(dataset_id, list(variables), origin=keys[0], end=keys[-1], **(curated or {}))
        man = manifest_doc(dataset_id, sources, **(manifest or {}))
        return write_dataset(tmp_path, dataset_id, cur, man)

    return make


@pytest.fixture
def static_dataset(tmp_path):
    def make(dataset_id="elev", variable="elevation"):
        path = tmp_path / "sources" / f"{variable}.tif"
        path.parent.mkdir(exist_ok=True)
        write_source(path, ["band1"])
        cur = curated_doc(dataset_id, [variable], profile="StaticRasterDataset")
        man = manifest_doc(dataset_id, {variable: str(path)}, static=True)
        return write_dataset(tmp_path, dataset_id, cur, man)

    return make
