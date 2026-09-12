import json
import os
import pytest

from app.store.index_loaders import (
    _get_cached_lookup,
    _set_cached_lookup,
    fetch_lookup_dict,
)


@pytest.fixture(autouse=True)
def isolate_cache_dir(tmp_path, monkeypatch):
    cache_dir = str(tmp_path / "cache")
    os.makedirs(cache_dir, exist_ok=True)
    monkeypatch.setattr("app.store.index_loaders._CACHE_DIR", cache_dir)
    return cache_dir


# ---------------------------------------------------------------------------
# fetch_lookup_dict — cache hit


async def test_cache_hit_returns_cached_data(minimal_lookup_data, mock_data_reader):
    _set_cached_lookup("cached-ds", minimal_lookup_data)
    result = await fetch_lookup_dict("cached-ds", "s3://bucket", mock_data_reader)
    assert result == minimal_lookup_data
    mock_data_reader.read_json.assert_not_called()


# ---------------------------------------------------------------------------
# fetch_lookup_dict — cache miss


async def test_cache_miss_fetches_from_origin(minimal_lookup_data, mock_data_reader):
    result = await fetch_lookup_dict("fresh-ds", "s3://bucket", mock_data_reader)
    assert result == minimal_lookup_data
    mock_data_reader.read_json.assert_called_once_with(
        "s3://bucket/fresh-ds/lookup.json"
    )


async def test_caches_after_miss(minimal_lookup_data, mock_data_reader):
    await fetch_lookup_dict("once-ds", "s3://bucket", mock_data_reader)
    # Second call should hit cache
    await fetch_lookup_dict("once-ds", "s3://bucket", mock_data_reader)
    assert mock_data_reader.read_json.call_count == 1


# ---------------------------------------------------------------------------
# fetch_lookup_dict — schema validation


async def test_invalid_schema_missing_file_key_raises(mock_data_reader):
    mock_data_reader.read_json.return_value = {"ppt": {"0100": {"bidx": 1}}}
    with pytest.raises(ValueError):
        await fetch_lookup_dict("bad-ds", "s3://bucket", mock_data_reader)


async def test_invalid_schema_missing_bidx_key_raises(mock_data_reader):
    mock_data_reader.read_json.return_value = {"ppt": {"0100": {"file": "x.tif"}}}
    with pytest.raises(ValueError):
        await fetch_lookup_dict("bad-ds2", "s3://bucket", mock_data_reader)


async def test_empty_data_raises(mock_data_reader):
    mock_data_reader.read_json.return_value = {}
    with pytest.raises(ValueError):
        await fetch_lookup_dict("empty-ds", "s3://bucket", mock_data_reader)


# ---------------------------------------------------------------------------
# fetch_lookup_dict — time key validation


async def test_invalid_time_key_format_raises(mock_data_reader):
    mock_data_reader.read_json.return_value = {
        "ppt": {"not-a-date": {"file": "x.tif", "bidx": 1}}
    }
    with pytest.raises(ValueError, match="time format"):
        await fetch_lookup_dict("fmt-ds", "s3://bucket", mock_data_reader)


async def test_out_of_order_keys_raises(mock_data_reader):
    # Python dicts preserve insertion order — 0103 before 0101 is out of order
    mock_data_reader.read_json.return_value = {
        "ppt": {
            "0103": {"file": "a.tif", "bidx": 1},
            "0101": {"file": "b.tif", "bidx": 1},
        }
    }
    with pytest.raises(ValueError, match="not sorted"):
        await fetch_lookup_dict("order-ds", "s3://bucket", mock_data_reader)


async def test_fetch_error_raises(mock_data_reader):
    mock_data_reader.read_json.side_effect = IOError("network down")
    with pytest.raises(ValueError, match="Failed to retrieve"):
        await fetch_lookup_dict("err-ds", "s3://bucket", mock_data_reader)


# ---------------------------------------------------------------------------
# _get_cached_lookup


def test_get_cached_lookup_returns_none_for_missing():
    result = _get_cached_lookup("never-stored")
    assert result is None


def test_get_cached_lookup_returns_none_for_corrupted_json(tmp_path, monkeypatch):
    cache_dir = str(tmp_path / "corrupt_cache")
    os.makedirs(cache_dir, exist_ok=True)
    monkeypatch.setattr("app.store.index_loaders._CACHE_DIR", cache_dir)
    bad_file = os.path.join(cache_dir, "corrupt-ds_lookup.json")
    with open(bad_file, "w") as f:
        f.write("{not valid json")
    result = _get_cached_lookup("corrupt-ds")
    assert result is None


# ---------------------------------------------------------------------------
# _set_cached_lookup


def test_set_cached_lookup_writes_atomically(
    minimal_lookup_data, tmp_path, monkeypatch
):
    cache_dir = str(tmp_path / "write_cache")
    os.makedirs(cache_dir, exist_ok=True)
    monkeypatch.setattr("app.store.index_loaders._CACHE_DIR", cache_dir)
    _set_cached_lookup("write-ds", minimal_lookup_data)
    expected_file = os.path.join(cache_dir, "write-ds_lookup.json")
    assert os.path.exists(expected_file)
    assert not os.path.exists(expected_file + ".tmp")
    with open(expected_file) as f:
        assert json.load(f) == minimal_lookup_data
