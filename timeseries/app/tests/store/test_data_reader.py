import json
import pytest

from app.store.data_reader import LocalDataReader, S3DataReader, get_data_reader

# ---------------------------------------------------------------------------
# get_data_reader factory


def test_get_data_reader_returns_s3_for_s3_url():
    reader = get_data_reader("s3://my-bucket/data")
    assert isinstance(reader, S3DataReader)


def test_get_data_reader_returns_local_for_file_path():
    reader = get_data_reader("/local/path")
    assert isinstance(reader, LocalDataReader)


def test_get_data_reader_returns_local_for_http_url():
    # Only s3:// triggers S3; anything else falls back to local
    reader = get_data_reader("http://example.com/data")
    assert isinstance(reader, LocalDataReader)


# ---------------------------------------------------------------------------
# LocalDataReader.read_json


async def test_local_data_reader_reads_valid_json(tmp_path):
    data = {"key": "value", "number": 42}
    json_file = tmp_path / "test.json"
    json_file.write_text(json.dumps(data))
    reader = LocalDataReader()
    result = await reader.read_json(str(json_file))
    assert result == data


async def test_local_data_reader_missing_file_raises(tmp_path):
    reader = LocalDataReader()
    with pytest.raises(FileNotFoundError):
        await reader.read_json(str(tmp_path / "nonexistent.json"))
