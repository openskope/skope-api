from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import HTTPException

from app.core.tiles import stream_tile


async def test_stream_tile_closes_upstream_error_response(monkeypatch):
    response = Mock(status_code=500)
    response.aclose = AsyncMock()
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "upstream failure",
        request=httpx.Request("GET", "http://titiler/tile"),
        response=response,
    )
    client = Mock()
    client.build_request.return_value = httpx.Request("GET", "http://titiler/tile")
    client.send = AsyncMock(return_value=response)
    app_state = SimpleNamespace(client=client, data_reader=Mock())

    monkeypatch.setattr(
        "app.core.tiles.fetch_lookup_dict",
        AsyncMock(return_value={"ppt": {"0001": {"file": "data.tif", "bidx": 1}}}),
    )
    monkeypatch.setattr(
        "app.core.tiles.resolve_uri_single_band",
        lambda *_args: ("/data/data.tif", 1),
    )

    with pytest.raises(HTTPException) as exc_info:
        await stream_tile(
            app_state=app_state,
            dataset_id="dataset",
            variable_id="ppt",
            year="0001",
            z=0,
            x=0,
            y=0,
            colormap="viridis",
            rescale="0,100",
        )

    assert exc_info.value.status_code == 502
    response.aclose.assert_awaited_once()
