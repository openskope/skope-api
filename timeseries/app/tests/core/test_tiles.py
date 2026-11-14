from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import HTTPException

from app.core.tiles import stream_tile


async def test_stream_tile_requests_a_one_band_view_of_the_cog():
    response = Mock(status_code=200, headers={"Content-Type": "image/png"})
    response.raise_for_status = Mock()
    client = Mock()
    client.build_request.return_value = httpx.Request("GET", "http://titiler/tile")
    client.send = AsyncMock(return_value=response)
    app_state = SimpleNamespace(client=client)

    await stream_tile(
        app_state=app_state,
        cog_path="/releases/r/cogs/ppt/ds--0001--0100.tif",
        band=37,
        z=8,
        x=48,
        y=99,
        colormap="viridis",
        rescale="0,100",
    )

    params = client.build_request.call_args.kwargs["params"]
    assert params["url"] == "vrt:///releases/r/cogs/ppt/ds--0001--0100.tif?bands=37"
    assert params["bidx"] == 1


async def test_stream_tile_closes_upstream_error_response():
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
    app_state = SimpleNamespace(client=client)

    with pytest.raises(HTTPException) as exc_info:
        await stream_tile(
            app_state=app_state,
            cog_path="/releases/r/cogs/ppt/ds--0001--0001.tif",
            band=1,
            z=0,
            x=0,
            y=0,
            colormap="viridis",
            rescale="0,100",
        )

    assert exc_info.value.status_code == 502
    response.aclose.assert_awaited_once()
