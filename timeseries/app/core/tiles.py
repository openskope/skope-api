import logging
import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def stream_tile(
    app_state,
    cog_path: str,
    band: int,
    z: int,
    x: int,
    y: int,
    colormap: str,
    rescale: str | None,
) -> StreamingResponse:
    """
    Requests one band of one COG from the internal tile server and streams the
    image bytes back to the client. The caller has resolved the timestep.
    """
    tile_provider_url = (
        f"{settings.tile_server_url}/cog/tiles/WebMercatorQuad/{z}/{x}/{y}"
    )

    # A one-band view of the COG. The tile server reprojects through a warped
    # dataset, and GDAL's warper processes every band of that dataset, so
    # opening the file itself would decode every timestep in the chunk.
    params = {
        "url": f"vrt://{cog_path}?bands={band}",
        "bidx": 1,
        "colormap_name": colormap,
    }
    if rescale is not None:
        params["rescale"] = rescale

    try:
        request = app_state.client.build_request(
            "GET", tile_provider_url, params=params
        )
        response = await app_state.client.send(request, stream=True)
        response.raise_for_status()

        async def iter_tile_bytes():
            try:
                async for chunk in response.aiter_bytes():
                    yield chunk
            finally:
                await response.aclose()

        return StreamingResponse(
            iter_tile_bytes(),
            media_type=response.headers.get("Content-Type", "image/png"),
            status_code=response.status_code,
        )

    except httpx.HTTPStatusError as exc:
        await exc.response.aclose()
        logger.error("Tile server returned status %d", exc.response.status_code)
        raise HTTPException(
            status_code=502, detail="Upstream tile server error."
        ) from exc
    except httpx.RequestError as exc:
        logger.error("Failed to connect to tile server: %s", exc)
        raise HTTPException(
            status_code=502, detail="Tile server is unreachable."
        ) from exc
