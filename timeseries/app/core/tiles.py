import logging
import httpx
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.core.slice_resolver import resolve_uri_single_band
from app.store.index_loaders import fetch_lookup_dict
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def stream_tile(
    app_state,
    dataset_id: str,
    variable_id: str,
    year: str,
    z: int,
    x: int,
    y: int,
    colormap: str,
    rescale: str,
) -> StreamingResponse:
    """
    Resolves the exact storage URI for the requested year, constructs the TiTiler URL,
    and streams the image bytes securely back to the client.
    """

    lookup_data = await fetch_lookup_dict(
        dataset_id=dataset_id,
        storage_base_url=settings.storage_base_url,
        data_reader=app_state.data_reader,
    )

    target_file, target_band = resolve_uri_single_band(
        lookup_data, variable_id, year, settings.storage_base_url
    )

    tile_provider_url = (
        f"{settings.tile_server_url}/cog/tiles/WebMercatorQuad/{z}/{x}/{y}"
    )

    params = {
        "url": target_file,
        "bidx": target_band,
        "colormap_name": colormap,
        "rescale": rescale,
    }

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
