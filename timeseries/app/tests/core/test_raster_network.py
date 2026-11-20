"""Exercise actual GDAL/libcurl timeouts against a stalled local HTTP server."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import time

import numpy as np
import pytest
from rasterio.errors import RasterioIOError
from rasterio.windows import Window

from app.core.timeseries_processing import (
    RASTERIO_ENV_KWARGS,
    extract_summarystat_timeseries,
)


@pytest.mark.parametrize("stall_body", [False, True])
def test_raster_http_transfer_timeout_includes_body(monkeypatch, stall_body):
    release = threading.Event()
    requested = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.send_response(200)
            self.send_header("Content-Length", "1048576")
            self.end_headers()

        def do_GET(self):
            requested.set()
            if stall_body:
                start, end = self.headers["Range"].removeprefix("bytes=").split("-")
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/1048576")
                self.send_header("Content-Length", str(int(end) - int(start) + 1))
                self.end_headers()
                self.wfile.flush()
            release.wait(10)

        def log_message(self, *args):
            pass

    monkeypatch.setitem(RASTERIO_ENV_KWARGS, "GDAL_HTTP_TIMEOUT", "1")
    monkeypatch.setitem(RASTERIO_ENV_KWARGS, "GDAL_HTTP_CONNECTTIMEOUT", "1")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        started = time.monotonic()
        with pytest.raises(RasterioIOError):
            extract_summarystat_timeseries(
                f"http://127.0.0.1:{server.server_port}/stalled.tif",
                [1],
                Window(0, 0, 1, 1),
                np.ones((1, 1), dtype=bool),
                1,
            )
        elapsed = time.monotonic() - started
        assert requested.is_set()
        assert 0.5 <= elapsed < 6
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
