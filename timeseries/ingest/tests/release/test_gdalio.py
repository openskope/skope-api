"""Bounded read windows (`gdalio.read_windows`)."""

import pytest

from skope_release import gdalio

MB = 1024 * 1024


def covered(windows):
    cells = [(x + i, y + j) for x, y, w, h in windows for i in range(w) for j in range(h)]
    return len(cells), set(cells)


def test_a_wide_many_band_raster_reads_a_few_blocks_at_a_time():
    """PaleoCAR: 1560 x 1440, 1,898 UInt32 bands in 128-pixel tiles, about 119 MB per block."""
    windows = gdalio.read_windows(1560, 1440, 1898, 4, 128, 128)
    assert (windows.window_width, windows.window_height) == (256, 128)
    assert not windows.full_width
    assert windows.window_width * windows.window_height * 1898 * 4 <= gdalio.READ_BUDGET_BYTES


def test_a_small_raster_is_read_whole():
    windows = gdalio.read_windows(115, 48, 2017, 4, 115, 1)
    assert list(windows) == [(0, 0, 115, 48)]


def test_a_stripped_source_is_read_in_whole_rows_within_the_budget():
    windows = gdalio.read_windows(5000, 4000, 100, 2, 5000, 1)
    assert windows.full_width
    assert windows.window_height * 5000 * 100 * 2 <= gdalio.READ_BUDGET_BYTES


def test_one_block_is_the_minimum(monkeypatch):
    monkeypatch.setattr(gdalio, "READ_BUDGET_BYTES", 1)
    windows = gdalio.read_windows(40, 30, 5, 4, 16, 16)
    assert (windows.window_width, windows.window_height) == (16, 16)


def test_tiles_off_the_16_pixel_grid_fall_back_to_whole_rows(monkeypatch):
    monkeypatch.setattr(gdalio, "READ_BUDGET_BYTES", 1)
    windows = gdalio.read_windows(100, 30, 5, 4, 20, 20)
    assert windows.full_width and windows.window_height == 20


@pytest.mark.parametrize("size", [(40, 30, 16, 16), (1560, 1440, 128, 128), (100, 30, 20, 20)])
def test_windows_cover_every_cell_once(monkeypatch, size):
    width, height, block_width, block_height = size
    monkeypatch.setattr(gdalio, "READ_BUDGET_BYTES", 1)
    count, cells = covered(gdalio.read_windows(width, height, 1, 1, block_width, block_height))
    assert count == len(cells) == width * height
