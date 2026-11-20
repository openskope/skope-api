import pytest

from app.core.zonal_stats import calculate_safe_chunk_size, generate_band_chunks
from app.exceptions import SelectedAreaPolygonIsTooLarge

# ---------------------------------------------------------------------------
# calculate_safe_chunk_size


def test_chunk_size_normal():
    result = calculate_safe_chunk_size(width=100, height=50, max_cells=500_000)
    assert result == 100  # 500_000 // 5_000


def test_chunk_size_zero_width_raises():
    with pytest.raises(ValueError, match="0-pixel"):
        calculate_safe_chunk_size(width=0, height=50, max_cells=500_000)


def test_chunk_size_zero_height_raises():
    with pytest.raises(ValueError, match="0-pixel"):
        calculate_safe_chunk_size(width=100, height=0, max_cells=500_000)


def test_chunk_size_area_exceeds_max_cells_raises():
    # 1000*1000 = 1_000_000 > 500_000 → n_bands_per_chunk = 0
    with pytest.raises(SelectedAreaPolygonIsTooLarge):
        calculate_safe_chunk_size(width=1000, height=1000, max_cells=500_000)


def test_chunk_size_exact_fit():
    # 100*100 = 10_000, max_cells = 10_000 → result = 1
    result = calculate_safe_chunk_size(width=100, height=100, max_cells=10_000)
    assert result == 1


# ---------------------------------------------------------------------------
# generate_band_chunks


def test_generate_band_chunks_exact_division():
    chunks = list(generate_band_chunks([1, 2, 3, 4, 5, 6], chunk_size=3))
    assert chunks == [[1, 2, 3], [4, 5, 6]]


def test_generate_band_chunks_partial_last_chunk():
    chunks = list(generate_band_chunks([1, 2, 3, 4, 5], chunk_size=3))
    assert chunks == [[1, 2, 3], [4, 5]]


def test_generate_band_chunks_single_element():
    chunks = list(generate_band_chunks([7], chunk_size=3))
    assert chunks == [[7]]


def test_generate_band_chunks_empty():
    chunks = list(generate_band_chunks([], chunk_size=3))
    assert chunks == []


def test_generate_band_chunks_chunk_size_larger_than_list():
    chunks = list(generate_band_chunks([1, 2], chunk_size=10))
    assert chunks == [[1, 2]]
