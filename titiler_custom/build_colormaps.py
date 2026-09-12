"""
Run once at Docker image build time to bake custom colormaps into rio-tiler's
cmap_data directory as .npy files. Each file is a (256, 4) uint8 array [R, G, B, A].
"""
import json
import sys
from pathlib import Path

import numpy as np
import rio_tiler


def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _stops_to_array(stops: list, name: str = "") -> np.ndarray:
    if len(stops) < 2:
        raise ValueError(f"Colormap {name!r} must have at least 2 color stops, got {len(stops)}.")
    n = len(stops)
    positions = [i / (n - 1) for i in range(n)]
    result = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256):
        t = i / 255
        lo, hi = 0, n - 1
        for j in range(n - 1):
            if positions[j] <= t <= positions[j + 1]:
                lo, hi = j, j + 1
                break
        span = positions[hi] - positions[lo]
        a = 0.0 if span == 0 else (t - positions[lo]) / span
        lr, lg, lb = _hex_to_rgb(stops[lo])
        hr, hg, hb = _hex_to_rgb(stops[hi])
        result[i] = [
            round(lr + a * (hr - lr)),
            round(lg + a * (hg - lg)),
            round(lb + a * (hb - lb)),
            255,
        ]
    return result


cmap_dir = Path(rio_tiler.__file__).parent / "cmap_data"
source = Path(__file__).parent / "custom.json"

with open(source) as f:
    colormaps = json.load(f)

for name, stops in colormaps.items():
    dest = cmap_dir / f"{name}.npy"
    np.save(str(dest), _stops_to_array(stops, name))
sys.exit(0)
