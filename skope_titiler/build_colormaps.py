"""
Run once at Docker image build time to register every palette in
deploy/display/palettes.yml as a named rio-tiler colormap (release consumption
spec DISP-011). Each becomes a (256, 4) uint8 .npy file [R, G, B, A] in
rio-tiler's cmap_data directory, replacing any built-in palette of the same
name, so tiles render with exactly the colours the API serves for legends.

- ramp: the colours are spread evenly over 0-255 and interpolated linearly.
- set: entry i is colour i, for categorical values 0..n-1; the rest are
  transparent.
"""
import sys
from pathlib import Path

import numpy as np
import rio_tiler
import yaml


def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _ramp(stops: list, name: str = "") -> np.ndarray:
    if len(stops) < 2:
        raise ValueError(f"Ramp {name!r} must have at least 2 colours, got {len(stops)}.")
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


def _set(colors: list, name: str = "") -> np.ndarray:
    if not 1 <= len(colors) <= 256:
        raise ValueError(f"Set {name!r} must have 1 to 256 colours, got {len(colors)}.")
    result = np.zeros((256, 4), dtype=np.uint8)
    for i, color in enumerate(colors):
        result[i] = [*_hex_to_rgb(color), 255]
    return result


cmap_dir = Path(rio_tiler.__file__).parent / "cmap_data"
source = Path(__file__).parent / "palettes.yml"

with open(source) as f:
    palettes = yaml.safe_load(f)

for name, palette in palettes.items():
    build = {"ramp": _ramp, "set": _set}[palette["kind"]]
    np.save(str(cmap_dir / f"{name}.npy"), build(palette["colors"], name))
sys.exit(0)
