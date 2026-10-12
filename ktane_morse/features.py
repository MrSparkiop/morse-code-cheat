"""Turning captured pixels into a brightness number for the Morse lamp.

The lit lamp is a saturated amber-yellow (about RGB 251,222,71) drawn with an
unlit shader, so it stays bright when the room lights go out; the unlit lamp
is a dull orange-brown (about 117,62,41) that follows room lighting. Scoring
pixels by "yellowness" = min(R, G) - B gives a big on/off contrast while
ignoring white highlights, grey metal and the red emergency-light tint.
"""

from __future__ import annotations

import numpy as np


def yellowness(rgb: np.ndarray) -> np.ndarray:
    """Per-pixel score min(R, G) - B clipped to 0..255, as int16. rgb is (..., 3) uint8."""
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    return np.clip(np.minimum(r, g) - b, 0, 255)


def lamp_level(rgb_roi: np.ndarray, top_fraction: float = 0.15, min_pixels: int = 4) -> float:
    """Mean yellowness of the most yellow pixels of a region around the lamp.

    Using only the top pixels keeps the signal strong when the region is drawn
    loosely around the lamp.
    """
    y = yellowness(rgb_roi).ravel()
    if y.size == 0:
        return 0.0
    k = min(y.size, max(min_pixels, int(y.size * top_fraction)))
    top = np.partition(y, y.size - k)[y.size - k:]
    return float(top.mean())
