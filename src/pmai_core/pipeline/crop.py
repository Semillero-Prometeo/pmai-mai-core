"""JPEG crop of a person bbox from a camera frame."""

from __future__ import annotations

import base64

import cv2
import numpy as np


def crop_jpeg_base64(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    quality: int = 85,
) -> str:
    """Clip ``bbox`` to ``frame``, encode as JPEG, and return raw base64."""
    xmin, ymin, xmax, ymax = (int(v) for v in bbox)
    height, width = frame.shape[:2]
    x1 = max(0, min(width, xmin))
    x2 = max(0, min(width, xmax))
    y1 = max(0, min(height, ymin))
    y2 = max(0, min(height, ymax))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("empty crop")
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encode failed")
    return base64.b64encode(buf).decode("ascii")
