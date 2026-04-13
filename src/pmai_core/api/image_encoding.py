"""Shared JPEG encoding for annotated frames (camera + vision APIs)."""

from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


def resize_to_max_width(frame: NDArray[np.uint8], max_width: int) -> NDArray[np.uint8]:
    """Downscale so width <= max_width; returns a copy if already smaller."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_w = max_width
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def frame_to_jpeg_base64(
    frame: NDArray[np.uint8],
    *,
    max_width: int | None = None,
    jpeg_quality: int = 85,
) -> str:
    """Encode BGR uint8 frame as base64 JPEG; optional resize by max width."""
    to_encode: Any = frame
    if max_width is not None and max_width > 0:
        to_encode = resize_to_max_width(frame, max_width)
    _, buf = cv2.imencode(
        ".jpg",
        to_encode,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    return base64.b64encode(buf.tobytes()).decode("ascii")
