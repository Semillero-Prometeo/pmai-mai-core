"""Draw YOLO detections and ReID labels on frames for visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pmai_core.domain.tracked_object import TrackedObject


def draw_detections(
    frame: NDArray[np.uint8],
    tracked_objects: list[TrackedObject],
) -> NDArray[np.uint8]:
    """Draw bounding boxes and labels (YOLO label + ReID global_id) on a copy of the frame."""
    out = frame.copy()
    for obj in tracked_objects:
        bbox = obj.bbox
        x1, y1 = bbox.xmin, bbox.ymin
        x2, y2 = bbox.xmax, bbox.ymax

        color = (0, 255, 0) if obj.global_id else (0, 165, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label_parts = [obj.label]
        if obj.global_id:
            label_parts.append(f"#{obj.global_id}")
        label_text = " ".join(label_parts)

        (tw, th), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1,
        )
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out,
            label_text,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
    return out
