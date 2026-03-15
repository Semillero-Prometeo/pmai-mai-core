"""Draw YOLO detections and ReID labels on frames for visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pmai_core.domain.tracked_object import TrackedObject

_COLORS: list[tuple[int, int, int]] = [
    (0, 255, 0),
    (255, 127, 0),
    (0, 127, 255),
    (255, 0, 127),
    (127, 255, 0),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
]


def _gid_color(global_id: str) -> tuple[int, int, int]:
    """Deterministic color per global_id so the same identity always has the same color."""
    if not global_id:
        return (128, 128, 128)
    return _COLORS[hash(global_id) % len(_COLORS)]


def draw_detections(
    frame: NDArray[np.uint8],
    tracked_objects: list[TrackedObject],
    *,
    cameras_seen_map: dict[str, list[str]] | None = None,
) -> NDArray[np.uint8]:
    """Draw bounding boxes, labels, global IDs, and cameras_seen on a frame copy.

    Parameters
    ----------
    cameras_seen_map:
        Optional mapping ``{global_id: [camera_ids...]}`` from the registry.
        When provided, a second line shows which cameras share this identity.
    """
    out = frame.copy()
    for obj in tracked_objects:
        bbox = obj.bbox
        x1, y1 = bbox.xmin, bbox.ymin
        x2, y2 = bbox.xmax, bbox.ymax

        color = _gid_color(obj.global_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        line1 = f"{obj.label} {obj.confidence:.0%}"
        if obj.global_id:
            short_gid = obj.global_id.replace("gid_", "")
            line1 += f"  ID:{short_gid}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1

        (tw1, th1), _ = cv2.getTextSize(line1, font, scale, thickness)
        label_y = max(y1, th1 + 10)

        cv2.rectangle(
            out,
            (x1, label_y - th1 - 6),
            (x1 + tw1 + 6, label_y + 4),
            color,
            -1,
        )
        cv2.putText(out, line1, (x1 + 3, label_y - 2), font, scale, (255, 255, 255), thickness)

        if obj.global_id and cameras_seen_map:
            cams = cameras_seen_map.get(obj.global_id, [])
            if len(cams) > 1:
                line2 = f"SEEN: {', '.join(sorted(cams))}"
                (tw2, th2), _ = cv2.getTextSize(line2, font, scale, thickness)
                row2_y = label_y + th2 + 8
                cv2.rectangle(
                    out,
                    (x1, row2_y - th2 - 4),
                    (x1 + tw2 + 6, row2_y + 4),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    out, line2, (x1 + 3, row2_y), font, scale, (0, 255, 255), thickness,
                )

    return out
