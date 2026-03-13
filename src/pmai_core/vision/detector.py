"""YOLO object detector using the trained Ultralytics model (.pt)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog
from numpy.typing import NDArray
from ultralytics import YOLO

from pmai_core.domain.detection import BBox, Detection
from pmai_core.settings import VisionSettings

logger = structlog.get_logger(__name__)


class YOLODetector:
    """Wraps the trained Ultralytics YOLO model (.pt) for detection on CPU."""

    def __init__(self, settings: VisionSettings, class_names: list[str] | None = None) -> None:
        model_path = Path(settings.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        self._model = YOLO(str(model_path), task="detect")
        # For N100 (CPU only) we always run on CPU.
        self._model.to(settings.device)

        self._confidence_threshold = settings.confidence_threshold
        # Prefer class names from the model; fallback to user-supplied
        self._class_names = self._model.names or class_names or []
        logger.info(
            "yolo_detector_loaded",
            model=str(model_path),
            threshold=self._confidence_threshold,
            device=settings.device,
        )

    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Run inference on a BGR frame and return a list of ``Detection``."""
        results = self._model(frame, verbose=False)[0]
        boxes = results.boxes

        detections: list[Detection] = []
        for i in range(len(boxes)):
            cls_id = int(boxes[i].cls.item())
            conf = float(boxes[i].conf.item())
            if conf < self._confidence_threshold:
                continue

            xyxy = boxes[i].xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy.tolist()

            label = self._class_names[cls_id] if cls_id < len(self._class_names) else str(cls_id)

            detections.append(
                Detection(
                    bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
                    label=label,
                    class_id=cls_id,
                    confidence=conf,
                )
            )

        # YOLO ya aplica NMS interno, pero mantenemos la función por si se
        # quiere filtrar aún más.
        return self._nms(detections, iou_threshold=0.45)

    @staticmethod
    def _nms(detections: list[Detection], iou_threshold: float = 0.45) -> list[Detection]:
        """Simple greedy non-maximum suppression."""
        if not detections:
            return []
        detections.sort(key=lambda d: d.confidence, reverse=True)
        kept: list[Detection] = []
        for det in detections:
            if any(YOLODetector._iou(det, k) > iou_threshold for k in kept):
                continue
            kept.append(det)
        return kept

    @staticmethod
    def _iou(a: Detection, b: Detection) -> float:
        ax1, ay1, ax2, ay2 = a.bbox.to_xyxy()
        bx1, by1, bx2, by2 = b.bbox.to_xyxy()
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0
