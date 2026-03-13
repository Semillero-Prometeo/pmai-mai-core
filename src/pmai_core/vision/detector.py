"""YOLO object detector using ONNX Runtime for CPU-optimised inference."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import structlog
from numpy.typing import NDArray

from pmai_core.domain.detection import BBox, Detection
from pmai_core.settings import VisionSettings

logger = structlog.get_logger(__name__)

# YOLO input size (square)
_INPUT_SIZE = 640


class YOLODetector:
    """Wraps a YOLO ONNX model for object detection on CPU.

    The model is expected to be exported from Ultralytics with the standard
    YOLO output format: ``(1, num_detections, 4+num_classes)`` or the
    transposed variant ``(1, 4+num_classes, num_detections)``.
    """

    def __init__(self, settings: VisionSettings, class_names: list[str] | None = None) -> None:
        model_path = Path(settings.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO ONNX model not found: {model_path}")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 2

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._confidence_threshold = settings.confidence_threshold
        self._class_names = class_names or []
        logger.info(
            "yolo_detector_loaded",
            model=str(model_path),
            threshold=self._confidence_threshold,
        )

    def detect(self, frame: NDArray[np.uint8]) -> list[Detection]:
        """Run inference on a BGR frame and return a list of ``Detection``."""
        original_h, original_w = frame.shape[:2]
        blob = self._preprocess(frame)
        outputs = self._session.run(None, {self._input_name: blob})
        raw = outputs[0]
        return self._postprocess(raw, original_w, original_h)

    def _preprocess(self, frame: NDArray[np.uint8]) -> NDArray[np.float32]:
        img = cv2.resize(frame, (_INPUT_SIZE, _INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def _postprocess(
        self,
        raw: NDArray[np.float32],
        orig_w: int,
        orig_h: int,
    ) -> list[Detection]:
        # Handle shape (1, 4+nc, N) by transposing to (1, N, 4+nc)
        if raw.ndim == 3 and raw.shape[1] < raw.shape[2]:
            raw = np.transpose(raw, (0, 2, 1))

        predictions = raw[0]  # shape: (N, 4+nc)
        detections: list[Detection] = []

        scale_x = orig_w / _INPUT_SIZE
        scale_y = orig_h / _INPUT_SIZE

        for pred in predictions:
            cx, cy, w, h = pred[:4]
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            if confidence < self._confidence_threshold:
                continue

            xmin = int((cx - w / 2) * scale_x)
            ymin = int((cy - h / 2) * scale_y)
            xmax = int((cx + w / 2) * scale_x)
            ymax = int((cy + h / 2) * scale_y)

            xmin = max(0, min(xmin, orig_w - 1))
            ymin = max(0, min(ymin, orig_h - 1))
            xmax = max(0, min(xmax, orig_w - 1))
            ymax = max(0, min(ymax, orig_h - 1))

            label = (
                self._class_names[class_id]
                if class_id < len(self._class_names)
                else str(class_id)
            )

            detections.append(
                Detection(
                    bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
                    label=label,
                    class_id=class_id,
                    confidence=confidence,
                )
            )

        detections = self._nms(detections, iou_threshold=0.45)
        return detections

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
