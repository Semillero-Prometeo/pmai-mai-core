"""Tests for YOLO detector postprocessing logic."""

from __future__ import annotations

from pmai_core.domain.detection import BBox, Detection
from pmai_core.vision.detector import YOLODetector


def _det(
    xmin: int, ymin: int, xmax: int, ymax: int,
    label: str, cid: int, conf: float,
) -> Detection:
    return Detection(
        bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
        label=label,
        class_id=cid,
        confidence=conf,
    )


class TestNMS:
    def test_keeps_non_overlapping(self) -> None:
        dets = [
            _det(0, 0, 50, 50, "a", 0, 0.9),
            _det(200, 200, 250, 250, "b", 1, 0.8),
        ]
        result = YOLODetector._nms(dets, iou_threshold=0.45)
        assert len(result) == 2

    def test_suppresses_overlapping(self) -> None:
        dets = [
            _det(0, 0, 100, 100, "a", 0, 0.9),
            _det(5, 5, 105, 105, "a", 0, 0.7),
        ]
        result = YOLODetector._nms(dets, iou_threshold=0.45)
        assert len(result) == 1
        assert result[0].confidence == 0.9

    def test_empty_input(self) -> None:
        assert YOLODetector._nms([]) == []


class TestIoU:
    def test_identical_boxes(self) -> None:
        a = _det(0, 0, 100, 100, "a", 0, 1.0)
        assert abs(YOLODetector._iou(a, a) - 1.0) < 1e-6

    def test_no_overlap(self) -> None:
        a = _det(0, 0, 50, 50, "a", 0, 1.0)
        b = _det(100, 100, 150, 150, "b", 0, 1.0)
        assert YOLODetector._iou(a, b) == 0.0
