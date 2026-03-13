"""Single-camera object tracker using a simple IoU-based approach.

This implementation provides a lightweight tracker suitable for CPU-only
hardware.  It assigns local track IDs within a single camera stream by
matching detections across consecutive frames using IoU overlap.  For a
production system with more compute budget, consider integrating ByteTrack
or BoT-SORT via ``ultralytics``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pmai_core.domain.detection import Detection
from pmai_core.domain.tracked_object import TrackedObject


@dataclass
class _Track:
    track_id: int
    detection: Detection
    last_seen: float = field(default_factory=time.time)
    missed_frames: int = 0


class ObjectTracker:
    """IoU-based single-camera tracker.

    Parameters
    ----------
    iou_threshold:
        Minimum IoU to consider a detection as matching an existing track.
    max_missed:
        Number of consecutive frames a track can go unmatched before removal.
    """

    def __init__(
        self,
        camera_id: str,
        iou_threshold: float = 0.3,
        max_missed: int = 30,
    ) -> None:
        self._camera_id = camera_id
        self._iou_threshold = iou_threshold
        self._max_missed = max_missed
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        """Match ``detections`` to existing tracks and return updated objects."""
        if not self._tracks:
            return self._init_tracks(detections)

        cost = self._build_iou_matrix(detections)
        matched_det: set[int] = set()
        matched_trk: set[int] = set()

        # Greedy matching by highest IoU
        if cost.size > 0:
            flat_order = np.argsort(-cost, axis=None)
            for flat_idx in flat_order:
                ti = int(flat_idx // cost.shape[1])
                di = int(flat_idx % cost.shape[1])
                if ti in matched_trk or di in matched_det:
                    continue
                if cost[ti, di] < self._iou_threshold:
                    break
                self._tracks[ti].detection = detections[di]
                self._tracks[ti].last_seen = time.time()
                self._tracks[ti].missed_frames = 0
                matched_trk.add(ti)
                matched_det.add(di)

        # Increment missed count for unmatched tracks
        for i, trk in enumerate(self._tracks):
            if i not in matched_trk:
                trk.missed_frames += 1

        # Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_det:
                self._tracks.append(_Track(track_id=self._next_id, detection=det))
                self._next_id += 1

        # Remove stale tracks
        self._tracks = [t for t in self._tracks if t.missed_frames <= self._max_missed]

        return self._to_tracked_objects()

    def _init_tracks(self, detections: list[Detection]) -> list[TrackedObject]:
        for det in detections:
            self._tracks.append(_Track(track_id=self._next_id, detection=det))
            self._next_id += 1
        return self._to_tracked_objects()

    def _build_iou_matrix(self, detections: list[Detection]) -> NDArray[np.float32]:
        n_tracks = len(self._tracks)
        n_dets = len(detections)
        if n_tracks == 0 or n_dets == 0:
            return np.empty((0, 0), dtype=np.float32)

        matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
        for ti, trk in enumerate(self._tracks):
            for di, det in enumerate(detections):
                matrix[ti, di] = self._iou(trk.detection, det)
        return matrix

    @staticmethod
    def _iou(a: Detection, b: Detection) -> float:
        ax1, ay1, ax2, ay2 = a.bbox.to_xyxy()
        bx1, by1, bx2, by2 = b.bbox.to_xyxy()
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _to_tracked_objects(self) -> list[TrackedObject]:
        objects: list[TrackedObject] = []
        for trk in self._tracks:
            if trk.missed_frames > 0:
                continue
            objects.append(
                TrackedObject(
                    id=trk.track_id,
                    global_id="",
                    label=trk.detection.label,
                    confidence=trk.detection.confidence,
                    bbox=trk.detection.bbox,
                    camera_id=self._camera_id,
                    last_seen=trk.last_seen,
                )
            )
        return objects
