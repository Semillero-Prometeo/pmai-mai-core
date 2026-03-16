"""Identification resource: detection, tracking, ReID, and event publishing."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np
import structlog
from numpy.typing import NDArray

from pmai_core.domain.context_object import GlobalObjectForContext
from pmai_core.domain.events import ObjectDetectedEvent, ObjectReIdentifiedEvent
from pmai_core.domain.tracked_object import TrackedObject
from pmai_core.pipeline.global_objects import compute_global_objects_for_context
from pmai_core.resources.reid.extractor import EmbeddingExtractor
from pmai_core.resources.reid.matcher import CosineMatcher
from pmai_core.resources.reid.registry import GlobalRegistry
from pmai_core.resources.vision.detector import YOLODetector
from pmai_core.resources.vision.tracker import ObjectTracker
from pmai_core.settings import Settings

if TYPE_CHECKING:
    from pmai_core.messaging.client import NATSClient
    from pmai_core.resources.camera.capture import CameraCapture

logger = structlog.get_logger(__name__)


class IdentificationService:
    """Detection, tracking, ReID and NATS publishing for the observation phase."""

    def __init__(
        self,
        settings: Settings,
        nats_client: NATSClient | None = None,
    ) -> None:
        self._settings = settings
        self._nats = nats_client

        self._trackers: dict[str, ObjectTracker] = {}

        # ONNX Runtime (ReID) before PyTorch (YOLO) to avoid OpenMP deadlocks
        self._registry = GlobalRegistry(max_size=settings.reid.gallery_max_size)
        self._extractor = EmbeddingExtractor(settings.reid)
        self._matcher = CosineMatcher(
            registry=self._registry,
            similarity_threshold=settings.reid.similarity_threshold,
        )
        self._detector = YOLODetector(settings.vision)

        self._frame_counter: dict[str, int] = {}
        self._last_emit_time: dict[str, float] = {}
        self._last_annotated: dict[str, tuple[NDArray[np.uint8], list[TrackedObject]]] = {}

    @property
    def registry(self) -> GlobalRegistry:
        return self._registry

    @property
    def trackers(self) -> dict[str, ObjectTracker]:
        return dict(self._trackers)

    @property
    def all_last_annotated(
        self,
    ) -> dict[str, tuple[NDArray[np.uint8], list[TrackedObject]]]:
        return dict(self._last_annotated)

    def get_last_annotated(
        self, camera_id: str
    ) -> tuple[NDArray[np.uint8], list[TrackedObject]] | None:
        return self._last_annotated.get(camera_id)

    def get_global_objects_for_context(self) -> list[GlobalObjectForContext]:
        return compute_global_objects_for_context(
            self._last_annotated,
            self._registry,
        )

    async def run_phase(
        self,
        captures: dict[str, CameraCapture],
    ) -> bool:
        """Run observation + ReID phase: consume frames, detect, track, ReID, publish.

        Returns True if at least one frame was processed.
        """
        processed_any = False
        for cam_id, capture in captures.items():
            result = capture.get_frame(timeout=0.05)
            if result is None:
                continue

            frame, _timestamp = result
            processed_any = True

            if cam_id not in self._trackers:
                self._trackers[cam_id] = ObjectTracker(camera_id=cam_id)
            self._frame_counter.setdefault(cam_id, 0)
            self._frame_counter[cam_id] += 1
            frame_idx = self._frame_counter[cam_id]

            reid_interval = self._settings.reid.embedding_update_interval
            do_reid = self._extractor.is_available and (
                frame_idx == 1 or frame_idx % reid_interval == 0
            )

            tracked = await asyncio.to_thread(
                self._process_frame_sync,
                cam_id,
                frame,
                frame_idx,
                do_reid,
            )
            self._last_annotated[cam_id] = (frame.copy(), list(tracked))

            interval = self._settings.pipeline.result_interval_seconds
            now = time.monotonic()
            last = self._last_emit_time.get(cam_id, 0.0)
            if interval <= 0 or (now - last) >= interval:
                await self._publish_events(tracked, cam_id)
                self._last_emit_time[cam_id] = now

        return processed_any

    def _process_frame_sync(
        self,
        cam_id: str,
        frame: NDArray[np.uint8],
        frame_idx: int,
        do_reid: bool,
    ) -> list[TrackedObject]:
        """Detection + tracking + ReID in a worker thread (avoids PyTorch/ONNX deadlock)."""
        detections = self._detector.detect(frame)
        tracked = self._trackers[cam_id].update(detections)

        if do_reid and tracked:
            for obj in tracked:
                embedding = self._extractor.extract(frame, obj.bbox)
                if embedding is not None:
                    obj.embedding = embedding
            self._matcher.match(tracked, camera_id=cam_id)

        return tracked

    async def _publish_events(
        self,
        tracked: list[TrackedObject],
        camera_id: str,
    ) -> None:
        """Publish detection and re-identification events via NATS."""
        if self._nats is None:
            return

        for obj in tracked:
            det_event = ObjectDetectedEvent(
                camera_id=camera_id,
                track_id=obj.id,
                label=obj.label,
                confidence=obj.confidence,
                bbox=obj.bbox,
            )
            await self._nats.publish("detection", det_event.model_dump())

            if obj.global_id:
                cameras = self._registry.get_cameras_for_identity(obj.global_id)
                reid_event = ObjectReIdentifiedEvent(
                    global_id=obj.global_id,
                    camera_id=camera_id,
                    track_id=obj.id,
                    label=obj.label,
                    confidence=obj.confidence,
                    matched_cameras=cameras,
                )
                await self._nats.publish("reid", reid_event.model_dump())
