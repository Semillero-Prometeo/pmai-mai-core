"""PipelineEngine – orchestrates the full processing flow.

cameras -> detect -> track -> extract embeddings -> cross-camera match -> publish
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np
import structlog
from numpy.typing import NDArray

from pmai_core.domain.events import ObjectDetectedEvent, ObjectReIdentifiedEvent
from pmai_core.domain.tracked_object import TrackedObject
from pmai_core.reid.extractor import EmbeddingExtractor
from pmai_core.reid.matcher import CosineMatcher
from pmai_core.reid.registry import GlobalRegistry
from pmai_core.settings import Settings
from pmai_core.vision.detector import YOLODetector
from pmai_core.vision.tracker import ObjectTracker

if TYPE_CHECKING:
    from pmai_core.camera.manager import CameraManager
    from pmai_core.messaging.client import NATSClient

logger = structlog.get_logger(__name__)


class PipelineEngine:
    """Core processing loop that ties every subsystem together."""

    def __init__(
        self,
        settings: Settings,
        camera_manager: CameraManager,
        nats_client: NATSClient | None = None,
    ) -> None:
        self._settings = settings
        self._camera_manager = camera_manager
        self._nats = nats_client

        self._trackers: dict[str, ObjectTracker] = {}

        # ONNX Runtime (ReID) MUST be initialised before PyTorch (YOLO)
        # to avoid OpenMP thread-pool deadlocks between the two runtimes.
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
        self._running = False

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
        """Return annotated state for all cameras."""
        return dict(self._last_annotated)

    def get_last_annotated(
        self, camera_id: str
    ) -> tuple[NDArray[np.uint8], list[TrackedObject]] | None:
        """Return the latest (frame, tracked_objects) for a camera, or None."""
        return self._last_annotated.get(camera_id)

    async def run(self) -> None:
        """Main async loop -- process frames from all cameras continuously."""
        self._running = True
        logger.info("pipeline_started")

        while self._running:
            captures = self._camera_manager.captures
            if not captures:
                await asyncio.sleep(0.5)
                continue

            processed_any = False
            for cam_id, capture in captures.items():
                result = capture.get_frame(timeout=0.05)
                if result is None:
                    continue

                frame, timestamp = result
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

                # if frame_idx == 1 or frame_idx % 10 == 0:
                #     self._log_pipeline_summary(cam_id, frame_idx, tracked)

                interval = self._settings.pipeline.result_interval_seconds
                now = time.monotonic()
                last = self._last_emit_time.get(cam_id, 0.0)
                if interval <= 0 or (now - last) >= interval:
                    await self._publish_events(tracked, cam_id)
                    self._last_emit_time[cam_id] = now

            if not processed_any:
                await asyncio.sleep(0.01)

    def stop(self) -> None:
        self._running = False
        logger.info("pipeline_stopped")

    def _process_frame_sync(
        self,
        cam_id: str,
        frame: NDArray[np.uint8],
        frame_idx: int,
        do_reid: bool,
    ) -> list[TrackedObject]:
        """Run detection + tracking + ReID synchronously in a worker thread.

        All CPU-bound inference (YOLO via PyTorch and ReID via ONNX Runtime)
        runs inside a single thread to avoid cross-library thread-pool deadlocks.
        """
        # t0 = time.monotonic()
        detections = self._detector.detect(frame)
        # t1 = time.monotonic()
        tracked = self._trackers[cam_id].update(detections)
        # t2 = time.monotonic()

        if do_reid and tracked:
            for obj in tracked:
                embedding = self._extractor.extract(frame, obj.bbox)
                if embedding is not None:
                    obj.embedding = embedding
            # t3 = time.monotonic()
            self._matcher.match(tracked, camera_id=cam_id)
            # t4 = time.monotonic()
            # logger.debug(
            #     "frame_timings",
            #     camera_id=cam_id,
            #     frame_idx=frame_idx,
            #     yolo_ms=round((t1 - t0) * 1000),
            #     track_ms=round((t2 - t1) * 1000),
            #     embed_ms=round((t3 - t2) * 1000),
            #     match_ms=round((t4 - t3) * 1000),
            # )

        return tracked

    def _log_pipeline_summary(
        self,
        cam_id: str,
        frame_idx: int,
        tracked: list[TrackedObject],
    ) -> None:
        with_gid = [o for o in tracked if o.global_id]
        logger.info(
            "pipeline_summary",
            camera_id=cam_id,
            frame_idx=frame_idx,
            detections=len(tracked),
            with_global_id=len(with_gid),
            objects=[
                {
                    "track_id": o.id,
                    "label": o.label,
                    "conf": round(o.confidence, 2),
                    "global_id": o.global_id or "-",
                    "cameras_seen": (
                        self._registry.get_cameras_for_identity(o.global_id) if o.global_id else []
                    ),
                }
                for o in tracked
            ],
            total_identities=self._registry.size,
        )

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
