"""PipelineEngine – orchestrates the full processing flow.

    cameras ➜ detect ➜ track ➜ extract embeddings ➜ cross-camera match ➜ publish
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
    """Core processing loop that ties every subsystem together.

    The engine iterates over all active camera captures in a round-robin
    fashion, runs detection ➜ tracking ➜ ReID on each frame, and publishes
    resulting events via NATS.
    """

    def __init__(
        self,
        settings: Settings,
        camera_manager: CameraManager,
        nats_client: NATSClient | None = None,
    ) -> None:
        self._settings = settings
        self._camera_manager = camera_manager
        self._nats = nats_client

        self._detector = YOLODetector(settings.vision)

        self._trackers: dict[str, ObjectTracker] = {}

        self._registry = GlobalRegistry(max_size=settings.reid.gallery_max_size)
        self._extractor = EmbeddingExtractor(settings.reid)
        self._matcher = CosineMatcher(
            registry=self._registry,
            similarity_threshold=settings.reid.similarity_threshold,
        )

        self._frame_counter: dict[str, int] = {}
        self._last_result_emit_time: float = 0.0
        self._last_annotated: dict[str, tuple[NDArray[np.uint8], list[TrackedObject]]] = {}
        self._running = False

    @property
    def registry(self) -> GlobalRegistry:
        return self._registry

    @property
    def trackers(self) -> dict[str, ObjectTracker]:
        return dict(self._trackers)

    def get_last_annotated(
        self, camera_id: str
    ) -> tuple[NDArray[np.uint8], list[TrackedObject]] | None:
        """Return the latest (frame, tracked_objects) for a camera, or None."""
        return self._last_annotated.get(camera_id)

    async def run(self) -> None:
        """Main async loop – process frames from all cameras continuously."""
        self._running = True
        logger.info("pipeline_started")

        while self._running:
            captures = self._camera_manager.captures
            if not captures:
                await asyncio.sleep(0.5)
                continue

            processed_any = False
            for cam_id, capture in captures.items():
                result = capture.get_frame(timeout=0.01)
                if result is None:
                    continue

                frame, timestamp = result
                processed_any = True

                if cam_id not in self._trackers:
                    self._trackers[cam_id] = ObjectTracker(camera_id=cam_id)
                self._frame_counter.setdefault(cam_id, 0)
                self._frame_counter[cam_id] += 1

                frame_index = self._frame_counter[cam_id]

                # --- Detection (YOLO) ---
                detections = await asyncio.to_thread(
                    self._detector.detect, frame,
                )

                if frame_index % 10 == 0:
                    logger.info(
                        "detection_summary",
                        camera_id=cam_id,
                        frame_index=frame_index,
                        num_detections=len(detections),
                        labels=[d.label for d in detections],
                    )

                tracked = self._trackers[cam_id].update(detections)

                if frame_index % 10 == 0:
                    logger.info(
                        "tracking_summary",
                        camera_id=cam_id,
                        frame_index=frame_index,
                        num_tracks=len(tracked),
                        track_ids=[obj.id for obj in tracked],
                    )

                reid_interval = self._settings.reid.embedding_update_interval
                do_reid = (
                    self._extractor.is_available
                    and self._frame_counter[cam_id] % reid_interval == 0
                )

                if do_reid:
                    await asyncio.to_thread(self._apply_reid, frame, tracked)

                    # After ReID has potentially assigned global IDs, log summary.
                    logger.info(
                        "reid_summary",
                        camera_id=cam_id,
                        frame_index=frame_index,
                        num_tracked=len(tracked),
                        with_identity=len(
                            [obj for obj in tracked if obj.global_id]
                        ),
                        identities=[
                            {
                                "track_id": obj.id,
                                "global_id": obj.global_id,
                            }
                            for obj in tracked
                            if obj.global_id
                        ],
                    )

                # Emit results (NATS + view state) only every result_interval_seconds.
                interval = self._settings.pipeline.result_interval_seconds
                now = time.monotonic()
                if interval <= 0 or (now - self._last_result_emit_time) >= interval:
                    await self._publish_events(tracked, cam_id)
                    self._last_result_emit_time = now
                    # Update last annotated for visualization (step 3).
                    self._last_annotated[cam_id] = (frame.copy(), list(tracked))

            if not processed_any:
                await asyncio.sleep(0.01)

    def stop(self) -> None:
        self._running = False
        logger.info("pipeline_stopped")

    def _apply_reid(
        self,
        frame: NDArray[np.uint8],
        tracked: list[TrackedObject],
    ) -> None:
        """Extract embeddings and run cross-camera matching."""
        for obj in tracked:
            embedding = self._extractor.extract(frame, obj.bbox)
            if embedding is not None:
                obj.embedding = embedding

        self._matcher.match(tracked)

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
