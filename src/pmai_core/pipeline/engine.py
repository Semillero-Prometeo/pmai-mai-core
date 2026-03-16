"""PipelineEngine – orchestrates the main loop; identification lives in resources."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import structlog
from numpy.typing import NDArray

from pmai_core.domain.context_object import GlobalObjectForContext
from pmai_core.domain.tracked_object import TrackedObject
from pmai_core.resources.identification import IdentificationService
from pmai_core.resources.reid.registry import GlobalRegistry
from pmai_core.resources.vision.tracker import ObjectTracker
from pmai_core.settings import Settings

if TYPE_CHECKING:
    from pmai_core.resources.camera.manager import CameraManager
    from pmai_core.messaging.client import NATSClient

logger = structlog.get_logger(__name__)


class PipelineEngine:
    """Runs the main loop: cameras -> identification phase -> global objects -> future phases."""

    def __init__(
        self,
        settings: Settings,
        camera_manager: CameraManager,
        nats_client: NATSClient | None = None,
    ) -> None:
        self._settings = settings
        self._camera_manager = camera_manager
        self._identification = IdentificationService(settings, nats_client)
        self._running = False

    @property
    def registry(self) -> GlobalRegistry:
        return self._identification.registry

    @property
    def trackers(self) -> dict[str, ObjectTracker]:
        return self._identification.trackers

    @property
    def all_last_annotated(
        self,
    ) -> dict[str, tuple[NDArray[np.uint8], list[TrackedObject]]]:
        return self._identification.all_last_annotated

    def get_last_annotated(
        self, camera_id: str
    ) -> tuple[NDArray[np.uint8], list[TrackedObject]] | None:
        return self._identification.get_last_annotated(camera_id)

    async def run(self) -> None:
        """Main loop: only the essential flow."""
        self._running = True
        logger.info("pipeline_started")

        while self._running:
            captures = self._camera_manager.captures
            if not captures:
                await asyncio.sleep(0.5)
                continue

            processed_any = await self._identification.run_phase(captures)

            global_objects: list[GlobalObjectForContext] = (
                self._identification.get_global_objects_for_context()
            )
            print(global_objects)

            if not processed_any:
                await asyncio.sleep(0.05)

    def stop(self) -> None:
        self._running = False
        logger.info("pipeline_stopped")
