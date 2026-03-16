"""PipelineEngine – orchestrates the main loop; identification lives in resources."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from pmai_core.domain.context_object import GlobalObjectForContext
from pmai_core.resources.identification import IdentificationService

if TYPE_CHECKING:
    from pmai_core.resources.camera.manager import CameraManager

logger = structlog.get_logger(__name__)


class PipelineEngine:
    """Runs the loop every [result_interval_seconds]: cameras -> identification -> global objects"""

    def __init__(
        self,
        camera_manager: CameraManager,
        identification_service: IdentificationService,
    ) -> None:
        self._camera_manager = camera_manager
        self._identification = identification_service
        self._running = False

    async def run(self) -> None:
        """Main loop: every [result_interval_seconds] run identification"""
        self._running = True

        logger.info(
            "pipeline_started",
            interval_seconds=self._identification._settings.pipeline.result_interval_seconds,
        )

        while self._running:
            await asyncio.sleep(self._identification._settings.pipeline.result_interval_seconds)

            captures = self._camera_manager.captures
            if not captures:
                logger.debug("no_captures_skipping")
                continue

            await self._identification.run_phase(captures)

            global_objects: list[GlobalObjectForContext] = (
                self._identification.get_global_objects_for_context()
            )
            if not global_objects:
                continue

            # Rest of flow (e.g. downstream phases consuming global_objects)
            logger.debug("context_objects_ready", count=len(global_objects))
            print(global_objects)

    def stop(self) -> None:
        self._running = False
        logger.info("pipeline_stopped")
