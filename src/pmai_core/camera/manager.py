"""CameraManager – orchestrates discovery and lifecycle of all camera captures."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import structlog

from pmai_core.camera.capture import CameraCapture
from pmai_core.camera.discovery import discover_usb_cameras
from pmai_core.domain.camera import CameraStatus
from pmai_core.settings import Settings

if TYPE_CHECKING:
    from pmai_core.domain.camera import CameraInfo

logger = structlog.get_logger(__name__)


class CameraManager:
    """Manages the full lifecycle of USB cameras: discover ➜ capture ➜ poll.

    Call :meth:`start` once to discover cameras and begin capturing.
    :meth:`poll_for_changes` can be scheduled periodically to hot-detect
    newly plugged or removed cameras.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._captures: dict[str, CameraCapture] = {}
        self._camera_infos: dict[str, CameraInfo] = {}

    @property
    def cameras(self) -> dict[str, CameraInfo]:
        return dict(self._camera_infos)

    @property
    def captures(self) -> dict[str, CameraCapture]:
        return dict(self._captures)

    def start(self) -> list[CameraInfo]:
        """Discover cameras and start capture threads. Returns discovered list."""
        cam_settings = self._settings.camera
        infos = discover_usb_cameras(
            default_resolution=cam_settings.default_resolution,
            default_fps=cam_settings.default_fps,
        )
        for info in infos:
            self._start_camera(info)
        return infos

    def stop(self) -> None:
        for cap in self._captures.values():
            cap.stop()
        self._captures.clear()
        self._camera_infos.clear()
        logger.info("camera_manager_stopped")

    def poll_for_changes(self) -> None:
        """Re-scan USB devices and start/stop cameras as needed."""
        cam_settings = self._settings.camera
        current = discover_usb_cameras(
            default_resolution=cam_settings.default_resolution,
            default_fps=cam_settings.default_fps,
        )
        current_ids = {c.camera_id for c in current}
        existing_ids = set(self._captures.keys())

        # Start newly discovered cameras
        for info in current:
            if info.camera_id not in existing_ids:
                logger.info("new_camera_detected", camera_id=info.camera_id)
                self._start_camera(info)

        # Stop removed cameras
        for cam_id in existing_ids - current_ids:
            logger.info("camera_removed", camera_id=cam_id)
            self._stop_camera(cam_id)

    async def run_polling_loop(self) -> None:
        """Async loop that periodically checks for camera changes."""
        interval = self._settings.camera.poll_interval_seconds
        while True:
            await asyncio.sleep(interval)
            self.poll_for_changes()

    def _start_camera(self, info: CameraInfo) -> None:
        capture = CameraCapture(info)
        try:
            capture.start()
            info.status = CameraStatus.ACTIVE
        except RuntimeError:
            logger.error("camera_start_failed", camera_id=info.camera_id)
            info.status = CameraStatus.ERROR
            return
        self._captures[info.camera_id] = capture
        self._camera_infos[info.camera_id] = info

    def _stop_camera(self, camera_id: str) -> None:
        cap = self._captures.pop(camera_id, None)
        if cap is not None:
            cap.stop()
        info = self._camera_infos.pop(camera_id, None)
        if info is not None:
            info.status = CameraStatus.DISCONNECTED
