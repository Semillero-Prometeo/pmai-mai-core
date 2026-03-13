"""CameraManager – orchestrates discovery and lifecycle of all camera captures."""

from __future__ import annotations

import asyncio
from pathlib import Path
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
    """Manages the full lifecycle of USB cameras: discover -> capture -> poll.

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
        """Detect newly plugged / unplugged cameras without disturbing active ones.

        Instead of re-running full discovery (which would try to open devices
        already held by capture threads and fail on V4L2's single-open
        constraint), we check the device nodes in ``/dev`` directly and only
        run discovery for **new** nodes.
        """
        cam_settings = self._settings.camera

        # Cheap filesystem check: which /dev/video* nodes exist right now?
        current_dev_paths = {str(p) for p in sorted(Path("/dev").glob("video*"))}

        # Which device paths do we already manage?
        active_dev_paths = {
            info.device_path for info in self._camera_infos.values()
        }

        # --- Detect removed cameras (device node disappeared) ---
        for cam_id, info in list(self._camera_infos.items()):
            if info.device_path not in current_dev_paths:
                logger.info("camera_removed", camera_id=cam_id)
                self._stop_camera(cam_id)

        # --- Detect removed cameras (capture thread died) ---
        for cam_id, cap in list(self._captures.items()):
            if not cap.is_running:
                logger.warning("camera_capture_died", camera_id=cam_id)
                self._stop_camera(cam_id)

        # --- Detect new cameras (device node appeared, not yet managed) ---
        new_dev_paths = current_dev_paths - active_dev_paths
        if new_dev_paths:
            logger.info("new_device_nodes_detected", paths=sorted(new_dev_paths))
            new_cameras = discover_usb_cameras(
                default_resolution=cam_settings.default_resolution,
                default_fps=cam_settings.default_fps,
            )
            for info in new_cameras:
                if info.camera_id not in self._captures:
                    logger.info("new_camera_detected", camera_id=info.camera_id)
                    self._start_camera(info)

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
