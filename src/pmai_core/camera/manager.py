"""CameraManager – orchestrates discovery and lifecycle of all camera captures."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from pmai_core.camera.capture import CameraCapture
from pmai_core.camera.discovery import discover_usb_cameras
from pmai_core.domain.camera import CameraInfo, CameraSourceType, CameraStatus
from pmai_core.settings import Settings

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)


class CameraManager:
    """Manages the full lifecycle of cameras: discover -> capture -> poll.

    Supports three kinds of sources:
    - **V4L2** (auto-discovered): USB cameras via ``/dev/video*``
    - **stream** (manual config): RTSP, HTTP MJPEG, etc.
    - **file** (manual config): local video files for development/testing
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
        """Discover/configure cameras and start capture threads."""
        started: list[CameraInfo] = []

        # 1. Manually configured sources (streams, files, explicit V4L2)
        for src in self._settings.camera.sources:
            source_type = CameraSourceType(src.type)
            device_path = src.url if source_type == CameraSourceType.STREAM else src.path
            if source_type == CameraSourceType.V4L2 and src.path:
                device_path = src.path

            info = CameraInfo(
                camera_id=src.id,
                source_type=source_type,
                device_path=device_path,
                name=f"Manual: {src.id}",
                resolution=src.resolution,
                fps=src.fps,
                loop=src.loop,
            )
            if self._start_camera(info):
                started.append(info)

        # 2. V4L2 auto-discovery (skip devices already covered by manual config)
        if self._settings.camera.auto_discover:
            cam_settings = self._settings.camera
            infos = discover_usb_cameras(
                default_resolution=cam_settings.default_resolution,
                default_fps=cam_settings.default_fps,
            )
            for info in infos:
                if info and info.camera_id not in self._captures and self._start_camera(info):
                    started.append(info)

        return started

    def stop(self) -> None:
        for cap in self._captures.values():
            cap.stop()
        self._captures.clear()
        self._camera_infos.clear()
        logger.info("camera_manager_stopped")

    def poll_for_changes(self) -> None:
        """Detect newly plugged / unplugged V4L2 cameras.

        Does not affect manually configured sources (stream/file).
        """
        cam_settings = self._settings.camera

        # Only check capture thread health for non-V4L2 sources.
        for cam_id, cap in list(self._captures.items()):
            if not cap.is_running:
                info = self._camera_infos.get(cam_id)
                src_type = info.source_type if info else "unknown"
                logger.warning(
                    "camera_capture_died",
                    camera_id=cam_id,
                    source_type=src_type,
                )
                self._stop_camera(cam_id)

        # V4L2 hot-plug detection.
        if not cam_settings.auto_discover:
            return

        current_dev_paths = {str(p) for p in sorted(Path("/dev").glob("video*"))}
        active_v4l2_paths = {
            info.device_path
            for info in self._camera_infos.values()
            if info.source_type == CameraSourceType.V4L2
        }

        # Detect removed V4L2 cameras.
        for cam_id, info in list(self._camera_infos.items()):
            is_removed = (
                info.source_type == CameraSourceType.V4L2
                and info.device_path not in current_dev_paths
            )
            if is_removed:
                logger.info("camera_removed", camera_id=cam_id)
                self._stop_camera(cam_id)

        # Detect new V4L2 cameras.
        new_dev_paths = current_dev_paths - active_v4l2_paths
        if new_dev_paths:
            logger.info("new_device_nodes_detected", paths=sorted(new_dev_paths))
            new_cameras = discover_usb_cameras(
                default_resolution=cam_settings.default_resolution,
                default_fps=cam_settings.default_fps,
                only_paths=new_dev_paths,
                skip_paths=active_v4l2_paths,
            )
            for info in new_cameras:
                if info.camera_id in self._captures:
                    continue
                logger.info("new_camera_detected", camera_id=info.camera_id)
                self._start_camera(info)

    async def run_polling_loop(self) -> None:
        """Async loop that periodically checks for camera changes."""
        interval = self._settings.camera.poll_interval_seconds
        while True:
            await asyncio.sleep(interval)
            self.poll_for_changes()

    def _start_camera(self, info: CameraInfo) -> bool:
        capture = CameraCapture(info)
        try:
            capture.start()
            info.status = CameraStatus.ACTIVE
        except (RuntimeError, ValueError) as exc:
            logger.error(
                "camera_start_failed",
                camera_id=info.camera_id,
                source_type=info.source_type,
                error=str(exc),
            )
            info.status = CameraStatus.ERROR
            return False
        self._captures[info.camera_id] = capture
        self._camera_infos[info.camera_id] = info
        return True

    def _stop_camera(self, camera_id: str) -> None:
        cap = self._captures.pop(camera_id, None)
        if cap is not None:
            cap.stop()
        info = self._camera_infos.pop(camera_id, None)
        if info is not None:
            info.status = CameraStatus.DISCONNECTED
