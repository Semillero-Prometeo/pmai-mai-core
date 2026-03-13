"""Auto-discovery of USB cameras via /dev/video* and V4L2 validation."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import cv2
import structlog

from pmai_core.domain.camera import CameraInfo, CameraStatus

logger = structlog.get_logger(__name__)


def _parse_v4l2_devices() -> dict[str, str]:
    """Run ``v4l2-ctl --list-devices`` and map device paths to human names.

    Returns a mapping like ``{"/dev/video0": "USB Camera (usb-0000:00:14.0-1)"}``.
    Falls back to an empty dict when the command is unavailable.
    """
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return {}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}

    devices: dict[str, str] = {}
    current_name = ""
    for line in result.stdout.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if not line.startswith("\t") and not line.startswith(" "):
            current_name = line.rstrip(":")
        else:
            dev_path = line.strip()
            if dev_path.startswith("/dev/video"):
                devices[dev_path] = current_name
    return devices


def _is_capture_device(device_path: str) -> bool:
    """Check if the V4L2 device supports video capture (not just metadata)."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--device", device_path, "--all"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return "Video Capture" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True


def _can_open_with_opencv(device_path: str) -> bool:
    """Validate that OpenCV can open the device.

    Only checks ``isOpened()`` — does NOT try to read a frame, because in
    WSL2 / Docker the first frame can take 10+ seconds to arrive and
    would cause a false-negative timeout.
    """
    idx_match = re.search(r"\d+$", device_path)
    if idx_match is None:
        return False
    idx = int(idx_match.group())

    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    opened = cap.isOpened()
    cap.release()

    if not opened:
        logger.debug("opencv_cannot_open", device=device_path)
    return opened


def discover_usb_cameras(
    default_resolution: tuple[int, int] = (640, 480),
    default_fps: int = 15,
) -> list[CameraInfo]:
    """Scan the system for USB cameras and return validated ``CameraInfo`` list.

    Strategy
    --------
    1. Enumerate ``/dev/video*`` device nodes.
    2. Filter through V4L2 to keep only capture-capable devices.
    3. Validate each with ``cv2.VideoCapture.isOpened()``.
    """
    video_devices = sorted(Path("/dev").glob("video*"))

    if not video_devices:
        logger.info("no_video_devices_found")
        return []

    v4l2_names = _parse_v4l2_devices()
    cameras: list[CameraInfo] = []

    for dev in video_devices:
        dev_str = str(dev)

        if not _is_capture_device(dev_str):
            logger.debug("skipping_non_capture_device", device=dev_str)
            continue

        if not _can_open_with_opencv(dev_str):
            continue

        idx_match = re.search(r"\d+$", dev_str)
        idx = idx_match.group() if idx_match else dev.name
        camera_id = f"usb_{idx}"

        cameras.append(
            CameraInfo(
                device_path=dev_str,
                camera_id=camera_id,
                name=v4l2_names.get(dev_str, f"USB Camera {idx}"),
                resolution=default_resolution,
                fps=default_fps,
                status=CameraStatus.DISCOVERED,
            )
        )
        logger.info("camera_discovered", camera_id=camera_id, device=dev_str)

    return cameras
