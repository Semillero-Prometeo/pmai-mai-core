"""Camera subsystem – discovery, capture, and management of USB cameras."""

from pmai_core.camera.capture import CameraCapture
from pmai_core.camera.discovery import discover_usb_cameras
from pmai_core.camera.manager import CameraManager

__all__ = ["CameraCapture", "CameraManager", "discover_usb_cameras"]
