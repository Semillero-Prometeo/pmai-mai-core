"""Camera-related domain models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class CameraStatus(StrEnum):
    DISCOVERED = "discovered"
    ACTIVE = "active"
    ERROR = "error"
    DISCONNECTED = "disconnected"


class CameraInfo(BaseModel):
    """Metadata for a discovered USB camera."""

    device_path: str = Field(description="OS device path, e.g. /dev/video0")
    camera_id: str = Field(description="Stable identifier derived from device")
    name: str = Field(default="", description="Human-readable name from V4L2")
    resolution: tuple[int, int] = (640, 480)
    fps: int = 15
    status: CameraStatus = CameraStatus.DISCOVERED
