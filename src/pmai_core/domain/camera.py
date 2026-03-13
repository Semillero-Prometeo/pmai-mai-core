"""Camera-related domain models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class CameraStatus(StrEnum):
    DISCOVERED = "discovered"
    ACTIVE = "active"
    ERROR = "error"
    DISCONNECTED = "disconnected"


class CameraSourceType(StrEnum):
    """How we connect to the camera."""

    V4L2 = "v4l2"
    STREAM = "stream"
    FILE = "file"


class CameraInfo(BaseModel):
    """Metadata for a camera — USB, network stream, or local file."""

    camera_id: str = Field(description="Stable identifier")
    source_type: CameraSourceType = CameraSourceType.V4L2
    device_path: str = Field(
        default="",
        description="OS device path for V4L2, URL for stream, file path for file",
    )
    name: str = Field(default="", description="Human-readable name")
    resolution: tuple[int, int] = (640, 480)
    fps: int = 15
    loop: bool = Field(
        default=False,
        description="Loop the video file when it ends (only for 'file' sources)",
    )
    status: CameraStatus = CameraStatus.DISCOVERED
