"""Events published to NATS when objects are detected or re-identified."""

from __future__ import annotations

import time

from pydantic import BaseModel, Field

from pmai_core.domain.detection import BBox


class ObjectDetectedEvent(BaseModel):
    """Emitted for each new detection in a frame."""

    camera_id: str
    track_id: int
    label: str
    confidence: float
    bbox: BBox
    timestamp: float = Field(default_factory=time.time)


class ObjectReIdentifiedEvent(BaseModel):
    """Emitted when a cross-camera identity match is established."""

    global_id: str
    camera_id: str
    track_id: int
    label: str
    confidence: float
    matched_cameras: list[str] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)
