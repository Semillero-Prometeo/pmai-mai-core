"""Domain models for tracked objects, detections, cameras, and events."""

from pmai_core.domain.camera import CameraInfo, CameraStatus
from pmai_core.domain.detection import BBox, Detection
from pmai_core.domain.events import ObjectDetectedEvent, ObjectReIdentifiedEvent
from pmai_core.domain.tracked_object import TrackedObject

__all__ = [
    "BBox",
    "CameraInfo",
    "CameraStatus",
    "Detection",
    "ObjectDetectedEvent",
    "ObjectReIdentifiedEvent",
    "TrackedObject",
]
