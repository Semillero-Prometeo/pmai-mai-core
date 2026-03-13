"""Low-level detection primitives returned by the vision model."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    @property
    def center(self) -> tuple[int, int]:
        return (self.xmin + self.xmax) // 2, (self.ymin + self.ymax) // 2

    def to_xyxy(self) -> tuple[int, int, int, int]:
        return self.xmin, self.ymin, self.xmax, self.ymax


class Detection(BaseModel):
    """Single object detection from a frame."""

    bbox: BBox
    label: str
    class_id: int
    confidence: float = Field(ge=0.0, le=1.0)
