"""Strict request/response models for vision dashboard NATS RPC (schemaVersion 1)."""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

from pmai_core import __version__

_CAMEL = ConfigDict(
    alias_generator=to_camel,
    populate_by_name=True,
    serialize_by_alias=True,
)


class VisionSnapshotRequest(BaseModel):
    """Client payload for ``visionService.getSnapshot`` (camelCase from gateway/TS)."""

    model_config = {"populate_by_name": True}

    selected_camera_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("selected_camera_id", "selectedCameraId"),
    )
    thumb_max_width: int = Field(
        default=160,
        ge=32,
        le=640,
        validation_alias=AliasChoices("thumb_max_width", "thumbMaxWidth"),
    )
    preview_max_width: int = Field(
        default=640,
        ge=160,
        le=1920,
        validation_alias=AliasChoices("preview_max_width", "previewMaxWidth"),
    )

    @field_validator("selected_camera_id", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v: Any) -> str | None:
        if v is None or v == "":
            return None
        if isinstance(v, str):
            return v
        return str(v)


class CameraStreamTile(BaseModel):
    """One camera row in the vision snapshot."""

    model_config = _CAMEL

    camera_id: str
    source_type: str
    device_path: str
    name: str
    resolution: list[int]
    fps: int
    status: str
    has_frame: bool
    thumbnail_jpeg_base64: str | None = None
    preview_jpeg_base64: str | None = None


class ReidIdentitySummary(BaseModel):
    model_config = _CAMEL

    global_id: str
    cameras_seen: list[str]
    cross_camera: bool


class ReidSummary(BaseModel):
    model_config = _CAMEL

    total_identities: int
    cross_camera_identities: int
    identities: list[ReidIdentitySummary]


class VisionSnapshotResponse(BaseModel):
    """Unified dashboard payload for WebSocket / UI consumers."""

    model_config = _CAMEL

    schema_version: int = Field(default=1, ge=1)
    timestamp_ms: int
    version: str = __version__
    cameras: list[CameraStreamTile]
    global_objects: list[dict[str, Any]]
    reid_summary: ReidSummary
