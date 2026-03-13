"""Typed application settings loaded from settings.toml and environment variables."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, model_validator

_DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "settings.toml"


class AppSettings(BaseModel):
    name: str = "pmai-core"
    log_level: str = "INFO"


class CameraSettings(BaseModel):
    auto_discover: bool = True
    poll_interval_seconds: int = 10
    default_resolution: tuple[int, int] = (640, 480)
    default_fps: int = 15


class VisionSettings(BaseModel):
    model_path: str = "models/yolo11n.onnx"
    confidence_threshold: float = 0.5
    device: str = "cpu"


class ReIDSettings(BaseModel):
    model_path: str = "models/osnet_ain_x0_25.onnx"
    similarity_threshold: float = 0.7
    embedding_update_interval: int = 5
    gallery_max_size: int = 1000


class NATSSettings(BaseModel):
    url: str = "nats://localhost:4222"
    subject_prefix: str = "pmai"


class APISettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class Settings(BaseModel):
    """Root settings container – merges TOML file values with defaults."""

    app: AppSettings = AppSettings()
    camera: CameraSettings = CameraSettings()
    vision: VisionSettings = VisionSettings()
    reid: ReIDSettings = ReIDSettings()
    nats: NATSSettings = NATSSettings()
    api: APISettings = APISettings()

    @classmethod
    def from_toml(cls, path: Path = _DEFAULT_SETTINGS_PATH) -> Self:
        if not path.exists():
            return cls()
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls.model_validate(data)

    @model_validator(mode="before")
    @classmethod
    def _coerce_resolution(cls, values: dict) -> dict:  # type: ignore[override]
        """Allow resolution as a list in TOML while storing as a tuple."""
        cam = values.get("camera")
        if isinstance(cam, dict):
            res = cam.get("default_resolution")
            if isinstance(res, list) and len(res) == 2:
                cam["default_resolution"] = tuple(res)
        return values
