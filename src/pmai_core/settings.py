"""Typed application settings loaded from settings.toml and environment variables."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, model_validator

_DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "settings.toml"


class AppSettings(BaseModel):
    name: str = "pmai-core"
    log_level: str = "INFO"


class CameraSourceConfig(BaseModel):
    """Manual camera source definition (stream or file)."""

    id: str
    type: str = "stream"
    url: str = ""
    path: str = ""
    resolution: tuple[int, int] = (640, 480)
    fps: int = 15
    loop: bool = False


class CameraSettings(BaseModel):
    source_mode: Literal["v4l2", "video_folder"] = "v4l2"
    auto_discover: bool = True
    poll_interval_seconds: int = 10
    default_resolution: tuple[int, int] = (640, 480)
    default_fps: int = 15
    sources: list[CameraSourceConfig] = []
    video_folder_path: str = "/data"
    video_extensions: list[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    video_loop: bool = True
    video_fps: int = 15

    @model_validator(mode="after")
    def _normalize_video_extensions(self) -> "CameraSettings":
        normalized: list[str] = []
        for ext in self.video_extensions:
            value = ext.strip().lower()
            if not value:
                continue
            if not value.startswith("."):
                value = f".{value}"
            normalized.append(value)
        self.video_extensions = normalized
        return self


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
    """Event subjects use ``subject_prefix``; RPC uses ``ms_name`` (e.g. ``PMAI_CORE.healthService.health``)."""

    url: str = "nats://localhost:4222"
    subject_prefix: str = "pmai"
    ms_name: str = "MAI_CORE_MS"

    @model_validator(mode="before")
    @classmethod
    def _override_url_from_env(cls, values: dict) -> dict:  # type: ignore[override]
        """Allow `NATS_SERVER` env var to override the URL.

        This is primarily used in Docker, where the broker is reachable via
        the `nats-server` service name instead of `localhost`.
        """
        env_url = os.getenv("NATS_SERVER")
        if env_url:
            values = dict(values) if isinstance(values, dict) else {}
            values["url"] = env_url
        return values


class PipelineSettings(BaseModel):
    """Configuration for the main processing loop."""

    # Interval in seconds between emitting results (NATS + view state).
    # Camera stays open; only the delivery of detections/ReID is throttled.
    result_interval_seconds: float = 5.0


class AudioSettings(BaseModel):
    """Configuration for background microphone listening and transcription."""

    enabled: bool = True
    auto_start_listening: bool = True
    auto_select_microphone: bool = True
    preferred_microphone_id: str | None = None

    sample_rate: int = 16000
    channels: int = 1
    chunk_ms: int = 30
    long_silence_ms: int = 1600
    post_wake_grace_ms: int = 1800
    max_recording_seconds: float = 12.0
    min_recording_seconds: float = 0.8
    preroll_seconds: float = 0.4
    min_useful_transcript_chars: int = 14
    fallback_speak_message: str = "No te escuche con claridad, puedes repetirlo por favor."

    wake_phrases: list[str] = ["hey fede", "hola fede"]
    wake_partial_match: bool = True

    vosk_model_path: str = "models/vosk-model-small-es-0.42"
    poll_interval_seconds: float = 5.0

    @model_validator(mode="after")
    def _normalize_audio_settings(self) -> AudioSettings:
        preferred = (self.preferred_microphone_id or "").strip()
        self.preferred_microphone_id = preferred or None
        self.wake_phrases = [p.strip().lower() for p in self.wake_phrases if p.strip()]
        self.min_useful_transcript_chars = max(1, self.min_useful_transcript_chars)
        return self


class Settings(BaseModel):
    """Root settings container – merges TOML file values with defaults."""

    app: AppSettings = AppSettings()
    camera: CameraSettings = CameraSettings()
    vision: VisionSettings = VisionSettings()
    reid: ReIDSettings = ReIDSettings()
    nats: NATSSettings = NATSSettings()
    pipeline: PipelineSettings = PipelineSettings()
    audio: AudioSettings = AudioSettings()

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
            for src in cam.get("sources", []):
                if isinstance(src, dict):
                    src_res = src.get("resolution")
                    if isinstance(src_res, list) and len(src_res) == 2:
                        src["resolution"] = tuple(src_res)
        return values
