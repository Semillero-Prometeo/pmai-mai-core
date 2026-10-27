from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class MicrophoneStatus(StrEnum):
    AVAILABLE = "available"
    DISCONNECTED = "disconnected"


class ListenPhase(StrEnum):
    DISCONNECTED = "disconnected"
    SILENT = "silent"
    HEARING = "hearing"
    CAPTURING = "capturing"
    ERROR = "error"


@dataclass(slots=True)
class MicrophoneInfo:
    microphone_id: str
    name: str
    hostapi_name: str
    device_index: int
    hardware_fingerprint: str
    channels: int
    sample_rate: int
    status: MicrophoneStatus = MicrophoneStatus.AVAILABLE


@dataclass(slots=True)
class AudioRuntimeStatus:
    running: bool
    listening_enabled: bool
    selected_microphone_id: str | None
    selected_microphone_name: str | None
    stream_open: bool
    phase: ListenPhase
    input_level: int
    wake_word_detected_at: str | None
    last_transcript: str | None
    last_transcript_at: str | None
    last_error: str | None
