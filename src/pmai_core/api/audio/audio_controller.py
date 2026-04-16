"""Microphone listing and background listener control over NATS."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from pmai_core import __version__
from pmai_core.resources.audio.service import AudioService


class SelectMicrophonePayload(BaseModel):
    microphone_id: str


class AudioController:
    def __init__(self, audio_service: AudioService) -> None:
        self._audio_service = audio_service

    async def list_microphones(self, _: dict[str, Any]) -> dict[str, Any]:
        microphones, selected = self._audio_service.list_microphones()
        return {
            "version": __version__,
            "count": len(microphones),
            "selected_microphone_id": selected,
            "microphones": [
                {
                    "microphone_id": mic.microphone_id,
                    "name": mic.name,
                    "hostapi_name": mic.hostapi_name,
                    "device_index": mic.device_index,
                    "hardware_fingerprint": mic.hardware_fingerprint,
                    "channels": mic.channels,
                    "sample_rate": mic.sample_rate,
                    "status": mic.status.value,
                    "is_selected": mic.microphone_id == selected,
                }
                for mic in microphones
            ],
        }

    async def select_microphone(self, data: dict[str, Any]) -> dict[str, Any]:
        try:
            payload = SelectMicrophonePayload.model_validate(data)
        except ValidationError as e:
            raise ValueError(str(e)) from e

        selected = self._audio_service.select_microphone(payload.microphone_id)
        return {"status": "ok", "selected_microphone_id": selected}

    async def start_listening(self, _: dict[str, Any]) -> dict[str, Any]:
        self._audio_service.start_listening()
        return {"status": "ok", "message": "Background listening started"}

    async def stop_listening(self, _: dict[str, Any]) -> dict[str, Any]:
        self._audio_service.stop_listening()
        return {"status": "ok", "message": "Background listening stopped"}

    async def get_listening_status(self, _: dict[str, Any]) -> dict[str, Any]:
        status = self._audio_service.get_status()
        return {
            "running": status.running,
            "listening_enabled": status.listening_enabled,
            "selected_microphone_id": status.selected_microphone_id,
            "wake_word_detected_at": status.wake_word_detected_at,
            "last_transcript": status.last_transcript,
            "last_transcript_at": status.last_transcript_at,
            "last_error": status.last_error,
        }
