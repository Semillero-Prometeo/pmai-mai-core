"""USB microphone discovery helpers."""

from __future__ import annotations

import hashlib
from typing import Any

import structlog

from pmai_core.resources.audio.models import MicrophoneInfo

logger = structlog.get_logger(__name__)

try:
    import sounddevice as sd
except (ModuleNotFoundError, OSError):  # pragma: no cover - guarded at runtime
    sd = None


def _is_usb_like(name: str, hostapi_name: str) -> bool:
    blob = f"{name} {hostapi_name}".lower()
    return "usb" in blob


def build_fingerprint(name: str, hostapi_name: str, channels: int) -> str:
    raw = f"{name}|{hostapi_name}|{channels}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()  # nosec B324


def microphones_from_devices(
    devices: list[dict[str, Any]],
    hostapis: list[dict[str, Any]],
    preferred_id: str | None = None,
) -> list[MicrophoneInfo]:
    discovered: list[MicrophoneInfo] = []
    fallback: list[MicrophoneInfo] = []
    for idx, dev in enumerate(devices):
        max_input = int(dev.get("max_input_channels", 0) or 0)
        if max_input <= 0:
            continue
        host_idx = int(dev.get("hostapi", -1))
        hostapi_name = (
            str(hostapis[host_idx].get("name", "unknown"))
            if 0 <= host_idx < len(hostapis)
            else "unknown"
        )
        name = str(dev.get("name", f"Audio Device {idx}"))
        sample_rate = int(float(dev.get("default_samplerate", 16000.0)))
        fingerprint = build_fingerprint(name, hostapi_name, max_input)
        info = MicrophoneInfo(
            microphone_id=fingerprint,
            name=name,
            hostapi_name=hostapi_name,
            device_index=idx,
            hardware_fingerprint=fingerprint,
            channels=max_input,
            sample_rate=sample_rate,
        )
        if _is_usb_like(name, hostapi_name):
            discovered.append(info)
        else:
            fallback.append(info)

    ordered = discovered or fallback
    if preferred_id:
        ordered = sorted(ordered, key=lambda item: item.microphone_id != preferred_id)
    return ordered


def discover_usb_microphones(preferred_id: str | None = None) -> list[MicrophoneInfo]:
    """Discover audio input devices; prioritize USB-like devices."""
    if sd is None:
        logger.warning("sounddevice_unavailable_for_microphone_discovery")
        return []
    devices = list(sd.query_devices())
    hostapis = list(sd.query_hostapis())
    return microphones_from_devices(devices, hostapis, preferred_id)
