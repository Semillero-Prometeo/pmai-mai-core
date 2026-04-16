"""USB microphone discovery helpers."""

from __future__ import annotations

import hashlib

import structlog

from pmai_core.resources.audio.models import MicrophoneInfo

logger = structlog.get_logger(__name__)

try:
    import sounddevice as sd
except ModuleNotFoundError:  # pragma: no cover - guarded at runtime
    sd = None


def _is_usb_like(name: str, hostapi_name: str) -> bool:
    blob = f"{name} {hostapi_name}".lower()
    return "usb" in blob


def _build_fingerprint(name: str, hostapi_name: str, channels: int, sample_rate: int) -> str:
    raw = f"{name}|{hostapi_name}|{channels}|{sample_rate}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()  # nosec B324


def discover_usb_microphones(preferred_id: str | None = None) -> list[MicrophoneInfo]:
    """Discover audio input devices; prioritize USB-like devices."""
    if sd is None:
        logger.warning("sounddevice_unavailable_for_microphone_discovery")
        return []

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
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
        microphone_id = f"mic_{idx}"
        info = MicrophoneInfo(
            microphone_id=microphone_id,
            name=name,
            hostapi_name=hostapi_name,
            device_index=idx,
            hardware_fingerprint=_build_fingerprint(name, hostapi_name, max_input, sample_rate),
            channels=max_input,
            sample_rate=sample_rate,
        )
        if _is_usb_like(name, hostapi_name):
            discovered.append(info)
        else:
            fallback.append(info)

    ordered = sorted(discovered, key=lambda item: item.microphone_id)
    if not ordered:
        ordered = sorted(fallback, key=lambda item: item.microphone_id)

    if preferred_id:
        ordered.sort(key=lambda item: item.microphone_id != preferred_id)

    return ordered
